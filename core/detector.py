"""YOLO detector module with custom Strip-Conv blocks support."""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import collections
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

import config
from utils.drawing import draw_detections
from utils.logging_setup import get_logger

logger = get_logger("detector")

# ---------------------------------------------------------
# Custom Strip-Convolution Modules (required for unpickling)
# ---------------------------------------------------------
class StripConvBlock(nn.Module):
    """Residual strip-convolution block for long, thin structures such as cracks."""
    def __init__(self, c: int, k: int = 11):
        super().__init__()
        p = k // 2
        self.dw_local = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.dw_h = nn.Conv2d(c, c, (1, k), padding=(0, p), groups=c, bias=False)
        self.dw_v = nn.Conv2d(c, c, (k, 1), padding=(p, 0), groups=c, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU()
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        nn.init.zeros_(self.pw.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw_local(x) + self.dw_h(x) + self.dw_v(x)
        return x + self.pw(self.act(self.bn(y)))


class NeckStrip(nn.Module):
    """Wraps an existing neck block and applies the strip block to its output."""
    def __init__(self, base: nn.Module, c: int, k: int):
        super().__init__()
        self.base = base
        self.strip = StripConvBlock(c, k)
        self.f, self.i, self.type, self.np = base.f, base.i, base.type, base.np

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.strip(self.base(x))


def register_custom_modules():
    """Ensure custom modules are present in __main__ and sys.modules."""
    import __main__
    __main__.StripConvBlock = StripConvBlock
    __main__.NeckStrip = NeckStrip


# Call once at module import
register_custom_modules()

_CACHED_DETECTOR: Optional[YOLO] = None


def get_detector(weights_path: Optional[str | Path] = None) -> YOLO:
    """Load and cache the YOLO detector model."""
    global _CACHED_DETECTOR
    register_custom_modules()
    path = Path(weights_path) if weights_path else config.MODEL_PATH
    if _CACHED_DETECTOR is None:
        logger.info(f"Loading YOLO weights from {path}...")
        _CACHED_DETECTOR = YOLO(str(path))
        # Ensure correct class names are mapped
        _CACHED_DETECTOR.model.names = config.CLASS_NAMES
    return _CACHED_DETECTOR


def where(box: Tuple[float, float, float, float], width: int, height: int) -> str:
    """Classify the spatial region of a bounding box inside the image (from notebook)."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0 / max(1, width)
    cy = (y1 + y2) / 2.0 / max(1, height)

    v = "top" if cy < 1.0 / 3.0 else ("bottom" if cy > 2.0 / 3.0 else "middle")
    h = "left" if cx < 1.0 / 3.0 else ("right" if cx > 2.0 / 3.0 else "center")
    return f"{v}-{h}"


def predict_cracks(
    image_bgr: np.ndarray,
    conf: float = config.DEFAULT_CONF,
    iou: float = config.DEFAULT_IOU,
    weights_path: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Run crack detection on input BGR image.

    Returns:
        dict containing:
            - 'annotated_bgr': annotated image in BGR
            - 'annotated_rgb': annotated image in RGB
            - 'detections': list of detection dictionaries
            - 'class_counts': dict of counts per class
            - 'total_cracks': total count of detections
            - 'average_confidence': mean confidence of detections
            - 'has_detections': bool indicating if cracks were found
    """
    model = get_detector(weights_path)
    height, width = image_bgr.shape[:2]

    # Predict using Ultralytics API
    results = model.predict(
        source=image_bgr,
        conf=conf,
        iou=iou,
        imgsz=config.IMGSZ,
        device=device or config.DEVICE,
        verbose=False,
    )[0]

    detections: List[Dict[str, Any]] = []
    class_counts = collections.defaultdict(int)

    for b in results.boxes:
        cls_id = int(b.cls[0].item()) if hasattr(b.cls, "__len__") else int(b.cls.item())
        confidence = float(b.conf[0].item()) if hasattr(b.conf, "__len__") else float(b.conf.item())
        coords = [float(v) for v in b.xyxy[0]]
        x1, y1, x2, y2 = coords
        class_name = config.CLASS_NAMES.get(cls_id, str(cls_id))

        area_pct = 100.0 * max(0.0, (x2 - x1) * (y2 - y1)) / float(max(1, width * height))
        pos = where((x1, y1, x2, y2), width, height)

        det = {
            "class": class_name,
            "class_id": cls_id,
            "conf": confidence,
            "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "position": pos,
            "area_pct": round(area_pct, 2),
        }
        detections.append(det)
        class_counts[class_name] += 1

    # Sort largest area first
    detections.sort(key=lambda d: d["area_pct"], reverse=True)

    # Draw custom annotated bounding boxes
    annotated_bgr = draw_detections(image_bgr, detections)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    total_cracks = len(detections)
    avg_conf = (
        round(sum(d["conf"] for d in detections) / total_cracks, 3)
        if total_cracks > 0
        else 0.0
    )

    return {
        "annotated_bgr": annotated_bgr,
        "annotated_rgb": annotated_rgb,
        "detections": detections,
        "class_counts": dict(class_counts),
        "total_cracks": total_cracks,
        "average_confidence": avg_conf,
        "has_detections": total_cracks > 0,
        "image_size": (width, height),
        "thresholds": {"conf": conf, "iou": iou},
    }
