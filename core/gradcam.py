"""Grad-CAM / HiResCAM explainability (XAI) implementation for YOLO26s."""

from typing import Dict, Tuple, Optional, Any
from pathlib import Path
import cv2
import numpy as np
import torch

import config
from core.detector import register_custom_modules
from utils.logging_setup import get_logger

logger = get_logger("gradcam")

EXPLAINABILITY_CAPTION = (
    "Grad-CAM Heatmap Explanation: Warmer regions (red, orange, yellow) indicate image areas "
    "that contributed most strongly to the model's crack detection activations. "
    "Cooler regions (blue, dark blue) had little or no influence on the decision."
)


class YoloGradCAM:
    """HiResCAM / Grad-CAM visualizer for YOLO neck features."""

    def __init__(
        self,
        weights_path: Optional[str | Path] = None,
        layer_ids: Tuple[int, ...] = (16, 19, 22),
        imgsz: int = config.IMGSZ,
        device: str = config.DEVICE,
    ):
        register_custom_modules()
        from ultralytics import YOLO

        self.imgsz = imgsz
        self.layer_ids = layer_ids
        self.device = device
        path = Path(weights_path) if weights_path else config.MODEL_PATH

        logger.info(f"Initializing YoloGradCAM with model {path} on {device}...")
        self.net = YOLO(str(path)).model.float().to(device).eval()
        self.acts: Dict[int, torch.Tensor] = {}
        self.grads: Dict[int, torch.Tensor] = {}
        self._hook_handles = []

        for i in layer_ids:
            if i < len(self.net.model):
                handle = self.net.model[i].register_forward_hook(self._make_hook(i))
                self._hook_handles.append(handle)
            else:
                logger.warning(f"Layer ID {i} is out of bounds for model with {len(self.net.model)} layers.")

    def _make_hook(self, i: int):
        def hook(module, inp, out):
            self.acts[i] = out
            if out.requires_grad:
                out.register_hook(lambda g, i=i: self.grads.__setitem__(i, g))
        return hook

    def _letterbox(self, bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        h, w = bgr.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        dw, dh = (self.imgsz - nw) / 2.0, (self.imgsz - nh) / 2.0
        img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return img, (top, left, nh, nw)

    @staticmethod
    def _scores(out: Any) -> torch.Tensor:
        """Extract classification / confidence scores from YOLO forward output."""
        tensors = []

        def walk(o):
            if torch.is_tensor(o):
                tensors.append(o)
            elif isinstance(o, (list, tuple)):
                for x in o:
                    walk(x)
            elif isinstance(o, dict):
                for x in o.values():
                    walk(x)

        walk(out)
        for t in tensors:
            # End-to-end format: [B, N, 6] (x1, y1, x2, y2, score, cls)
            if t.dim() == 3 and t.shape[-1] == 6:
                return t[0, :, 4]

        num_classes = len(config.CLASS_NAMES)
        for t in tensors:
            # Dense anchor format: [B, 4 + nc, N]
            if t.dim() == 3 and t.shape[1] == 4 + num_classes:
                return t[0, 4:, :].max(0).values

        # Fallback to the largest 3D tensor score slice
        for t in tensors:
            if t.dim() == 3 and t.shape[1] >= 5:
                return t[0, 4:, :].max(0).values

        raise RuntimeError(f"Unrecognized YOLO output shapes for Grad-CAM: {[tuple(t.shape) for t in tensors]}")

    def __call__(
        self,
        bgr: np.ndarray,
        conf: float = config.DEFAULT_CONF,
        border_px: int = 16,
    ) -> Tuple[Dict[int, np.ndarray], int]:
        h, w = bgr.shape[:2]
        lb, (top, left, nh, nw) = self._letterbox(bgr)
        x = (
            torch.from_numpy(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB))
            .permute(2, 0, 1)
            .float()
            .div(255.0)[None]
            .to(self.device)
        )
        x.requires_grad_(True)

        self.acts.clear()
        self.grads.clear()
        self.net.zero_grad(set_to_none=True)

        with torch.enable_grad():
            out = self.net(x)
            scores = self._scores(out)
            keep = scores >= conf
            target = scores[keep].sum() if keep.any() else scores.max()

            if not target.requires_grad:
                raise RuntimeError("Grad-CAM target is not differentiable.")
            target.backward()

        cams: Dict[int, np.ndarray] = {}
        for i in self.layer_ids:
            g = self.grads.get(i)
            a = self.acts.get(i)
            if g is None or a is None:
                cams[i] = np.zeros((h, w), dtype=np.float32)
                continue

            # HiResCAM: element-wise gradient * activation keeps spatial location
            cam = torch.relu((g * a).sum(1))[0].detach().cpu().numpy().astype(np.float32)

            # Remove border artifacts (~16 input px converted to cells for this stride)
            stride = config.GRADCAM_STRIDES.get(i, 8)
            b = int(np.ceil(border_px / stride))
            cam[:b, :] = 0
            cam[-b:, :] = 0
            cam[:, :b] = 0
            cam[:, -b:] = 0

            # Clip upper percentile outliers
            hi = np.percentile(cam, 99.5)
            if hi > 0:
                cam = np.clip(cam, 0, hi)

            # Un-letterbox and resize back to original image dimensions
            cam = cv2.resize(cam, (self.imgsz, self.imgsz))[top : top + nh, left : left + nw]
            cam = cv2.resize(cam, (w, h))

            # Normalize to [0, 1]
            max_val = cam.max()
            cams[i] = cam / (max_val + 1e-8) if max_val > 0 else cam

        self.acts.clear()
        self.grads.clear()
        return cams, int(keep.sum().item())

    def cleanup(self):
        """Remove hooks and clear memory."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self.acts.clear()
        self.grads.clear()


_CACHED_GRADCAM_ENGINE: Optional[YoloGradCAM] = None


def get_gradcam_engine(weights_path: Optional[str | Path] = None) -> YoloGradCAM:
    """Get or create singleton YoloGradCAM engine."""
    global _CACHED_GRADCAM_ENGINE
    if _CACHED_GRADCAM_ENGINE is None:
        _CACHED_GRADCAM_ENGINE = YoloGradCAM(weights_path=weights_path)
    return _CACHED_GRADCAM_ENGINE


def overlay_heatmap(
    image_bgr: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = config.DEFAULT_GRADCAM_ALPHA,
) -> np.ndarray:
    """Create JET colored heatmap overlay blended with original image."""
    heat_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1.0 - alpha, heat_colored, alpha, 0)


def generate_gradcam(
    image_bgr: np.ndarray,
    conf: float = config.DEFAULT_CONF,
    layer_id: int = config.DEFAULT_GRADCAM_LAYER,
    alpha: float = config.DEFAULT_GRADCAM_ALPHA,
    weights_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Generate Grad-CAM heatmap and overlay for input image.

    Returns dict containing:
        - 'overlay_bgr': overlay image in BGR
        - 'overlay_rgb': overlay image in RGB
        - 'raw_heatmap': 2D float32 array in [0.0, 1.0]
        - 'heatmap_colored_bgr': standalone colored heatmap BGR
        - 'heatmap_colored_rgb': standalone colored heatmap RGB
        - 'layer_id': target layer used
        - 'layer_name': name of the target layer
        - 'alpha': alpha value used
        - 'caption': explanation caption
        - 'all_cams': dict of raw heatmaps for all configured layers
    """
    engine = get_gradcam_engine(weights_path)
    cams, n_keep = engine(image_bgr, conf=conf)

    cam_layer = cams.get(layer_id)
    if cam_layer is None:
        # Fallback to first available layer
        layer_id = next(iter(cams.keys()))
        cam_layer = cams[layer_id]

    overlay_bgr = overlay_heatmap(image_bgr, cam_layer, alpha=alpha)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    heat_colored_bgr = cv2.applyColorMap(np.uint8(255 * cam_layer), cv2.COLORMAP_JET)
    heat_colored_rgb = cv2.cvtColor(heat_colored_bgr, cv2.COLOR_BGR2RGB)

    layer_name = config.GRADCAM_LAYERS.get(layer_id, f"Layer {layer_id}")

    return {
        "overlay_bgr": overlay_bgr,
        "overlay_rgb": overlay_rgb,
        "raw_heatmap": cam_layer,
        "heatmap_colored_bgr": heat_colored_bgr,
        "heatmap_colored_rgb": heat_colored_rgb,
        "layer_id": layer_id,
        "layer_name": layer_name,
        "alpha": alpha,
        "caption": EXPLAINABILITY_CAPTION,
        "all_cams": cams,
        "n_keep": n_keep,
    }
