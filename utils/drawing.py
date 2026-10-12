"""Drawing and visualization utilities for Road Crack Detection."""

from typing import List, Dict, Any, Tuple
import cv2
import numpy as np

import config


def get_adaptive_scale(width: int, height: int) -> Tuple[int, float, int]:
    """Calculate adaptive line thickness, font scale, and padding based on image dimensions."""
    dim = max(width, height)
    thickness = max(2, int(round(dim / 350)))
    font_scale = max(0.45, dim / 900)
    padding = max(3, int(thickness * 1.5))
    return thickness, font_scale, padding


def draw_detections(
    image_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
) -> np.ndarray:
    """Draw bounding boxes and class labels with high contrast on image."""
    canvas = image_bgr.copy()
    h, w = canvas.shape[:2]
    thickness, font_scale, padding = get_adaptive_scale(w, h)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        box = det["box"]  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        class_name = det["class"]
        confidence = det["conf"]
        class_id = det.get("class_id", -1)

        # Get color
        color = config.CLASS_COLORS.get(class_id, (0, 255, 255))

        # Draw box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        # Text label
        label = f"{class_name} {confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness=max(1, thickness - 1))

        # Position label above box if space permits, else inside
        if y1 - text_h - padding * 2 > 0:
            lbl_y1 = y1 - text_h - padding * 2
            lbl_y2 = y1
            text_pos = (x1 + padding, y1 - padding - baseline // 2)
        else:
            lbl_y1 = y1
            lbl_y2 = y1 + text_h + padding * 2
            text_pos = (x1 + padding, y1 + text_h + padding)

        lbl_x1 = x1
        lbl_x2 = min(w, x1 + text_w + padding * 2)

        # Draw filled background rectangle for label
        cv2.rectangle(canvas, (lbl_x1, lbl_y1), (lbl_x2, lbl_y2), color, -1)

        # Draw contrasting text (black or white depending on luminance)
        b, g, r = color
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)
        cv2.putText(
            canvas,
            label,
            text_pos,
            font,
            font_scale,
            text_color,
            thickness=max(1, thickness - 1),
            lineType=cv2.LINE_AA,
        )

    return canvas
