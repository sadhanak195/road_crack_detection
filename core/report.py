"""Report generation module for Road Crack Detection."""

from typing import Dict, Any, Optional
from datetime import datetime, timezone
import json
import cv2
import numpy as np

import config


def generate_json_report(
    source_name: str,
    detection_result: Dict[str, Any],
    gradcam_result: Optional[Dict[str, Any]] = None,
    vlm_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compile structured inspection report as a JSON-serializable dictionary."""
    report = {
        "metadata": {
            "system": "Road Crack Detection System (YOLO26s + Grad-CAM + Qwen2-VL)",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source": source_name,
            "device": config.DEVICE,
            "image_size": {
                "width": detection_result.get("image_size", (0, 0))[0],
                "height": detection_result.get("image_size", (0, 0))[1],
            },
        },
        "detection_summary": {
            "thresholds": detection_result.get("thresholds", {}),
            "total_cracks_detected": detection_result.get("total_cracks", 0),
            "class_counts": detection_result.get("class_counts", {}),
            "average_confidence": detection_result.get("average_confidence", 0.0),
            "detections": detection_result.get("detections", []),
        },
    }

    if gradcam_result:
        report["gradcam_xai"] = {
            "layer_id": gradcam_result.get("layer_id"),
            "layer_name": gradcam_result.get("layer_name"),
            "alpha": gradcam_result.get("alpha"),
            "caption": gradcam_result.get("caption"),
        }

    if vlm_result:
        report["vlm_assessment"] = {
            "model_used": vlm_result.get("model_used"),
            "1_vlm_blind_prediction": vlm_result.get("vlm_blind"),
            "2_rule_based_severity": vlm_result.get("rule_severity_text"),
            "3_yolo_prediction_check": vlm_result.get("yolo_check"),
            "4_recommended_repair_action": vlm_result.get("recommended_action"),
            "vlm_agrees_with_yolo": vlm_result.get("agrees_with_yolo", True),
            "success": vlm_result.get("success", False),
        }

    return report


def encode_image_to_png_bytes(image_bgr: np.ndarray) -> bytes:
    """Encode OpenCV BGR image to PNG byte buffer."""
    success, encoded = cv2.imencode(".png", image_bgr)
    if not success:
        raise ValueError("Failed to encode image to PNG.")
    return encoded.tobytes()


def encode_json_to_bytes(report_dict: Dict[str, Any]) -> bytes:
    """Encode report dictionary to pretty-printed UTF-8 JSON bytes."""
    return json.dumps(report_dict, indent=2).encode("utf-8")
