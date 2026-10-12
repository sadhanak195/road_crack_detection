"""Comprehensive end-to-end test suite for Road Crack Detection System."""

import sys
from pathlib import Path
from PIL import Image
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from core.image_io import (
    load_image_from_file_path,
    load_image_from_url,
    load_image_from_bytes,
    ImageProcessingError,
)
from core.detector import predict_cracks
from core.gradcam import generate_gradcam
from core.vlm import analyze_with_vlm
from core.report import (
    generate_json_report,
    encode_image_to_png_bytes,
    encode_json_to_bytes,
)
from utils.logging_setup import get_logger

logger = get_logger("test_end_to_end")


def test_upload_flow():
    """Verify local image loading and end-to-end pipeline."""
    sample_path = PROJECT_ROOT / "sample_images" / "pothole_sample.jpg"
    logger.info(f"Testing Upload Flow with {sample_path}...")

    with open(sample_path, "rb") as f:
        raw_bytes = f.read()

    pil_img, rgb, bgr = load_image_from_bytes(raw_bytes)
    assert pil_img is not None
    assert bgr is not None and bgr.shape[2] == 3

    # Run detection
    det_res = predict_cracks(bgr, conf=0.15)
    logger.info(f"Detections: {det_res['total_cracks']}, class counts: {det_res['class_counts']}")

    # Run Grad-CAM
    cam_res = generate_gradcam(bgr, conf=0.15, layer_id=16, alpha=0.45)
    assert cam_res["overlay_bgr"] is not None
    assert cam_res["raw_heatmap"] is not None

    # Run VLM (4-step sequence)
    vlm_res = analyze_with_vlm(pil_img, det_res["detections"], force_fallback=True)
    assert "vlm_blind" in vlm_res
    assert "rule_severity_text" in vlm_res
    assert "yolo_check" in vlm_res
    assert "recommended_action" in vlm_res

    # Generate JSON report
    report = generate_json_report("pothole_sample.jpg", det_res, cam_res, vlm_res)
    assert report["detection_summary"]["total_cracks_detected"] == det_res["total_cracks"]

    # Verify PNG and JSON encoders
    annotated_png = encode_image_to_png_bytes(det_res["annotated_bgr"])
    cam_png = encode_image_to_png_bytes(cam_res["overlay_bgr"])
    json_bytes = encode_json_to_bytes(report)

    assert len(annotated_png) > 1000
    assert len(cam_png) > 1000
    assert len(json_bytes) > 100
    logger.info("Upload flow passed successfully.")


def test_url_flow():
    """Verify URL loading and graceful error handling on invalid URLs."""
    logger.info("Testing URL Flow...")

    # 1. Test error on invalid URL
    try:
        load_image_from_url("https://invalid-non-existent-domain-12345.xyz/image.jpg")
        assert False, "Should have raised ImageProcessingError"
    except ImageProcessingError as e:
        logger.info(f"Correctly caught invalid URL error: {e}")

    # 2. Test error on empty URL
    try:
        load_image_from_url("")
        assert False, "Should have raised ImageProcessingError"
    except ImageProcessingError as e:
        logger.info(f"Correctly caught empty URL error: {e}")


def test_empty_detections_handling():
    """Verify that images with 0 detections do not crash Grad-CAM or VLM."""
    logger.info("Testing zero-detection handling...")
    # Create blank gray road asphalt image
    blank_bgr = np.full((300, 300, 3), 120, dtype=np.uint8)
    pil_blank = Image.fromarray(blank_bgr)

    # Use very high confidence to guarantee 0 detections
    det_res = predict_cracks(blank_bgr, conf=0.99)
    assert det_res["total_cracks"] == 0
    assert det_res["has_detections"] is False

    # Grad-CAM should still generate safely
    cam_res = generate_gradcam(blank_bgr, conf=0.99, layer_id=16)
    assert cam_res["overlay_bgr"] is not None

    # VLM should still generate safely (confirming road is intact)
    vlm_res = analyze_with_vlm(pil_blank, det_res["detections"], force_fallback=True)
    assert "vlm_blind" in vlm_res
    logger.info(f"Zero detection VLM blind prediction: {vlm_res['vlm_blind']}")

    report = generate_json_report("blank_road.jpg", det_res, cam_res, vlm_res)
    assert report["detection_summary"]["total_cracks_detected"] == 0
    logger.info("Zero-detection test passed successfully.")


if __name__ == "__main__":
    test_upload_flow()
    test_url_flow()
    test_empty_detections_handling()
    print("ALL END-TO-END TESTS PASSED SUCCESSFULLY!")
