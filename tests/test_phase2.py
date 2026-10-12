"""Test script for Phase 2: image_io and detector verification."""

import sys
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from core.image_io import load_image_from_file_path
from core.detector import predict_cracks
from utils.logging_setup import get_logger

logger = get_logger("test_phase2")


def test_detection_phase():
    sample_img_path = PROJECT_ROOT / "sample_images" / "transverse_crack_sample.jpg"
    assert sample_img_path.exists(), f"Sample image not found at {sample_img_path}"

    logger.info(f"Loading sample image from {sample_img_path}...")
    pil_img, rgb, bgr = load_image_from_file_path(str(sample_img_path))
    assert bgr is not None and bgr.shape[0] > 0 and bgr.shape[1] > 0

    logger.info(f"Running crack detection (conf=0.20)...")
    result = predict_cracks(bgr, conf=0.20)

    logger.info(f"Detections found: {result['total_cracks']}")
    logger.info(f"Class counts: {result['class_counts']}")
    logger.info(f"Average confidence: {result['average_confidence']}")

    for i, d in enumerate(result["detections"]):
        logger.info(f"  [{i}] {d['class']} ({d['conf']:.2f}) at {d['box']}, pos: {d['position']}, area: {d['area_pct']}%")

    # Save output annotated image
    out_dir = PROJECT_ROOT / "tests" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase2_annotated_output.jpg"
    cv2.imwrite(str(out_path), result["annotated_bgr"])
    assert out_path.exists()
    logger.info(f"Annotated output successfully saved to: {out_path}")

    print("PHASE 2 CHECKPOINT PASSED: Detection and annotation completed successfully.")


if __name__ == "__main__":
    test_detection_phase()
