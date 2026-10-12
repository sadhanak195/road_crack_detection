"""Test script for Phase 3: Grad-CAM / HiResCAM verification."""

import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from core.image_io import load_image_from_file_path
from core.gradcam import generate_gradcam
from utils.logging_setup import get_logger

logger = get_logger("test_phase3")


def test_gradcam_phase():
    sample_img_path = PROJECT_ROOT / "sample_images" / "transverse_crack_sample.jpg"
    assert sample_img_path.exists(), f"Sample image not found at {sample_img_path}"

    logger.info(f"Loading sample image from {sample_img_path}...")
    pil_img, rgb, bgr = load_image_from_file_path(str(sample_img_path))
    h, w = bgr.shape[:2]

    logger.info("Generating Grad-CAM for Layer 16 (P3 fine)...")
    res16 = generate_gradcam(bgr, conf=0.20, layer_id=16, alpha=0.45)

    assert res16["raw_heatmap"].shape == (h, w), f"Heatmap shape {res16['raw_heatmap'].shape} != {(h, w)}"
    assert res16["overlay_bgr"].shape == (h, w, 3), f"Overlay shape {res16['overlay_bgr'].shape} != {(h, w, 3)}"
    assert 0.0 <= res16["raw_heatmap"].min() <= 1.0
    assert 0.0 <= res16["raw_heatmap"].max() <= 1.0
    assert res16["raw_heatmap"].max() > 0.0, "Heatmap has no activations!"

    out_dir = PROJECT_ROOT / "tests" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    overlay_path = out_dir / "phase3_gradcam_overlay_layer16.jpg"
    heatmap_path = out_dir / "phase3_gradcam_heat_only.png"
    cv2.imwrite(str(overlay_path), res16["overlay_bgr"])
    cv2.imwrite(str(heatmap_path), res16["heatmap_colored_bgr"])

    logger.info(f"Saved Grad-CAM overlay to: {overlay_path}")
    logger.info(f"Saved Colored heatmap to: {heatmap_path}")
    logger.info(f"Heatmap stats: min={res16['raw_heatmap'].min():.4f}, max={res16['raw_heatmap'].max():.4f}, mean={res16['raw_heatmap'].mean():.4f}")

    # Also test layer 19 and 22
    for lid in [19, 22]:
        res_l = generate_gradcam(bgr, conf=0.20, layer_id=lid, alpha=0.45)
        path_l = out_dir / f"phase3_gradcam_layer{lid}.jpg"
        cv2.imwrite(str(path_l), res_l["overlay_bgr"])
        logger.info(f"Saved Layer {lid} overlay to: {path_l}")

    print("PHASE 3 CHECKPOINT PASSED: Grad-CAM overlay successfully produced and verified.")


if __name__ == "__main__":
    test_gradcam_phase()
