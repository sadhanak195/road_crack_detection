"""Test script for Phase 4: VLM analysis and 4-step verification."""

import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from core.image_io import load_image_from_file_path
from core.detector import predict_cracks
from core.vlm import (
    compute_rule_severity,
    compute_overall_severity,
    build_explanation_prompt,
    parse_explanation_lines,
    analyze_with_vlm,
)
from utils.logging_setup import get_logger

logger = get_logger("test_phase4")


def test_vlm_phase():
    sample_img_path = PROJECT_ROOT / "sample_images" / "transverse_crack_sample.jpg"
    assert sample_img_path.exists(), f"Sample image not found at {sample_img_path}"

    logger.info("Loading sample image...")
    pil_img, rgb, bgr = load_image_from_file_path(str(sample_img_path))
    w, h = pil_img.size

    logger.info("Getting detections...")
    det_res = predict_cracks(bgr, conf=0.20)
    dets = det_res["detections"]
    logger.info(f"Got {len(dets)} detections.")

    # 1. Test rule severity calculation
    overall_sev, reason = compute_overall_severity(dets, w, h)
    logger.info(f"Computed rule-based severity: {overall_sev} ({reason})")
    assert overall_sev in ["none", "low", "medium", "high"]

    # 2. Test prompt building
    prompt = build_explanation_prompt(dets, overall_sev)
    logger.info(f"Built Pass B prompt:\n{prompt}")
    assert "Severity, computed from detection size by fixed rules:" in prompt

    # 3. Test parser on sample Pass B lines
    sample_pass_b = (
        "1. Visible damage: Longitudinal crack extending along the center lane with minor branching.\n"
        "2. Detector check: Agree with YOLO longitudinal crack detection at middle-center.\n"
        "3. Recommended action: High urgency - Rout and apply hot-pour polymer sealant to prevent water infiltration."
    )
    v_dam, d_chk, r_act = parse_explanation_lines(sample_pass_b, overall_sev)
    assert "longitudinal crack" in v_dam.lower()
    assert "agree" in d_chk.lower()
    assert "hot-pour" in r_act.lower()

    # 4. Test analyze_with_vlm 4-step structure
    res = analyze_with_vlm(pil_img, dets, force_fallback=True)
    assert "vlm_blind" in res and res["vlm_blind"] != ""
    assert "rule_severity_text" in res and res["rule_severity_text"] != ""
    assert "yolo_check" in res and res["yolo_check"] != ""
    assert "recommended_action" in res and res["recommended_action"] != ""

    logger.info(f"1. VLM Blind Prediction: {res['vlm_blind']}")
    logger.info(f"2. Rule-Based Severity:  {res['rule_severity_text']}")
    logger.info(f"3. YOLO Check:           {res['yolo_check']}")
    logger.info(f"4. Recommended Action:   {res['recommended_action']}")

    print("PHASE 4 CHECKPOINT PASSED: 4-step sequence (Blind VLM, Rule Severity, YOLO Check, Action) verified successfully.")


if __name__ == "__main__":
    test_vlm_phase()
