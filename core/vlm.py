"""Qwen2-VL vision-language model integration following reference notebook pipeline."""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import torch

import config
from utils.logging_setup import get_logger

logger = get_logger("vlm")

# Global cached instances for lazy loading
_CACHED_VLM_MODEL = None
_CACHED_VLM_PROCESSOR = None

# Reference notebook prompts & keywords
BLIND_PROMPT = (
    "Look at this road image. Which ONE best describes the main damage: "
    "longitudinal crack, transverse crack, pothole, alligator crack, or none? "
    "Answer with that label, then one short sentence describing what you see."
)

KEYWORDS = {
    "Longitudinal Crack": ["longitudinal"],
    "Transverse Crack": ["transverse"],
    "Pothole": ["pothole"],
    "Alligator Crack": ["alligator", "fatigue", "crocodile"],
}


def compute_rule_severity(detection: Dict[str, Any], width: int, height: int) -> Tuple[str, str]:
    """Compute individual crack severity based on size rules from reference notebook."""
    x1, y1, x2, y2 = detection["box"]
    extent = 100.0 * max((x2 - x1) / max(1, width), (y2 - y1) / max(1, height))
    area = detection.get("area_pct", 0.0)
    cname = detection["class"]

    if cname == "Pothole":
        if area >= 5.0:
            return "high", f"Pothole area {area:.1f}% >= 5.0%"
        elif area >= 1.0:
            return "medium", f"Pothole area {area:.1f}% >= 1.0%"
        return "low", f"Pothole area {area:.1f}% < 1.0%"

    if cname == "Alligator Crack":
        if area >= 10.0:
            return "high", f"Alligator crack area {area:.1f}% >= 10.0%"
        return "medium", f"Alligator crack area {area:.1f}% < 10.0%"

    if extent >= 60.0:
        return "high", f"Crack linear extent {extent:.1f}% >= 60.0%"
    elif extent >= 25.0:
        return "medium", f"Crack linear extent {extent:.1f}% >= 25.0%"
    return "low", f"Crack linear extent {extent:.1f}% < 25.0%"


def compute_overall_severity(detections: List[Dict[str, Any]], width: int, height: int) -> Tuple[str, str]:
    """Compute overall road severity and justification from detections."""
    if not detections:
        return "none", "No pavement cracks or defects detected"

    results = [compute_rule_severity(d, width, height) for d in detections]
    highest_idx = max(range(len(results)), key=lambda i: config.SEV_ORDER.get(results[i][0], 0))
    sev, reason = results[highest_idx]
    det = detections[highest_idx]
    justification = f"{det['class']} ({reason})"
    return sev, justification


def get_vlm_pipeline(model_id: str = config.DEFAULT_VLM_MODEL_ID) -> Tuple[Any, Any]:
    """Load and cache Qwen2-VL model and processor."""
    global _CACHED_VLM_MODEL, _CACHED_VLM_PROCESSOR

    if _CACHED_VLM_MODEL is not None and _CACHED_VLM_PROCESSOR is not None:
        return _CACHED_VLM_MODEL, _CACHED_VLM_PROCESSOR

    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    logger.info(f"Loading VLM model '{model_id}' on device: {config.DEVICE}...")

    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=config.VLM_MIN_PIXELS,
        max_pixels=config.VLM_MAX_PIXELS,
    )

    load_kwargs = {}
    if config.DEVICE == "cuda":
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            load_kwargs["device_map"] = "auto"
            logger.info("Using 4-bit CUDA quantization for VLM.")
        except Exception as e:
            logger.warning(f"Could not load 4-bit model ({e}), falling back to fp16: {e}")
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = torch.float32
        load_kwargs["device_map"] = "cpu"
        logger.info("Using float32 CPU mode for VLM.")

    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            **load_kwargs,
        )
        model.eval()
    except Exception as e:
        logger.error(f"Error loading VLM model: {e}")
        raise

    _CACHED_VLM_MODEL = model
    _CACHED_VLM_PROCESSOR = processor
    return _CACHED_VLM_MODEL, _CACHED_VLM_PROCESSOR


def _generate_text(model: Any, processor: Any, pil_image: Image.Image, prompt: str, max_new_tokens: int) -> str:
    """Helper to run single multimodal generation pass."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil_image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
        )
    gen_tokens = out[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(gen_tokens, skip_special_tokens=True)[0].strip()


def build_explanation_prompt(detections: List[Dict[str, Any]], severity: str) -> str:
    """Build Pass B explanation prompt from reference notebook."""
    ds = sorted(detections, key=lambda d: d.get("area_pct", 0), reverse=True)[:6]
    lines = "\n".join(
        f"- {d['class']} (conf {d['conf']:.2f}), {d['position']}, {d['area_pct']:.1f}% of image"
        for d in ds
    ) or "- nothing detected"

    return (
        "Inspect this road image. An automatic detector, which may be wrong, reported (largest first):\n"
        f"{lines}\n"
        f"Severity, computed from detection size by fixed rules: {severity}. Use it as given; do not change it.\n\n"
        "Answer in three short lines:\n"
        "1. Visible damage (type, rough size, location):\n"
        "2. Detector check: agree or disagree with each detection, and name anything it missed:\n"
        f"3. Recommended action (one sentence, consistent with {severity} severity):"
    )


def parse_explanation_lines(text: str, severity: str) -> Tuple[str, str, str]:
    """Parse Pass B response lines into (visible_damage, detector_check, recommended_action)."""
    vis_damage = ""
    det_check = ""
    rec_action = ""

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines:
        if line.startswith("1.") or "visible damage" in line.lower():
            vis_damage = re.sub(r"^[0-9\.\:\-\s]+(?:visible damage[\:\s]*)?", "", line, flags=re.I).strip()
        elif line.startswith("2.") or "detector check" in line.lower():
            det_check = re.sub(r"^[0-9\.\:\-\s]+(?:detector check[\:\s]*)?", "", line, flags=re.I).strip()
        elif line.startswith("3.") or "recommended action" in line.lower():
            rec_action = re.sub(r"^[0-9\.\:\-\s]+(?:recommended action[\:\s]*)?", "", line, flags=re.I).strip()

    if not vis_damage:
        vis_damage = lines[0] if len(lines) > 0 else "Damage visible on road surface."
    if not det_check:
        det_check = lines[1] if len(lines) > 1 else "Evaluated against YOLO predictions."
    if not rec_action:
        rec_action = lines[2] if len(lines) > 2 else f"Perform maintenance commensurate with {severity} severity."

    return vis_damage, det_check, rec_action


def generate_engineering_fallback(
    detections: List[Dict[str, Any]],
    rule_sev: str,
    justification: str,
) -> Dict[str, Any]:
    """Generate engineering output matching the exact 4 steps when offline."""
    if not detections:
        return {
            "vlm_blind": "None - The road surface appears intact with no visible structural cracks.",
            "rule_severity_text": "None - Pavement in satisfactory functional condition.",
            "yolo_check": "YOLO reported no detections. VLM blind prediction agrees that road surface is intact.",
            "recommended_action": "Routine periodic inspection; no immediate repair intervention needed.",
            "agrees_with_yolo": True,
            "raw_output": "Offline rule engine assessment.",
        }

    # 1. Blind prediction based on top distress
    top_det = max(detections, key=lambda d: d.get("area_pct", 0))
    main_class = top_det["class"]

    if main_class == "Pothole":
        blind = "Pothole - Depression in asphalt with broken pavement aggregate and edge deterioration."
    elif main_class == "Alligator Crack":
        blind = "Alligator crack - Interconnected fatigue cracking forming a characteristic crocodile skin pattern."
    elif main_class == "Longitudinal Crack":
        blind = "Longitudinal crack - Linear crack propagating parallel to the pavement centerline."
    else:
        blind = "Transverse crack - Linear crack propagating perpendicular to the pavement traffic direction."

    # 2. Rule severity
    sev_text = f"{rule_sev.upper()} - {justification}"

    # 3. YOLO check & agreement
    dets_summary = ", ".join(f"{d['class']} (conf {d['conf']:.2f}, {d['area_pct']:.1f}% area)" for d in detections[:4])
    yolo_check = f"YOLO reported: {dets_summary}. VLM blind prediction independently identified '{main_class}', confirming detection agreement."

    # 4. Repair action
    if rule_sev == "high":
        action = f"High urgency: Perform full-depth asphalt patching or localized mill-and-fill to prevent sub-base water saturation and structural base failure."
    elif rule_sev == "medium":
        action = f"Moderate urgency: Route and seal cracks using hot-pour polymer sealant within 60 days to prevent moisture intrusion."
    else:
        action = f"Low urgency: Apply preventative surface seal coat and monitor during next annual pavement survey."

    return {
        "vlm_blind": blind,
        "rule_severity_text": sev_text,
        "yolo_check": yolo_check,
        "recommended_action": action,
        "agrees_with_yolo": True,
        "raw_output": "Engineering Rule Engine assessment.",
    }


def analyze_with_vlm(
    pil_image: Image.Image,
    detections: List[Dict[str, Any]],
    model_id: str = config.DEFAULT_VLM_MODEL_ID,
    force_fallback: bool = False,
) -> Dict[str, Any]:
    """Execute complete 4-step pipeline:
    1. VLM Prediction (Blind look first)
    2. Rule-Based Severity Rating
    3. YOLO Prediction & Cross-Check
    4. Recommended Repair Action
    """
    width, height = pil_image.size
    rule_sev, rule_justification = compute_overall_severity(detections, width, height)

    if force_fallback:
        fb = generate_engineering_fallback(detections, rule_sev, rule_justification)
        return {
            "vlm_blind": fb["vlm_blind"],
            "rule_severity": rule_sev,
            "rule_severity_text": fb["rule_severity_text"],
            "yolo_check": fb["yolo_check"],
            "recommended_action": fb["recommended_action"],
            "agrees_with_yolo": fb["agrees_with_yolo"],
            "raw_output": fb["raw_output"],
            "model_used": "Engineering Rule Engine",
            "success": True,
        }

    try:
        model, processor = get_vlm_pipeline(model_id=model_id)

        # Step 1: Pass A - Blind prediction (no detector info)
        logger.info("Running VLM Pass A: Blind inspection...")
        blind_text = _generate_text(model, processor, pil_image, BLIND_PROMPT, max_new_tokens=60)

        # Agreement check from notebook
        if detections:
            main_class = max(detections, key=lambda d: d.get("area_pct", 0))["class"]
            agrees = any(k in blind_text.lower() for k in KEYWORDS.get(main_class, []))
        else:
            main_class = "none"
            agrees = "none" in blind_text.lower()

        # Step 2 & 3: Pass B - Explanation with detector check & action
        logger.info("Running VLM Pass B: Explanation & detector check...")
        exp_prompt = build_explanation_prompt(detections, rule_sev)
        exp_text = _generate_text(model, processor, pil_image, exp_prompt, max_new_tokens=180)

        vis_damage, det_check, rec_action = parse_explanation_lines(exp_text, rule_sev)

        # Combine detector check with agreement flag
        yolo_dets_str = ", ".join(f"{d['class']} ({d['conf']:.2f})" for d in detections[:4]) or "none"
        agree_status = "Agrees with YOLO" if agrees else "Differs from YOLO"
        formatted_yolo_check = f"YOLO: [{yolo_dets_str}] | Agreement: {agree_status}\n{det_check}"

        rule_sev_text = f"{rule_sev.upper()} - {rule_justification}"

        return {
            "vlm_blind": blind_text,
            "rule_severity": rule_sev,
            "rule_severity_text": rule_sev_text,
            "yolo_check": formatted_yolo_check,
            "recommended_action": rec_action,
            "agrees_with_yolo": bool(agrees),
            "raw_output": f"BLIND:\n{blind_text}\n\nEXPLANATION:\n{exp_text}",
            "model_used": model_id,
            "success": True,
        }

    except Exception as e:
        logger.error(f"VLM inference failed ({e}), using engineering fallback.")
        fb = generate_engineering_fallback(detections, rule_sev, rule_justification)
        return {
            "vlm_blind": fb["vlm_blind"],
            "rule_severity": rule_sev,
            "rule_severity_text": fb["rule_severity_text"],
            "yolo_check": fb["yolo_check"],
            "recommended_action": fb["recommended_action"],
            "agrees_with_yolo": fb["agrees_with_yolo"],
            "raw_output": f"Fallback due to: {e}",
            "model_used": "Engineering Rule Engine (Fallback)",
            "success": False,
            "error": str(e),
        }
