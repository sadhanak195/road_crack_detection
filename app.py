import spaces
from pathlib import Path
import cv2
import gradio as gr
import numpy as np

import config
from core.detector import predict_cracks
from core.gradcam import generate_gradcam
from core.vlm import analyze_with_vlm

@spaces.GPU
def inspect_road(
    image,
    conf,
    iou,
    enable_gradcam,
    layer_id,
    cam_alpha,
    enable_vlm,
    offline_mode,
):
    if image is None:
        raise gr.Error("Please upload a road pavement image first.")

    # Gradio supplies a PIL RGB image.
    image = image.convert("RGB")
    image_rgb = np.array(image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    try:
        # Step 1: crack/pothole detection
        det = predict_cracks(
            image_bgr=image_bgr,
            conf=float(conf),
            iou=float(iou),
        )

        # Step 2: optional Grad-CAM
        cam = None
        if enable_gradcam:
            cam = generate_gradcam(
                image_bgr=image_bgr,
                conf=float(conf),
                layer_id=int(layer_id),
                alpha=float(cam_alpha),
            )

        # Step 3: optional VLM assessment
        vlm = None
        if enable_vlm:
            vlm = analyze_with_vlm(
                pil_image=image,
                detections=det["detections"],
                force_fallback=bool(offline_mode),
            )

        class_breakdown = det["class_counts"] or {}
        summary = [
            "## Road Inspection Results",
            f"- **Total detections:** {det['total_cracks']}",
            f"- **Average confidence:** "
            f"{det['average_confidence']:.3f}"
            if det["total_cracks"] else "- **Average confidence:** N/A",
            "- **Class breakdown:** " + (
                ", ".join(
                    f"{name}: {count}"
                    for name, count in class_breakdown.items()
                ) or "No defects detected"
            ),
        ]

        if not det["has_detections"]:
            summary.append(
                "\nNo defects were detected at this confidence threshold. "
                "You can try lowering the threshold."
            )

        if vlm:
            summary.extend([
                "\n## Structural Assessment",
                f"**Assessment engine:** {vlm.get('model_used', 'N/A')}",
                f"**Severity:** {vlm.get('rule_severity', 'N/A')}",
                f"\n**Visual assessment:**\n{vlm.get('vlm_blind', 'N/A')}",
                f"\n**Severity details:**\n{vlm.get('rule_severity_text', 'N/A')}",
                f"\n**Detection cross-check:**\n{vlm.get('yolo_check', 'N/A')}",
                f"\n**Suggested repair:**\n{vlm.get('recommended_action', 'N/A')}",
            ])

        detections_json = {
            "total_detections": det["total_cracks"],
            "average_confidence": det["average_confidence"],
            "class_counts": det["class_counts"],
            "detections": det["detections"],
        }

        gradcam_image = cam["overlay_rgb"] if cam else None

        return (
            image_rgb,
            det["annotated_rgb"],
            gradcam_image,
            "\n\n".join(summary),
            detections_json,
        )

    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(
            f"Inspection failed: {exc}"
        ) from exc


with gr.Blocks(title="Road Crack Detection AI") as demo:
    gr.Markdown(
        """
        # Road Crack Detection & Structural Assessment

        Upload a road image to detect longitudinal cracks, transverse
        cracks, alligator cracking, and potholes. Grad-CAM and the
        optional vision-language assessment can be enabled below.
        """
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Road Pavement Image",
                type="pil",
            )

            conf = gr.Slider(
                0.05, 1.0,
                value=config.DEFAULT_CONF,
                step=0.05,
                label="Confidence Threshold",
            )
            iou = gr.Slider(
                0.10, 1.0,
                value=config.DEFAULT_IOU,
                step=0.05,
                label="IoU Threshold",
            )

            enable_gradcam = gr.Checkbox(
                value=True,
                label="Enable Grad-CAM Explainability",
            )
            layer_id = gr.Dropdown(
                choices=[
                    (f"{key}: {value}", key)
                    for key, value in config.GRADCAM_LAYERS.items()
                ],
                value=config.DEFAULT_GRADCAM_LAYER,
                type="value",
                label="Grad-CAM Target Layer",
            )
            cam_alpha = gr.Slider(
                0.10, 0.90,
                value=config.DEFAULT_GRADCAM_ALPHA,
                step=0.05,
                label="Grad-CAM Overlay Opacity",
            )

            enable_vlm = gr.Checkbox(
                value=False,
                label="Enable Qwen2-VL Structural Assessment",
            )
            offline_mode = gr.Checkbox(
                value=True,
                label="Use Offline Rule-Based Assessment",
                info="Avoids heavy VLM generation when supported by the existing VLM module.",
            )

            run_button = gr.Button(
                "Run Road Inspection",
                variant="primary",
            )

        with gr.Column():
            original_output = gr.Image(label="Original Image")
            detection_output = gr.Image(label="Detection Results")
            cam_output = gr.Image(label="Grad-CAM Heatmap")

    summary_output = gr.Markdown(label="Inspection Summary")
    detections_output = gr.JSON(label="Detection Details")

    run_button.click(
        fn=inspect_road,
        inputs=[
            input_image,
            conf,
            iou,
            enable_gradcam,
            layer_id,
            cam_alpha,
            enable_vlm,
            offline_mode,
        ],
        outputs=[
            original_output,
            detection_output,
            cam_output,
            summary_output,
            detections_output,
        ],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
