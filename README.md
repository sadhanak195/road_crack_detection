
---
title: Road Crack Detection AI
emoji: 🛣️
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
---

# Road Crack Detection AI

Road pavement defect detection using a custom-trained Ultralytics model, Grad-CAM explainability, and optional Qwen2-VL structural assessment.

Upload a road image to detect longitudinal cracks, transverse cracks, alligator cracks, and potholes.


# Road Crack Detection System (YOLO26s + Grad-CAM XAI + Qwen2-VL)

An end-to-end, multi-modal road pavement crack detection and structural assessment system built with **Streamlit**, **Ultralytics YOLO26s (with custom Strip-Convolution blocks)**, **Grad-CAM (HiResCAM) explainability**, and **Qwen2-VL** multimodal vision-language intelligence.

---

## 🌟 Key Features

1. **Autonomous Pavement Defect Detection**:
   - Injected **StripConvBlock** and **NeckStrip** modules tailored for long, slender crack structures.
   - 4 trained defect classes:
     - `Longitudinal Crack`
     - `Transverse Crack`
     - `Alligator Crack`
     - `Pothole`
   - Real-time bounding box annotations with adaptive typography, class badges, and confidences.
2. **Grad-CAM Explainable AI (XAI)**:
   - High-resolution Class Activation Mapping (**HiResCAM**) ported directly from research benchmarks.
   - Multi-scale neck feature maps:
     - `Layer 16`: $P_3$ fine feature map (stride 8, optimized for fine crack structures)
     - `Layer 19`: $P_4$ medium feature map (stride 16)
     - `Layer 22`: $P_5$ coarse feature map (stride 32)
   - Real-time opacity blending slider ($\alpha \in [0.10, 0.90]$) and raw heatmap export.
3. **Vision-Language Model (VLM) Structural Assessment**:
   - Multi-modal pavement condition assessment using **Qwen2-VL**.
   - Integrates rule-based civil engineering pavement rating formulas based on crack extent and area ratio.
   - Structured three-point engineering output:
     1. **Visible Damage Description**: Pattern, morphology, and spatial distribution.
     2. **Structural Severity Rating**: `Low`, `Medium`, or `High` with engineering justification.
     3. **Recommended Repair Action**: Concrete maintenance procedures, material suggestions, and urgency.
4. **Flexible Image Input**:
   - File upload (`JPG`, `PNG`, `WEBP`, `BMP`) with automatic EXIF orientation correction.
   - Direct image URL download with Content-Type validation, timeout guards, and OpenGraph `og:image` fallback.
   - Built-in sample test images from benchmark evaluations.
5. **Audit-Ready Exports**:
   - Download annotated detection image (`.png`).
   - Download Grad-CAM activation overlay (`.png`).
   - Export structured audit report (`.json`) with timestamps, device information, class counts, and bounding box coordinates.

---

## 🏗️ Architecture & Project Structure

```
road-crack-detection/
├── app.py                     # Streamlit frontend & interactive dashboard
├── config.py                  # Central configuration, class mapping, thresholds, paths
├── requirements.txt           # Pinned project dependencies
├── README.md                  # System documentation & run guide
├── models/
│   └── best_fixednames.pt     # Trained YOLO26s + Strip-Conv weights (21.1 MB)
├── core/
│   ├── image_io.py            # Upload & URL fetching, validation, and conversion
│   ├── detector.py            # YOLO26s loader with custom modules & prediction
│   ├── gradcam.py             # HiResCAM gradient hooks & heatmap blending
│   ├── vlm.py                 # Qwen2-VL multimodal inference & rule-based engine
│   └── report.py              # Audit report generation & PNG/JSON serialization
├── utils/
│   ├── drawing.py             # Adaptive bounding box rendering & high-contrast labels
│   └── logging_setup.py       # Centralized application logging
├── sample_images/             # Benchmark test images (Potholes, Cracks)
├── docs/
│   └── EXTRACTED_FROM_NOTEBOOKS.md # Architectural extraction report
└── tests/
    ├── test_model_loading.py  # Phase 1: Model deserialization checkpoint
    ├── test_phase2.py         # Phase 2: Detection and annotation checkpoint
    ├── test_phase3.py         # Phase 3: Grad-CAM heatmap alignment checkpoint
    ├── test_phase4.py         # Phase 4: VLM parsing & severity checkpoint
    └── test_end_to_end.py     # End-to-end integration test suite
```

---

## ⚙️ Installation & Setup (Windows)

### Prerequisites
- Python 3.10 or 3.11 installed on Windows.
- Windows PowerShell or Command Prompt.

### Step 1: Clone or Navigate to the Project
```powershell
cd C:\Users\dell\Downloads\yolo26\road-crack-detection
```

### Step 2: Create and Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

> **Note on Model Download Size**:
> - Detector weights (`best_fixednames.pt`) are included locally (~21.1 MB).
> - On first use with neural VLM, `Qwen/Qwen2-VL-2B-Instruct` will download weights (~4.5 GB) from Hugging Face into `~/.cache/huggingface/hub/`.
> - On CPU-only environments, the application includes a built-in instant **Rule-Based Engineering Engine** toggle for zero-latency assessments without downloading the 4.5 GB model.

---

## 🚀 Running the Web Application

Launch the Streamlit web application:

```powershell
streamlit run app.py
```

The application will launch and open in your default web browser at:
```
http://localhost:8501
```

---

## 🧪 Running Smoke and Regression Tests

To run the complete automated test suite:

```powershell
# Run Phase 1 model loading test
python tests/test_model_loading.py

# Run Phase 2 detector and annotation test
python tests/test_phase2.py

# Run Phase 3 Grad-CAM XAI test
python tests/test_phase3.py

# Run Phase 4 VLM and rule engine test
python tests/test_phase4.py

# Run End-to-End integration suite
python tests/test_end_to_end.py
```

---

## 🛡️ Defect Classification & Severity Standards

| Class ID | Defect Name | Severity Criteria | Typical Engineering Action |
|---|---|---|---|
| 0 | Longitudinal Crack | Extent $\ge 60\%$: High<br>Extent $\ge 25\%$: Medium<br>Extent $< 25\%$: Low | Hot-pour crack sealant, rout and seal. |
| 1 | Transverse Crack | Extent $\ge 60\%$: High<br>Extent $\ge 25\%$: Medium<br>Extent $< 25\%$: Low | Flexible polymer sealant, prevent water penetration. |
| 2 | Alligator Crack | Area $\ge 10\%$: High<br>Area $< 10\%$: Medium | Full-depth asphalt patch, sub-base stabilization. |
| 3 | Pothole | Area $\ge 5\%$: High<br>Area $\ge 1\%$: Medium<br>Area $< 1\%$: Low | Cold/hot asphalt patch, localized mill-and-fill. |
