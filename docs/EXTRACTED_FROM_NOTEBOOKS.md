# Extracted Specifications from Reference Notebooks

This document details the exact implementations, parameters, architectures, and algorithms extracted from the reference notebooks:
1. `yolo26s_strip (1).ipynb`: YOLO26s with custom Strip-Convolution blocks, training, validation, and metrics.
2. `yolo26s_strip_vlm_v2 (1).ipynb`: Checkpoint patching (`best_fixednames.pt`), Grad-CAM implementation (`YoloGradCAM`), and VLM pipeline with Qwen2-VL.

---

## 1. Model Loading & Architecture

### Architecture: YOLO26s + StripConvBlock Neck
The detector uses an Ultralytics YOLO26s base model with custom strip-convolution blocks injected into the neck at layers 16, 19, and 22:
- **Layer 16 (`P3`, fine)**: Stride 8, 128 channels, strip kernel $k = 11$.
- **Layer 19 (`P4`, medium)**: Stride 16, 256 channels, strip kernel $k = 11$.
- **Layer 22 (`P5`, coarse)**: Stride 32, 512 channels, strip kernel $k = 7$.

### Custom Modules
The checkpoint `best_fixednames.pt` contains pickled instances of custom PyTorch modules:
```python
class StripConvBlock(nn.Module):
    def __init__(self, c, k=11):
        super().__init__()
        p = k // 2
        self.dw_local = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.dw_h = nn.Conv2d(c, c, (1, k), padding=(0, p), groups=c, bias=False)
        self.dw_v = nn.Conv2d(c, c, (k, 1), padding=(p, 0), groups=c, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU()
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        nn.init.zeros_(self.pw.weight)

    def forward(self, x):
        y = self.dw_local(x) + self.dw_h(x) + self.dw_v(x)
        return x + self.pw(self.act(self.bn(y)))

class NeckStrip(nn.Module):
    def __init__(self, base, c, k):
        super().__init__()
        self.base = base
        self.strip = StripConvBlock(c, k)
        self.f, self.i, self.type, self.np = base.f, base.i, base.type, base.np

    def forward(self, x):
        return self.strip(self.base(x))
```
> **CRITICAL REQUIREMENT**: Because Python's `pickle` looks up custom classes in the module namespace during deserialization, `StripConvBlock` and `NeckStrip` must be defined and bound to `__main__` prior to invoking `YOLO(weights_path)`.

### Checkpoint & Class Names
- File: `best_fixednames.pt` (patched in Cell 2 of `yolo26s_strip_vlm_v2 (1).ipynb` to correct inverted class labels for classes 2 and 3).
- Classes mapping (`model.names`):
  - `0`: `'Longitudinal Crack'`
  - `1`: `'Transverse Crack'`
  - `2`: `'Alligator Crack'`
  - `3`: `'Pothole'`
- Input size: $640 \times 640$ pixels (`imgsz=640`).

---

## 2. Detection Inference & Pre/Post-Processing

### Preprocessing
- Letterboxing to $640 \times 640$ maintaining aspect ratio with constant border padding `(114, 114, 114)`.
- Normalization: Scale $[0, 255] \to [0.0, 1.0]$.
- Image color format: RGB for inference, BGR for OpenCV operations.

### Inference Parameters
- Confidence threshold default: `CONF = 0.25` (configurable in UI).
- IoU threshold default: `0.45` (standard Ultralytics NMS default).
- Image size: `640`.

### Spatial Categorization (`where` helper)
The notebook classifies detection spatial location:
```python
def where(box, W, H):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
    v = 'top' if cy < 1/3 else 'bottom' if cy > 2/3 else 'middle'
    h = 'left' if cx < 1/3 else 'right' if cx > 2/3 else 'center'
    return f'{v}-{h}'
```
Area percentage: `area_pct = 100 * (x2 - x1) * (y2 - y1) / (W * H)`.

---

## 3. Grad-CAM / HiResCAM Explainability (XAI)

### Implementation Mechanism
Extracted from `YoloGradCAM` in Cell 10 of `yolo26s_strip_vlm_v2 (1).ipynb`:
- **Architecture target layers**: Layer 16 (`P3 (fine)`, stride 8), Layer 19 (`P4`, stride 16), Layer 22 (`P5 (coarse)`, stride 32).
- **Hook setup**:
  - Forward hooks on `net.model[i]` store activations in `acts[i]`.
  - When `out.requires_grad`, a backward hook `lambda g, i=i: grads.__setitem__(i, g)` records output gradients.
- **Score Target Formulation**:
  - Differentiable score extraction (`_scores(out)`): searches the model outputs for end-to-end `[B, N, 6]` (column 4 = confidence score) or dense `[B, 4+nc, N]` (`max(0).values`).
  - Target calculation:
    `keep = scores >= conf`
    `target = scores[keep].sum() if keep.any() else scores.max()`
    `target.backward()`
- **HiResCAM Formula**:
  Element-wise product of gradient and activation retains precise spatial location of thin cracks:
  $$ \text{CAM} = \text{ReLU}\left( \sum_c g_{c,h,w} \cdot a_{c,h,w} \right) $$
- **Artifact suppression & Normalization**:
  1. Border artifact removal: Zeroes out boundary rows/columns based on stride and 16px border: $b = \lceil 16 / \text{stride} \rceil$.
  2. Hot-spot clipping: Clips values above the 99.5th percentile (`np.percentile(cam, 99.5)`).
  3. Resizing: Resized from feature map to letterbox $640 \times 640$, cropped to original aspect region `[top:top+nh, left:left+nw]`, then resized to original `(W, H)`.
  4. Normalization: $\text{cam} / (\max(\text{cam}) + 10^{-8})$.
- **Overlay Rendering**:
  - Colormap: `cv2.COLORMAP_JET`.
  - Opacity blending: `cv2.addWeighted(bgr, 1 - alpha, heat, alpha, 0)` with default $\alpha = 0.45$.

---

## 4. VLM (Qwen2-VL) Pipeline

### Model & Configuration
- Notebook model: `Qwen/Qwen2-VL-7B-Instruct` with 4-bit quantization via `BitsAndBytesConfig` on GPU.
- App default fallback: `Qwen/Qwen2-VL-2B-Instruct` (CPU-compatible and lightweight) with option to select 7B if GPU VRAM allows.
- Processor: `AutoProcessor.from_pretrained(VLM_ID, min_pixels=256*28*28, max_pixels=640*28*28)`.
- Generation settings: `do_sample=False`, `repetition_penalty=1.05`, `max_new_tokens=180`.

### Rule-Based Severity Logic
From Notebook Cell 11:
```python
def rule_severity(d, W, H):
    x1, y1, x2, y2 = d['box']
    extent = 100 * max((x2 - x1) / W, (y2 - y1) / H)
    area = d['area_pct']
    if d['class'] == 'Pothole':
        return 'high' if area >= 5 else 'medium' if area >= 1 else 'low'
    if d['class'] == 'Alligator Crack':
        return 'high' if area >= 10 else 'medium'
    return 'high' if extent >= 60 else 'medium' if extent >= 25 else 'low'
```

### Prompt Templates
- **Blind Prompt**:
  `Look at this road image. Which ONE best describes the main damage: longitudinal crack, transverse crack, pothole, alligator crack, or none? Answer with that label, then one short sentence describing what you see.`
- **Inspection & Repair Prompt**:
  Passed YOLO detections (class, confidence, position, area %) and rule-based severity.
  Structured output requested:
  1. Description of visible damage
  2. Severity assessment & detector cross-check
  3. Recommended action & urgency

---

## 5. Discrepancies & Resolutions (Notebook vs Master Prompt)

1. **VLM Model Size**:
   - Notebook: `Qwen/Qwen2-VL-7B-Instruct` loaded with 4-bit quantization (requires CUDA).
   - Prompt: Specifies `Qwen/Qwen2-VL-2B-Instruct` as fallback, capable of running on CPU.
   - *Resolution*: Default to `Qwen/Qwen2-VL-2B-Instruct` in `config.py` for universal CPU/GPU reliability, allowing 7B if CUDA is available and selected.
2. **VLM Output Format**:
   - Notebook: 3 plain text lines.
   - Prompt: JSON-formatted output with sections (Description, Severity, Recommended Action), with robust fallback.
   - *Resolution*: Instruct the VLM to produce structured JSON matching the three sections while maintaining the notebook's prompt grounding.
3. **Grad-CAM Multi-Layer Selection**:
   - Notebook saves 3 distinct heatmaps (P3 fine, P4, P5 coarse).
   - Prompt mentions Grad-CAM overlay.
   - *Resolution*: Compute P3 (fine) as the default primary crack heatmap (as P3 at stride 8 is optimal for thin cracks), while allowing user selection of P3, P4, or P5 in the UI.
