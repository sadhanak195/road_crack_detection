"""Configuration for Road Crack Detection System."""

from pathlib import Path
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
SAMPLE_DIR = BASE_DIR / "sample_images"

# Model weights path
DEFAULT_WEIGHTS_PATH = MODELS_DIR / "best_fixednames.pt"
ORIGINAL_WEIGHTS_PATH = BASE_DIR.parent / "best_fixednames.pt"

# Active weights path (fall back to original if local not present)
MODEL_PATH = DEFAULT_WEIGHTS_PATH if DEFAULT_WEIGHTS_PATH.exists() else ORIGINAL_WEIGHTS_PATH

# Compute Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# YOLO Model Defaults
IMGSZ = 640
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45

# Class names mapping (verified from notebook)
CLASS_NAMES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Pothole",
}

# Distinct UI BGR colors for annotations:
# Longitudinal Crack: Cyan/Sky Blue
# Transverse Crack: Orange
# Alligator Crack: Magenta/Purple
# Pothole: Crimson/Red
CLASS_COLORS = {
    0: (230, 216, 0),    # Cyan/Sky Blue (BGR)
    1: (0, 165, 255),    # Orange (BGR)
    2: (255, 0, 255),    # Magenta (BGR)
    3: (34, 34, 220),    # Crimson Red (BGR)
}

# Grad-CAM Settings
GRADCAM_LAYERS = {
    16: "P3 (fine - cracks)",
    19: "P4 (medium)",
    22: "P5 (coarse)",
}
GRADCAM_STRIDES = {16: 8, 19: 16, 22: 32}
DEFAULT_GRADCAM_LAYER = 16
DEFAULT_GRADCAM_ALPHA = 0.45

# VLM Settings
DEFAULT_VLM_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
VLM_7B_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
VLM_MIN_PIXELS = 256 * 28 * 28
VLM_MAX_PIXELS = 640 * 28 * 28
VLM_MAX_NEW_TOKENS = 220

# Severity order
SEV_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}
