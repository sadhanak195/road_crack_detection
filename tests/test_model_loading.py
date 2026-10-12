"""Smoke test for Phase 1: Verify model loads and prints class names."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from utils.logging_setup import get_logger

logger = get_logger("test_phase1")

# Define custom strip convolution blocks required for unpickling
import torch
import torch.nn as nn

class StripConvBlock(nn.Module):
    """Residual strip-convolution block for long, thin structures such as cracks."""
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
    """Wraps an existing neck block and applies the strip block to its output."""
    def __init__(self, base, c, k):
        super().__init__()
        self.base = base
        self.strip = StripConvBlock(c, k)
        self.f, self.i, self.type, self.np = base.f, base.i, base.type, base.np

    def forward(self, x):
        return self.strip(self.base(x))


def ensure_custom_classes_registered():
    """Ensure custom modules are present in __main__ so torch.load can find them."""
    import __main__
    __main__.StripConvBlock = StripConvBlock
    __main__.NeckStrip = NeckStrip


def test_model_load():
    ensure_custom_classes_registered()
    from ultralytics import YOLO

    weights_path = config.MODEL_PATH
    logger.info(f"Loading YOLO model from: {weights_path}")
    assert weights_path.exists(), f"Weights file not found at {weights_path}"

    model = YOLO(str(weights_path))
    names = model.names
    logger.info(f"Model successfully loaded! Classes: {names}")
    print(f"CHECKPOINT_PASSED: model.names = {names}")
    assert len(names) == 4
    assert names[0] == "Longitudinal Crack"
    assert names[1] == "Transverse Crack"
    assert names[2] == "Alligator Crack"
    assert names[3] == "Pothole"
    return names

if __name__ == "__main__":
    test_model_load()
