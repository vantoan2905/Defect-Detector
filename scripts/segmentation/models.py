"""Instance segmentation model selection."""

from scripts._runner import load_model

TASK = "segment"
DEFAULT_WEIGHTS = "yolov8n-seg.pt"


def load(weights: str = DEFAULT_WEIGHTS):
    return load_model(weights, TASK)
