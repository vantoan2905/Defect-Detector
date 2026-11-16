"""Image classification model selection."""

from scripts._runner import load_model

TASK = "classify"
DEFAULT_WEIGHTS = "yolov8n-cls.pt"


def load(weights: str = DEFAULT_WEIGHTS):
    return load_model(weights, TASK)
