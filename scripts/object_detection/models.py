"""Object detection model selection."""

from scripts._runner import load_model

TASK = "detect"
DEFAULT_WEIGHTS = "yolov8n.pt"


def load(weights: str = DEFAULT_WEIGHTS):
    return load_model(weights, TASK)
