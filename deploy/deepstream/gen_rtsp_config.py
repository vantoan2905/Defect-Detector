"""Generate a DeepStream nvinfer config for a full-frame YOLO RTSP pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path


def generate(model_name: str, root: Path) -> Path:
    if not model_name or not all(c.isalnum() or c in "_-" for c in model_name):
        raise ValueError("model name must contain only letters, digits, underscores or hyphens")
    root = Path(root).resolve()
    models = root / "models"
    onnx = models / f"{model_name}.onnx"
    labels = models / f"{model_name}_labels.txt"
    library = root / "DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so"
    for path in (onnx, labels, library):
        if not path.is_file():
            raise FileNotFoundError(path)
    classes = [line.strip() for line in labels.read_text().splitlines() if line.strip()]
    if not classes:
        raise ValueError(f"no class labels in {labels}")
    engine = models / f"{model_name}.onnx_b1_gpu0_fp16.engine"
    config = models / f"{model_name}_rtsp_infer.txt"
    config.write_text(
        "[property]\n"
        "gpu-id=0\n"
        "gie-unique-id=1\n"
        "process-mode=1\n"
        "network-type=0\n"
        "batch-size=1\n"
        "net-scale-factor=0.00392156862745098\n"
        "model-color-format=0\n"
        "maintain-aspect-ratio=1\n"
        "symmetric-padding=1\n"
        f"onnx-file={onnx}\n"
        f"model-engine-file={engine}\n"
        "network-mode=2\n"
        f"labelfile-path={labels}\n"
        f"num-detected-classes={len(classes)}\n"
        "interval=0\n"
        "cluster-mode=2\n"
        "parse-bbox-func-name=NvDsInferParseYolo\n"
        f"custom-lib-path={library}\n"
        "engine-create-func-name=NvDsInferYoloCudaEngineGet\n"
        "\n[class-attrs-all]\n"
        "pre-cluster-threshold=0.25\n"
        "nms-iou-threshold=0.45\n"
        "topk=300\n"
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="defect_detector")
    args = parser.parse_args()
    print(generate(args.model_name, Path(__file__).parent))


if __name__ == "__main__":
    main()
