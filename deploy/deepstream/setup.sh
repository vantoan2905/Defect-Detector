#!/usr/bin/env bash
# Build the DeepStream-Yolo bbox parser and export the trained YOLO weights to ONNX.
# Run on the DeepStream machine (inside the DeepStream container or on Jetson):
#
#   CUDA_VER=12.6 deploy/deepstream/setup.sh runs/capacitor_detector/train/weights/best.pt
#   CUDA_VER=12.6 deploy/deepstream/setup.sh runs/defect_detector/train/weights/best.pt 1024 defect_detector
#
# CUDA_VER must match the CUDA version shipped with your DeepStream release.
set -euo pipefail

WEIGHTS=$(realpath "${1:?usage: setup.sh <best.pt> [imgsz] [model_name]}")
IMGSZ=${2:-1344}
MODEL_NAME=${3:-capacitor_detector}
[[ "$MODEL_NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "invalid model name: $MODEL_NAME" >&2; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd)
DS_YOLO="$HERE/DeepStream-Yolo"
MODELS="$HERE/models"
: "${CUDA_VER:?set CUDA_VER, e.g. CUDA_VER=12.6}"

if [ ! -d "$DS_YOLO" ]; then
    git clone https://github.com/marcoslucianops/DeepStream-Yolo.git "$DS_YOLO"
fi
make -C "$DS_YOLO/nvdsinfer_custom_impl_Yolo" CUDA_VER="$CUDA_VER"

# Export with the DeepStream-Yolo script: it adds the output layer NvDsInferParseYolo expects
# and writes labels.txt in the class order of the trained model.
mkdir -p "$MODELS"
WORK=$(mktemp -d)
cp "$WEIGHTS" "$WORK/$MODEL_NAME.pt"
(cd "$WORK" && python3 "$DS_YOLO/utils/export_yoloV8.py" -w "$MODEL_NAME.pt" -s "$IMGSZ" --dynamic --simplify)
mv "$(ls -t "$WORK"/*.onnx | head -1)" "$MODELS/$MODEL_NAME.onnx"
mv "$WORK/labels.txt" "$MODELS/${MODEL_NAME}_labels.txt"
if [ "$MODEL_NAME" = capacitor_detector ]; then
    cp "$MODELS/${MODEL_NAME}_labels.txt" "$MODELS/labels.txt"
fi
rm -rf "$WORK"

echo "Exported $MODELS/$MODEL_NAME.onnx ($(wc -l < "$MODELS/${MODEL_NAME}_labels.txt") classes)"
echo "The TensorRT engine is built on the first run of app.py."
