# Defect Detector

Prepare annotated images for object or defect detection, train a YOLO model, and run predictions. The general workflow accepts any product and any set of defect labels. A capacitor tray pipeline is retained as a worked example.

## Quick start

Requires Python 3.10 or newer.

```sh
git clone https://github.com/vantoan2905/Defect-Detector.git
cd Defect-Detector
python -m venv .venv
source .venv/bin/activate
pip install -e ".[yolo,dev]"
defectdet --help
```

For the Faster R-CNN example, install `.[frcnn]`. `pip install -r requirements.txt` installs all extras.

## General workflow

Put images (`.bmp`, `.png`, `.jpg`, `.jpeg`, `.tif` or `.tiff`) under one source folder. Each image needs a sibling Labelme JSON file with the same stem. Draw rectangles or polygons and give each object its intended class label, such as `scratch`, `dent` or `missing_part`. Shapes become YOLO bounding boxes. Nested folders are supported.

```text
data/raw/my_product/
├── image_001.png
├── image_001.json
├── image_002.png
└── image_002.json
```

Prepare overlapping tiles and a YOLO train/validation dataset:

```sh
defectdet prepare data/raw/my_product data/processed/defects \
  --tile-size 1024 --overlap 128 --val-fraction 0.2
```

The command discovers class names from the Labelme files and writes `images/{train,val}`, `labels/{train,val}` and `dataset.yaml`. It splits by original image, so tiles from one image stay in the same split. Boxes crossing a tile edge are clipped; `--min-visible` (default `0.5`) controls the minimum retained fraction. Empty tiles are kept as negative examples. Use a fresh output directory for each dataset build.

Edit [configs/defect_detector.yaml](configs/defect_detector.yaml) to select the model, dataset, image size and training settings. Then run:

```sh
defectdet train configs/defect_detector.yaml --device cpu
defectdet predict runs/defect_detector/train/weights/best.pt data/raw/my_product --conf 0.25
```

`train` accepts `--model`, `--data`, `--epochs`, `--batch` and `--device` overrides. The example config points to `data/processed/defects/dataset.yaml`. Dataset paths in the generated YAML are absolute; regenerate the dataset after moving it.

## Capacitor tray example

The original capacitor pipeline includes tray region detection, regular cell grids, defect spreadsheet merging, specialized tiling, Faster R-CNN and DeepStream deployment. It remains available through `capdet` and is documented in [docs/capacitor-example.md](docs/capacitor-example.md). Its sample configs are [configs/capacitor_detector.yaml](configs/capacitor_detector.yaml) and [configs/region_detector.yaml](configs/region_detector.yaml). The old `capacitor_detection` Python package and `capdet` command remain for existing scripts.

For a live RTSP camera on an NVIDIA GPU, see the [DeepStream RTSP flow](deploy/deepstream/README.md#live-rtsp-camera-general-defect-detector).

## Project structure

```text
configs/                 General and example YOLO training configs
data/                    Local datasets (ignored by Git; see data/README.md)
src/defect_detector/     General Labelme-to-YOLO preparation, training and CLI
src/capacitor_detection/ Capacitor example and compatibility wrapper
deploy/deepstream/       Capacitor example deployment
tests/                   Dataset and legacy pipeline checks
```

## Development

```sh
pytest
ruff check src tests
```
