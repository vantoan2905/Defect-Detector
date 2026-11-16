# Vision task scripts

Run these modules from the repository root with `python -m scripts.<task>.<action>`. Install the project with `pip install -e ".[yolo,dev]"` first. The model commands import Ultralytics only when run.

| Task directory | `models.py` | `label.py` input | Training data |
| --- | --- | --- | --- |
| `object_detection/` | YOLO detection model (`yolov8n.pt`) | Images with sibling Labelme JSON rectangles/polygons | YOLO `dataset.yaml` |
| `segmentation/` | YOLO instance segmentation model (`yolov8n-seg.pt`) | Images with sibling Labelme JSON polygons/rectangles | YOLO `dataset.yaml` |
| `image_classification/` | YOLO classification model (`yolov8n-cls.pt`) | MixedWM38 V2 `metadata.csv` and images | `train/`, `val/`, `test/` class folders |

Each task has `label.py`, `train.py`, `val.py` and `test.py`. `val.py` uses the validation split; `test.py` requires a held-out test split. The model loader rejects weights from a different task. Use `--help` on any module for its options.

## Object detection

```sh
python -m scripts.object_detection.label data/raw/my_product data/processed/detect
python -m scripts.object_detection.train data/processed/detect/dataset.yaml --epochs 50 --device 0
python -m scripts.object_detection.val path/to/best.pt data/processed/detect/dataset.yaml --device 0
python -m scripts.object_detection.test path/to/best.pt data/processed/detect/dataset.yaml --device 0
```

## Instance segmentation

```sh
python -m scripts.segmentation.label data/raw/segmented_product data/processed/segment
python -m scripts.segmentation.train data/processed/segment/dataset.yaml --epochs 50 --device 0
python -m scripts.segmentation.val path/to/best.pt data/processed/segment/dataset.yaml --device 0
python -m scripts.segmentation.test path/to/best.pt data/processed/segment/dataset.yaml --device 0
```

`label.py` rasterizes each Labelme polygon in a tile, clips it at the tile edge and writes normalized polygon coordinates. Tiny disconnected fragments that cannot form a polygon are discarded. The `--min-visible` fraction is measured against the source polygon area. This is **instance** segmentation; it expects one polygon per object in Labelme.

## Image classification: MixedWM38 V2

```sh
python -m scripts.image_classification.label /mnt/d/archive data/processed/wafer_cls_v2
python -m scripts.image_classification.train data/processed/wafer_cls_v2 \
  --model yolov8n-cls.pt --epochs 2 --imgsz 64 --batch 128 --device 0 --no-augment
python -m scripts.image_classification.val path/to/best.pt data/processed/wafer_cls_v2 --imgsz 64 --device 0
python -m scripts.image_classification.test path/to/best.pt data/processed/wafer_cls_v2 --imgsz 64 --device 0
```

The wafer label script creates symlinks, a class-name mapping and a split manifest. It groups identical `map_sha256` values before assigning train/val/test, so the source archive must remain mounted. `scripts/prepare_wafer_classification.py` remains as a compatibility entry point.

All three label commands require an empty output directory. Detection and segmentation split by source image before tiling; classification splits by map hash. Saved model outputs go under Ultralytics' `runs/` tree unless `--project` is given.
