# MixedWM38 V2 classification trial

The archive at `/mnt/d/archive` contains 37,015 colored 52×52 wafer-map images and 38 image-level pattern labels. It has no object bounding boxes, so this trial trains a **38-class classifier**. Its checkpoint is not a drop-in replacement for the DeepStream RTSP **detector**.

## Dataset preparation

```sh
python -m scripts.image_classification.label /mnt/d/archive data/processed/wafer_cls_v2
```

The script creates symlinks to the colored images and a `split_manifest.csv`. It uses an 80/10/10 class-stratified split at the `map_sha256` group level, with seed 42. Counts: 29,615 train, 3,700 validation, 3,700 test. No map hash appears in more than one split. The source archive stays unchanged; `/mnt/d/archive` must remain mounted while training or evaluating. To train on raw categorical maps instead, use `--representation raw` and a different output directory.

## Trial training

The run used Python 3.11, Ultralytics 8.4.215, PyTorch 2.15.0+cu132 and an RTX 5060 Laptop GPU. Install Ultralytics in a Python 3.11 environment, then run:

```python
from ultralytics import YOLO

YOLO("yolov8n-cls.pt").train(
    data="data/processed/wafer_cls_v2",
    epochs=2,
    imgsz=64,
    batch=128,
    device=0,
    workers=4,
    scale=0.0,
    fliplr=0.0,
    flipud=0.0,
    erasing=0.0,
    auto_augment=None,
    seed=42,
    project="runs/wafer_cls_v2",
    name="smoke",
)
```

This run's best checkpoint is at `runs/classify/runs/wafer_cls_v2/smoke/weights/best.pt` (Ultralytics resolved the relative `project` path under `runs/classify/`). `results.csv` in the same run folder contains the per-epoch metrics.

## Results

| Split | Top-1 accuracy | Top-5 accuracy |
| --- | ---: | ---: |
| Validation, epoch 1 | 65.32% | 97.54% |
| Validation, epoch 2 | 84.73% | 99.84% |
| Independent test, best checkpoint | 84.76% | 99.89% |

The independent test evaluation used `YOLO(best.pt).val(data=dataset_dir, split="test", imgsz=64, batch=128, device=0)`. Its `metrics.json` is in `runs/wafer_cls_v2/test_eval/`. Copies of the [count matrix](assets/wafer-classification/confusion_matrix.png) and [normalized matrix](assets/wafer-classification/confusion_matrix_normalized.png) are tracked in Git and displayed in the main README. This is a short smoke run, not a final model selection or deployment benchmark.
