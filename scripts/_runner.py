"""Shared Ultralytics command line handling for the three vision tasks."""

from __future__ import annotations

import argparse
from pathlib import Path


def load_model(weights: str, task: str):
    from ultralytics import YOLO

    model = YOLO(weights)
    if model.task != task:
        raise ValueError(f"{weights} is a {model.task} model; expected {task}")
    return model


def train(task: str, default_weights: str, argv: list[str] | None = None, no_augment: bool = False) -> None:
    parser = argparse.ArgumentParser(description=f"Train a YOLO {task} model")
    parser.add_argument("data", type=Path, help="dataset.yaml for detect/segment, dataset directory for classify")
    parser.add_argument("--model", default=default_weights)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640 if task != "classify" else 224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", help="Ultralytics device, e.g. 0 or cpu; default is automatic")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--name", default="train")
    parser.add_argument("--seed", type=int, default=42)
    if no_augment:
        parser.add_argument("--no-augment", action="store_true", help="disable crop, flips and erasing for pattern maps")
    args = parser.parse_args(argv)
    _check_data(args.data, task, "train")
    model = load_model(args.model, task)
    options = {
        "data": str(args.data.resolve()), "epochs": args.epochs, "imgsz": args.imgsz,
        "batch": args.batch, "device": args.device, "workers": args.workers,
        "name": args.name, "seed": args.seed,
    }
    if args.project is not None:
        options["project"] = str(args.project.resolve())
    if args.device is None:
        options.pop("device")
    if no_augment and args.no_augment:
        options.update(scale=0.0, fliplr=0.0, flipud=0.0, erasing=0.0, auto_augment=None)
    model.train(**options)


def evaluate(task: str, split: str, argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=f"Evaluate a YOLO {task} model on {split}")
    parser.add_argument("weights", type=Path)
    parser.add_argument("data", type=Path, help="dataset.yaml for detect/segment, dataset directory for classify")
    parser.add_argument("--imgsz", type=int, default=640 if task != "classify" else 224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", help="Ultralytics device, e.g. 0 or cpu; default is automatic")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--name", default=split)
    args = parser.parse_args(argv)
    if not args.weights.is_file():
        parser.error(f"weights not found: {args.weights}")
    try:
        _check_data(args.data, task, split)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    model = load_model(str(args.weights), task)
    options = {
        "data": str(args.data.resolve()), "split": split, "imgsz": args.imgsz,
        "batch": args.batch, "device": args.device, "workers": args.workers,
        "name": args.name,
    }
    if args.project is not None:
        options["project"] = str(args.project.resolve())
    if args.device is None:
        options.pop("device")
    metrics = model.val(**options)
    if task == "classify":
        print(f"{split}: top1={metrics.top1:.4f} top5={metrics.top5:.4f}")
    elif task == "segment":
        print(f"{split}: box_mAP50={metrics.box.map50:.4f} mask_mAP50={metrics.seg.map50:.4f}")
    else:
        print(f"{split}: box_mAP50={metrics.box.map50:.4f}")


def _check_data(data: Path, task: str, split: str) -> None:
    if task == "classify":
        if not (data / split).is_dir():
            raise FileNotFoundError(f"classification split missing: {data / split}")
        return
    if not data.is_file():
        raise FileNotFoundError(f"dataset YAML missing: {data}")
    import yaml

    config = yaml.safe_load(data.read_text())
    if not isinstance(config, dict) or not config.get(split):
        raise ValueError(f"{data} has no {split} split")
