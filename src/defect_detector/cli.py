"""Command line tools for general object and defect detection."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="defectdet", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="Convert Labelme images to a tiled YOLO dataset")
    p.add_argument("source", type=Path, help="folder of images with sibling Labelme JSON files")
    p.add_argument("output", type=Path, help="new YOLO dataset folder")
    p.add_argument("--tile-size", type=int, default=1024)
    p.add_argument("--overlap", type=int, default=128)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-visible", type=float, default=0.5)
    p = commands.add_parser("train", help="Train a YOLO model from an Ultralytics YAML config")
    p.add_argument("config", type=Path)
    p.add_argument("--model")
    p.add_argument("--data")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch", type=int)
    p.add_argument("--device")
    p = commands.add_parser("predict", help="Run a trained YOLO model on images")
    p.add_argument("weights", type=Path)
    p.add_argument("source", type=Path)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--save-txt", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        from defect_detector.dataset import prepare

        counts = prepare(args.source, args.output, args.tile_size, args.overlap, args.val_fraction, args.seed, args.min_visible)
        print(f"wrote {counts['train']} train and {counts['val']} val tiles to {args.output}")
    elif args.command == "train":
        from defect_detector.yolo import train

        overrides = {key: getattr(args, key) for key in ("model", "data", "epochs", "batch", "device") if getattr(args, key) is not None}
        train(args.config, **overrides)
    else:
        from defect_detector.yolo import predict

        predict(args.weights, args.source, args.conf, args.iou, args.save_txt)


if __name__ == "__main__":
    main()
