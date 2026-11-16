"""Convert Labelme rectangles or polygons into tiled YOLO detection labels."""

from __future__ import annotations

import argparse
from pathlib import Path

from defect_detector.dataset import prepare


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="images with sibling Labelme JSON files")
    parser.add_argument("output", type=Path)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-visible", type=float, default=0.5)
    args = parser.parse_args(argv)
    counts = prepare(args.source, args.output, args.tile_size, args.overlap, args.val_fraction,
                     args.seed, args.min_visible, args.test_fraction)
    print(counts)


if __name__ == "__main__":
    main()
