"""Convert Labelme polygons into tiled YOLO instance segmentation labels."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import yaml

from defect_detector.dataset import IMAGE_SUFFIXES, _starts


def _shapes(image_path: Path) -> list[tuple[str, np.ndarray]]:
    annotation = image_path.with_suffix(".json")
    if not annotation.is_file():
        raise FileNotFoundError(f"missing Labelme annotation: {annotation}")
    shapes = []
    for shape in json.loads(annotation.read_text()).get("shapes", []):
        label = shape.get("label")
        kind = shape.get("shape_type", "polygon")
        points = np.asarray(shape.get("points", []), dtype=np.float32)
        if not label or points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"invalid shape in {annotation}")
        if kind == "rectangle" and len(points) == 2:
            (x1, y1), (x2, y2) = points
            points = np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        elif kind != "polygon" or len(points) < 3:
            raise ValueError(f"only polygons and rectangles are supported: {annotation}")
        if cv2.contourArea(points) <= 0:
            raise ValueError(f"zero-area polygon in {annotation}")
        shapes.append((label, points))
    return shapes


def _segment_lines(shapes, classes: dict[str, int], x0: int, y0: int, width: int, height: int,
                   min_visible: float) -> list[str]:
    lines = []
    offset = np.asarray([x0, y0], dtype=np.float32)
    for label, points in shapes:
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.rint(points - offset).astype(np.int32)], 1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        visible_area = sum(cv2.contourArea(contour) for contour in contours)
        if visible_area <= 0 or visible_area / cv2.contourArea(points) < min_visible:
            continue
        for contour in contours:
            xy = contour.reshape(-1, 2)
            if len(xy) < 3 or cv2.contourArea(contour) <= 0:
                continue
            normalized = xy.astype(float) / np.asarray([width, height])
            coords = " ".join(f"{value:.6f}" for value in normalized.flatten())
            lines.append(f"{classes[label]} {coords}")
    return lines


def prepare(source: Path, output: Path, tile_size: int = 1024, overlap: int = 128,
            val_fraction: float = 0.2, test_fraction: float = 0.1,
            seed: int = 42, min_visible: float = 0.5) -> dict[str, int]:
    source, output = Path(source), Path(output)
    if not source.is_dir():
        raise NotADirectoryError(source)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError("output must be outside source")
    if not 0 < val_fraction < 1 or not 0 <= test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("invalid validation/test fractions")
    if not 0 <= min_visible <= 1:
        raise ValueError("min_visible must be between 0 and 1")
    _starts(1, tile_size, overlap)
    images = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if len(images) < 2:
        raise ValueError("at least two annotated images are needed")
    records = [(image, _shapes(image)) for image in images]
    names = sorted({label for _, shapes in records for label, _ in shapes})
    if not names:
        raise ValueError("no labelled objects found")
    classes = {name: index for index, name in enumerate(names)}
    shuffled = images.copy()
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, round(len(images) * val_fraction))
    test_count = max(1, round(len(images) * test_fraction)) if test_fraction else 0
    if val_count + test_count >= len(images):
        raise ValueError("split fractions leave no training images")
    validation = set(shuffled[:val_count])
    testing = set(shuffled[val_count : val_count + test_count])
    counts = {"train": 0, "val": 0}
    if test_count:
        counts["test"] = 0
    seen = set()
    for image_path, shapes in records:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"cannot read image: {image_path}")
        height, width = image.shape[:2]
        split = "val" if image_path in validation else "test" if image_path in testing else "train"
        stem = "__".join(image_path.relative_to(source).with_suffix("").parts)
        for y0 in _starts(height, tile_size, overlap):
            for x0 in _starts(width, tile_size, overlap):
                name = f"{stem}__x{x0}_y{y0}"
                if name in seen:
                    raise ValueError(f"duplicate output name: {name}")
                seen.add(name)
                tile = image[y0 : y0 + tile_size, x0 : x0 + tile_size]
                tile_h, tile_w = tile.shape[:2]
                image_out = output / "images" / split / f"{name}.png"
                label_out = output / "labels" / split / f"{name}.txt"
                image_out.parent.mkdir(parents=True, exist_ok=True)
                label_out.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(image_out), tile):
                    raise OSError(f"cannot write image: {image_out}")
                lines = _segment_lines(shapes, classes, x0, y0, tile_w, tile_h, min_visible)
                label_out.write_text("\n".join(lines) + ("\n" if lines else ""))
                counts[split] += 1
    dataset = {"path": str(output.resolve()), "train": "images/train", "val": "images/val", "names": names}
    if test_count:
        dataset["test"] = "images/test"
    (output / "dataset.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False, allow_unicode=True))
    return counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="images with sibling Labelme polygons")
    parser.add_argument("output", type=Path)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-visible", type=float, default=0.5)
    args = parser.parse_args(argv)
    print(prepare(args.source, args.output, args.tile_size, args.overlap, args.val_fraction,
                  args.test_fraction, args.seed, args.min_visible))


if __name__ == "__main__":
    main()
