"""Convert Labelme annotated images into a tiled YOLO detection dataset."""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import yaml

IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size <= 0 or not 0 <= overlap < tile_size:
        raise ValueError("tile size must be positive and overlap must be smaller than tile size")
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, tile_size - overlap))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def _annotations(image_path: Path) -> list[tuple[str, tuple[float, float, float, float]]]:
    json_path = image_path.with_suffix(".json")
    if not json_path.is_file():
        raise FileNotFoundError(f"missing Labelme annotation: {json_path}")
    document = json.loads(json_path.read_text())
    objects = []
    for shape in document.get("shapes", []):
        label = shape.get("label")
        points = shape.get("points", [])
        if not label or len(points) < 2:
            raise ValueError(f"invalid label or points in {json_path}")
        xs, ys = zip(*points)
        box = (min(xs), min(ys), max(xs), max(ys))
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"zero-area shape in {json_path}")
        objects.append((label, box))
    return objects


def _yolo_lines(objects, names: dict[str, int], x0: int, y0: int, width: int, height: int, min_visible: float):
    lines = []
    for label, (left, top, right, bottom) in objects:
        clipped = (max(left, x0), max(top, y0), min(right, x0 + width), min(bottom, y0 + height))
        visible_w, visible_h = clipped[2] - clipped[0], clipped[3] - clipped[1]
        if visible_w <= 0 or visible_h <= 0:
            continue
        if visible_w * visible_h / ((right - left) * (bottom - top)) < min_visible:
            continue
        cx = ((clipped[0] + clipped[2]) / 2 - x0) / width
        cy = ((clipped[1] + clipped[3]) / 2 - y0) / height
        lines.append(f"{names[label]} {cx:.6f} {cy:.6f} {visible_w / width:.6f} {visible_h / height:.6f}")
    return lines


def prepare(
    source: Path,
    output: Path,
    tile_size: int = 1024,
    overlap: int = 128,
    val_fraction: float = 0.2,
    seed: int = 42,
    min_visible: float = 0.5,
) -> dict[str, int]:
    """Write YOLO train/val images and labels, splitting by source image to avoid leakage."""
    source, output = Path(source), Path(output)
    if not source.is_dir():
        raise NotADirectoryError(source)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError("output directory must be outside the source folder")
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    if not 0 <= min_visible <= 1:
        raise ValueError("min_visible must be between 0 and 1")
    _starts(1, tile_size, overlap)
    images = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if len(images) < 2:
        raise ValueError("at least two annotated images are needed for a train/val split")
    records = [(image, _annotations(image)) for image in images]
    labels = sorted({label for _, objects in records for label, _ in objects})
    if not labels:
        raise ValueError("no labelled objects found")
    names = {label: i for i, label in enumerate(labels)}
    shuffled = images.copy()
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, min(len(images) - 1, round(len(images) * val_fraction)))
    validation = set(shuffled[:val_count])
    counts = {"train": 0, "val": 0}
    seen = set()
    for image_path, objects in records:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"cannot read image: {image_path}")
        height, width = image.shape[:2]
        split = "val" if image_path in validation else "train"
        stem = "__".join(image_path.relative_to(source).with_suffix("").parts)
        for y0 in _starts(height, tile_size, overlap):
            for x0 in _starts(width, tile_size, overlap):
                name = f"{stem}__x{x0}_y{y0}"
                if name in seen:
                    raise ValueError(f"duplicate output name: {name}; rename the source images")
                seen.add(name)
                tile = image[y0 : y0 + tile_size, x0 : x0 + tile_size]
                tile_h, tile_w = tile.shape[:2]
                image_out = output / "images" / split / f"{name}.png"
                label_out = output / "labels" / split / f"{name}.txt"
                image_out.parent.mkdir(parents=True, exist_ok=True)
                label_out.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(image_out), tile):
                    raise OSError(f"cannot write image: {image_out}")
                lines = _yolo_lines(objects, names, x0, y0, tile_w, tile_h, min_visible)
                label_out.write_text("\n".join(lines) + ("\n" if lines else ""))
                counts[split] += 1
    dataset = {"path": str(output.resolve()), "train": "images/train", "val": "images/val", "names": labels}
    (output / "dataset.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False, allow_unicode=True))
    return counts
