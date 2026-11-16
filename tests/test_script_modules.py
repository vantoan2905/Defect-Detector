import csv
import json

import cv2
import numpy as np
import pytest
import yaml
from scripts._runner import _check_data
from scripts.image_classification.label import prepare as prepare_classification
from scripts.segmentation.label import _segment_lines
from scripts.segmentation.label import prepare as prepare_segmentation

from defect_detector.dataset import prepare as prepare_detection


def _labelme_image(folder, name, shape):
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / f"{name}.png"), np.zeros((64, 64, 3), np.uint8))
    (folder / f"{name}.json").write_text(json.dumps({"shapes": [shape]}))


def test_detection_labels_have_independent_test_split(tmp_path):
    source = tmp_path / "source"
    for index in range(3):
        _labelme_image(source, str(index), {"label": "scratch", "points": [[4, 4], [20, 20]]})
    output = tmp_path / "detect"
    assert prepare_detection(source, output, tile_size=64, overlap=0, val_fraction=0.3, test_fraction=0.3) == {
        "train": 1, "val": 1, "test": 1,
    }
    assert yaml.safe_load((output / "dataset.yaml").read_text())["test"] == "images/test"
    _check_data(output / "dataset.yaml", "detect", "test")


def test_segmentation_labels_contain_normalized_polygon(tmp_path):
    source = tmp_path / "source"
    for index in range(3):
        _labelme_image(source, str(index), {
            "label": "chip", "shape_type": "polygon", "points": [[10, 10], [40, 10], [40, 40], [10, 40]],
        })
    output = tmp_path / "segment"
    assert prepare_segmentation(source, output, tile_size=64, overlap=0, val_fraction=0.3, test_fraction=0.3) == {
        "train": 1, "val": 1, "test": 1,
    }
    lines = [p.read_text().strip() for p in (output / "labels").rglob("*.txt")]
    assert len(lines) == 3
    for line in lines:
        values = line.split()
        assert values[0] == "0"
        assert len(values) >= 7 and (len(values) - 1) % 2 == 0
        assert all(0 <= float(value) <= 1 for value in values[1:])
    _check_data(output / "dataset.yaml", "segment", "test")


def test_segmentation_clips_polygon_at_tile_edge():
    polygon = np.asarray([[40, 10], [80, 10], [80, 30], [40, 30]], dtype=np.float32)
    left = _segment_lines([("chip", polygon)], {"chip": 0}, 0, 0, 64, 64, 0.5)
    right = _segment_lines([("chip", polygon)], {"chip": 0}, 64, 0, 64, 64, 0.5)
    assert len(left) == 1
    assert right == []
    assert all(0 <= float(value) <= 1 for value in left[0].split()[1:])


def test_classification_label_split_keeps_duplicate_hash_together(tmp_path):
    archive = tmp_path / "archive"
    images = archive / "colored" / "colored"
    images.mkdir(parents=True)
    (archive / "dataset_info.json").write_text(json.dumps({"classes": {"class_names": {"C1": "Normal"}}}))
    with (archive / "metadata.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_id", "map_sha256", "colored_image_filename"])
        writer.writeheader()
        for index in range(10):
            for copy in range(2 if index == 0 else 1):
                name = f"{index}_{copy}.png"
                (images / name).write_bytes(b"image")
                writer.writerow({"class_id": "C1", "map_sha256": f"hash-{index}", "colored_image_filename": name})
    output = tmp_path / "classify"
    assert sum(prepare_classification(archive, output).values()) == 11
    with (output / "split_manifest.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    duplicates = [row["split"] for row in rows if row["map_sha256"] == "hash-0"]
    assert len(set(duplicates)) == 1
    assert sorted({row["split"] for row in rows}) == ["test", "train", "val"]
    _check_data(output, "classify", "test")


def test_test_command_requires_test_split(tmp_path):
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text("train: images/train\nval: images/val\n")
    with pytest.raises(ValueError, match="no test split"):
        _check_data(dataset, "detect", "test")
