import json

import cv2
import numpy as np
import pytest
import yaml

from defect_detector.dataset import _starts, prepare


def _annotated_image(folder, stem, label, x0):
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / f"{stem}.png"), np.zeros((80, 120, 3), np.uint8))
    shapes = [{"label": label, "shape_type": "rectangle", "points": [[x0, 10], [x0 + 20, 30]]}]
    (folder / f"{stem}.json").write_text(json.dumps({"shapes": shapes}))


def test_prepare_splits_by_image_and_preserves_classes(tmp_path):
    src, out = tmp_path / "raw", tmp_path / "processed"
    _annotated_image(src / "batch_a", "first", "scratch", 35)
    _annotated_image(src / "batch_b", "second", "dent", 10)
    counts = prepare(src, out, tile_size=60, overlap=10, val_fraction=0.5)
    assert counts == {"train": 6, "val": 6}
    config = yaml.safe_load((out / "dataset.yaml").read_text())
    assert config["names"] == ["dent", "scratch"]
    assert config["path"] == str(out.resolve())
    assert len(list((out / "images" / "train").glob("*.png"))) == 6
    assert len(list((out / "images" / "val").glob("*.png"))) == 6
    assert all(len(list((out / "images" / split).glob("batch_a*"))) in (0, 6) for split in ("train", "val"))
    labels = [p.read_text().strip() for p in (out / "labels").rglob("*.txt")]
    assert any(line.startswith("0 ") for line in labels)
    assert any(line.startswith("1 ") for line in labels)
    assert all(0 <= float(value) <= 1 for line in labels if line for value in line.split()[1:])


def test_prepare_rejects_missing_annotation_and_existing_output(tmp_path):
    src = tmp_path / "raw"
    _annotated_image(src, "first", "scratch", 5)
    cv2.imwrite(str(src / "second.png"), np.zeros((20, 20, 3), np.uint8))
    with pytest.raises(FileNotFoundError):
        prepare(src, tmp_path / "processed")
    (tmp_path / "processed").mkdir()
    (tmp_path / "processed" / "old.txt").write_text("old")
    with pytest.raises(FileExistsError):
        prepare(src, tmp_path / "processed")


def test_tile_starts_cover_right_edge():
    assert _starts(120, 60, 10) == [0, 50, 60]
    assert _starts(40, 60, 10) == [0]
