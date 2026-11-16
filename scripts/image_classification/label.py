"""Split MixedWM38 V2 wafer maps for image classification without hash leakage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def prepare(source: Path, output: Path, representation: str = "colored", seed: int = 42) -> dict[str, int]:
    source, output = Path(source).resolve(), Path(output).resolve()
    if representation not in {"colored", "raw"}:
        raise ValueError("representation must be colored or raw")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output must be empty: {output}")
    info = json.loads((source / "dataset_info.json").read_text())
    class_names = info["classes"]["class_names"]
    image_dir = source / representation / representation
    if not image_dir.is_dir():
        raise NotADirectoryError(image_dir)
    groups = defaultdict(lambda: defaultdict(list))
    with (source / "metadata.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            groups[row["class_id"]][row["map_sha256"]].append(row)
    if set(groups) != set(class_names):
        raise ValueError("metadata classes do not match dataset_info.json")
    hash_classes = defaultdict(set)
    for class_id, hashes in groups.items():
        for map_hash in hashes:
            hash_classes[map_hash].add(class_id)
    if any(len(classes) > 1 for classes in hash_classes.values()):
        raise ValueError("the same map hash appears under different classes")
    assignments = []
    for class_id in sorted(groups, key=lambda value: int(value[1:])):
        hashes = sorted(groups[class_id], key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())
        n = len(hashes)
        n_test = max(1, round(n * 0.1))
        n_val = max(1, round(n * 0.1))
        for index, map_hash in enumerate(hashes):
            split = "test" if index < n_test else "val" if index < n_test + n_val else "train"
            for row in groups[class_id][map_hash]:
                image = image_dir / row[f"{representation}_image_filename"]
                assignments.append((split, class_id, map_hash, image))
    output.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with (output / "split_manifest.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "class_id", "map_sha256", "image"])
        for split, class_id, map_hash, image in assignments:
            target_dir = output / split / class_id
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / image.name).symlink_to(image)
            writer.writerow([split, class_id, map_hash, str(image)])
            counts[split] += 1
    (output / "class_names.json").write_text(json.dumps(class_names, indent=2))
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="MixedWM38 V2 archive folder")
    parser.add_argument("output", type=Path, help="new train/val/test directory")
    parser.add_argument("--representation", choices=("colored", "raw"), default="colored")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(prepare(args.source, args.output, args.representation, args.seed))


if __name__ == "__main__":
    main()
