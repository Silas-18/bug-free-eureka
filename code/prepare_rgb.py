"""Prepare LLVIP visible images and YOLO labels with a 70/20/10 split."""

import json
import os
from pathlib import Path
import random
import shutil
import xml.etree.ElementTree as ET


def main():
    root = Path(__file__).resolve().parent
    raw = root / "LLVIP"
    output = root / "datasets" / "llvip_rgb_721"
    samples = {}
    for original_split in ("train", "test"):
        for image in sorted((raw / "visible" / original_split).glob("*.jpg")):
            if image.stem in samples:
                raise ValueError(f"Duplicate sample ID: {image.stem}")
            if not (raw / "infrared" / original_split / image.name).is_file():
                raise FileNotFoundError(f"No infrared partner for {image}")
            annotation = raw / "Annotations" / f"{image.stem}.xml"
            if not annotation.is_file():
                raise FileNotFoundError(annotation)
            samples[image.stem] = (image, annotation)

    ids = sorted(samples)
    if len(ids) < 10:
        raise ValueError(f"Expected LLVIP image pairs, found {len(ids)} under {raw}")
    random.Random(42).shuffle(ids)
    train_end = round(len(ids) * 0.7)
    val_end = round(len(ids) * 0.9)
    splits = {"train": ids[:train_end], "val": ids[train_end:val_end], "test": ids[val_end:]}
    invalid = []

    for split, sample_ids in splits.items():
        images = output / "images" / split
        labels = output / "labels" / split
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        paths = []
        for sample_id in sample_ids:
            source, annotation = samples[sample_id]
            tree = ET.parse(annotation).getroot()
            width = int(tree.findtext("size/width"))
            height = int(tree.findtext("size/height"))
            rows = []
            for obj in tree.findall("object"):
                if obj.findtext("name") != "person":
                    continue
                x1, y1, x2, y2 = (
                    float(obj.findtext(f"bndbox/{key}"))
                    for key in ("xmin", "ymin", "xmax", "ymax")
                )
                x1, x2 = max(0, x1), min(width, x2)
                y1, y2 = max(0, y1), min(height, y2)
                if x1 >= x2 or y1 >= y2:
                    invalid.append(sample_id)
                    continue
                values = (
                    (x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                    (x2 - x1) / width, (y2 - y1) / height,
                )
                rows.append("0 " + " ".join(f"{value:.8f}" for value in values))
            target = images / source.name
            if not target.exists():
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copy2(source, target)
            (labels / f"{sample_id}.txt").write_text("\n".join(rows), encoding="utf-8")
            paths.append(target.as_posix())
        (output / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
        if split in ("train", "val"):
            limit = 64 if split == "train" else 32
            (output / f"smoke_{split}.txt").write_text(
                "\n".join(paths[:limit]) + "\n", encoding="utf-8"
            )

    (output / "split_ids.json").write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    (output / "invalid_boxes.txt").write_text("\n".join(invalid), encoding="utf-8")
    for config, prefix in (("rgb", ""), ("rgb_smoke", "smoke_")):
        (output / f"{config}.yaml").write_text(
            f"path: {json.dumps(output.as_posix(), ensure_ascii=False)}\n"
            f"train: {prefix}train.txt\nval: {prefix}val.txt\n"
            "test: test.txt\nnames: {0: person}\n",
            encoding="utf-8",
        )
    print({split: len(sample_ids) for split, sample_ids in splits.items()})
    print(f"Skipped degenerate boxes: {len(invalid)}")
    print(output)


if __name__ == "__main__":
    main()
