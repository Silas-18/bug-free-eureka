"""Build the IR-only YOLO dataset using the exact RGB sample lists."""

import json
import os
from pathlib import Path
import shutil


def main():
    root = Path(__file__).resolve().parent
    rgb = root / "datasets" / "llvip_rgb_721"
    output = root / "datasets" / "llvip_ir_721"
    infrared = root / "LLVIP" / "infrared"
    sources = {}
    for original_split in ("train", "test"):
        for image in (infrared / original_split).glob("*.jpg"):
            if image.stem in sources:
                raise ValueError(f"Duplicate IR sample ID: {image.stem}")
            sources[image.stem] = image

    lists = {}
    for name in ("train", "val", "test", "smoke_train", "smoke_val"):
        split = name.removeprefix("smoke_")
        entries = []
        for line in (rgb / f"{name}.txt").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sample_id = Path(line.strip()).stem
            source = sources.get(sample_id)
            if source is None:
                raise FileNotFoundError(f"No IR image for RGB sample {sample_id}")
            label = rgb / "labels" / split / f"{sample_id}.txt"
            if not label.is_file():
                raise FileNotFoundError(label)
            entries.append((source, label, split))
        if not entries:
            raise ValueError(f"Empty RGB image list: {name}")
        lists[name] = entries

    for name, entries in lists.items():
        paths = []
        for source, label, split in entries:
            target = output / "images" / split / source.name
            target_label = output / "labels" / split / label.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copy2(source, target)
            shutil.copyfile(label, target_label)
            paths.append(target.as_posix())
        (output / f"{name}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")

    for config, prefix in (("ir", ""), ("ir_smoke", "smoke_")):
        (output / f"{config}.yaml").write_text(
            f"path: {json.dumps(output.as_posix(), ensure_ascii=False)}\n"
            f"train: {prefix}train.txt\nval: {prefix}val.txt\n"
            "test: test.txt\nnames: {0: person}\n",
            encoding="utf-8",
        )
    print({name: len(entries) for name, entries in lists.items()})
    print(output)


if __name__ == "__main__":
    main()
