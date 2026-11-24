"""Validate generated RGB/IR YOLO dataset files and paired sample IDs."""
import argparse
from pathlib import Path


def read_list(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return [Path(x.strip().strip('"')) for x in path.read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=project_root / "dataset")
    args = parser.parse_args()
    roots = {
        "RGB": args.data_root / "llvip_rgb_721",
        "IR": args.data_root / "llvip_ir_721",
    }
    failed = False
    split_ids = {}

    for modality, root in roots.items():
        print("\n" + "=" * 60)
        print(f"{modality} dataset: {root}")
        split_ids[modality] = {}
        for split in ("train", "val", "test"):
            list_path = root / f"{split}.txt"
            try:
                images = read_list(list_path)
            except FileNotFoundError:
                print(f"[ERROR] missing list: {list_path}")
                failed = True
                continue
            ids = set()
            missing_images, missing_labels, empty_labels = [], [], []
            for image in images:
                ids.add(image.stem)
                if not image.is_file():
                    missing_images.append(str(image))
                    continue
                label = root / "labels" / split / f"{image.stem}.txt"
                if not label.is_file():
                    missing_labels.append(str(label))
                elif label.stat().st_size == 0:
                    empty_labels.append(str(label))
            split_ids[modality][split] = ids
            print(
                f"{split}: list={len(images)}, missing_images={len(missing_images)}, "
                f"missing_labels={len(missing_labels)}, empty_labels={len(empty_labels)}"
            )
            for title, values in (
                ("missing image", missing_images),
                ("missing label", missing_labels),
                ("empty label", empty_labels),
            ):
                for value in values[:3]:
                    print(f"  {title}: {value}")
            if missing_images or missing_labels:
                failed = True

    print("\n" + "=" * 60)
    print("RGB / IR pairing check")
    for split in ("train", "val", "test"):
        rgb = split_ids.get("RGB", {}).get(split, set())
        ir = split_ids.get("IR", {}).get(split, set())
        only_rgb, only_ir = rgb - ir, ir - rgb
        print(f"{split}: RGB={len(rgb)}, IR={len(ir)}, only_RGB={len(only_rgb)}, only_IR={len(only_ir)}")
        if only_rgb or only_ir or not rgb:
            failed = True

    if failed:
        raise SystemExit("Dataset integrity check FAILED. Please fix the paths/files above.")
    print("Dataset integrity check PASSED.")


if __name__ == "__main__":
    main()
