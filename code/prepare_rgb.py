"""Build the RGB YOLO dataset from the downloaded LLVIP raw folder.

Default layout:
  dataset/LLVIP/{visible,infrared,Annotations}
Output:
  dataset/llvip_rgb_721
The script deliberately preserves the project's existing 70/20/10 random
split over all 15,488 paired samples (seed=42); see README for this choice.
"""
import argparse
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Prepare LLVIP RGB YOLO dataset")
    parser.add_argument("--raw-root", type=Path, default=project_root / "dataset" / "LLVIP")
    parser.add_argument("--output-root", type=Path, default=project_root / "dataset" / "llvip_rgb_721")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--smoke-train", type=int, default=64)
    parser.add_argument("--smoke-val", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true",
                        help="删除并重新生成 output-root；默认不覆盖已有数据")
    return parser.parse_args()


def prepare_output(path: Path, overwrite: bool):
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"输出目录非空：{path}\n"
                "如确认需要重新生成，请添加 --overwrite。"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def index_images(folder: Path):
    if not folder.is_dir():
        raise FileNotFoundError(f"数据目录不存在：{folder}")
    result = {}
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        parts = [part.lower() for part in path.parts]
        split = next((x for x in ("train", "test") if x in parts), None)
        if split is None:
            raise ValueError(f"无法从路径判断官方数据划分：{path}")
        if path.stem in result:
            raise ValueError(f"发现重复图像 ID：{path.stem}")
        result[path.stem] = (path, split)
    return result


def copy_or_link(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def xml_to_yolo(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    width = int(root.findtext("size/width", "0"))
    height = int(root.findtext("size/height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"XML 中图像尺寸无效：{xml_path}")
    rows = []
    invalid = False
    for obj in root.findall("object"):
        if obj.findtext("name", "").strip() != "person":
            continue
        box = obj.find("bndbox")
        if box is None:
            invalid = True
            continue
        x1, y1, x2, y2 = (
            float(box.findtext(key, "0"))
            for key in ("xmin", "ymin", "xmax", "ymax")
        )
        x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
        y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            invalid = True
            continue
        xc, yc = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
        bw, bh = (x2 - x1) / width, (y2 - y1) / height
        rows.append(f"0 {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}")
    return rows, invalid


def main():
    args = parse_args()
    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train-ratio 与 val-ratio 必须大于0，且两者之和小于1。")

    raw = args.raw_root.resolve()
    output = args.output_root.resolve()
    visible = index_images(raw / "visible")
    infrared = index_images(raw / "infrared")
    annotation_dir = raw / "Annotations"
    if not annotation_dir.is_dir():
        raise FileNotFoundError(f"找不到 XML 标注目录：{annotation_dir}")

    annotations = {}
    for xml_path in sorted(annotation_dir.rglob("*.xml")):
        if xml_path.stem in annotations:
            raise ValueError(f"发现重复 XML ID：{xml_path.stem}")
        annotations[xml_path.stem] = xml_path

    if set(visible) != set(infrared):
        raise ValueError(
            f"RGB/IR 文件名不一致：仅 RGB {len(set(visible)-set(infrared))}，"
            f"仅 IR {len(set(infrared)-set(visible))}"
        )
    if set(visible) != set(annotations):
        raise ValueError(
            f"图像与标注不一致：缺少 XML {len(set(visible)-set(annotations))}，"
            f"多余 XML {len(set(annotations)-set(visible))}"
        )
    for sample_id in visible:
        if visible[sample_id][1] != infrared[sample_id][1]:
            raise ValueError(f"RGB/IR 官方划分不一致：{sample_id}")

    ids = sorted(visible)
    if len(ids) == 0:
        raise RuntimeError("没有发现 LLVIP 图像。")
    random.Random(args.seed).shuffle(ids)
    train_end = round(len(ids) * args.train_ratio)
    val_end = round(len(ids) * (args.train_ratio + args.val_ratio))
    splits = {
        "train": ids[:train_end],
        "val": ids[train_end:val_end],
        "test": ids[val_end:],
    }

    prepare_output(output, args.overwrite)
    invalid_ids = []
    for split, sample_ids in splits.items():
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        image_list = []
        for sample_id in sample_ids:
            image_path = visible[sample_id][0]
            rows, invalid = xml_to_yolo(annotations[sample_id])
            if invalid:
                invalid_ids.append(sample_id)
            target_image = image_dir / image_path.name
            copy_or_link(image_path, target_image)
            (label_dir / f"{sample_id}.txt").write_text("\n".join(rows), encoding="utf-8")
            image_list.append(target_image.resolve().as_posix())
        (output / f"{split}.txt").write_text("\n".join(image_list) + "\n", encoding="utf-8")

    for name, source_split, limit in (
        ("smoke_train", "train", args.smoke_train),
        ("smoke_val", "val", args.smoke_val),
    ):
        source_lines = (output / f"{source_split}.txt").read_text(encoding="utf-8").splitlines()
        selected = source_lines[:max(0, limit)]
        (output / f"{name}.txt").write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")

    (output / "split_ids.json").write_text(
        json.dumps(splits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "invalid_boxes.txt").write_text("\n".join(sorted(set(invalid_ids))), encoding="utf-8")
    yaml_text = (
        f"path: {json.dumps(output.as_posix(), ensure_ascii=False)}\n"
        "train: train.txt\nval: val.txt\ntest: test.txt\n"
        "names: {0: person}\n"
    )
    (output / "rgb.yaml").write_text(yaml_text, encoding="utf-8")
    smoke_yaml = (
        f"path: {json.dumps(output.as_posix(), ensure_ascii=False)}\n"
        "train: smoke_train.txt\nval: smoke_val.txt\ntest: test.txt\n"
        "names: {0: person}\n"
    )
    (output / "rgb_smoke.yaml").write_text(smoke_yaml, encoding="utf-8")

    print("LLVIP RGB 数据准备完成")
    print(f"原始样本对：{len(ids)}")
    print("划分数量：" + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    print(f"存在被裁剪/无效框记录的样本数：{len(set(invalid_ids))}")
    print(f"输出目录：{output}")
    print(f"训练配置：{output / 'rgb.yaml'}")


if __name__ == "__main__":
    main()
