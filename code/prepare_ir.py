"""Build the IR YOLO dataset using exactly the RGB split lists."""
import argparse
import json
import os
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Prepare LLVIP IR YOLO dataset")
    parser.add_argument("--raw-root", type=Path, default=project_root / "dataset" / "LLVIP")
    parser.add_argument("--rgb-root", type=Path, default=project_root / "dataset" / "llvip_rgb_721")
    parser.add_argument("--output-root", type=Path, default=project_root / "dataset" / "llvip_ir_721")
    parser.add_argument("--overwrite", action="store_true",
                        help="删除并重新生成 output-root；默认不覆盖已有数据")
    return parser.parse_args()


def prepare_output(path: Path, overwrite: bool):
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"输出目录非空：{path}\n如确认需要重新生成，请添加 --overwrite。"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    raw_root = args.raw_root.resolve()
    rgb_root = args.rgb_root.resolve()
    output = args.output_root.resolve()
    ir_root = raw_root / "infrared"
    if not ir_root.is_dir():
        raise FileNotFoundError(f"找不到红外目录：{ir_root}")
    if not rgb_root.is_dir():
        raise FileNotFoundError(f"找不到 RGB 处理结果：{rgb_root}，请先运行 prepare_rgb.py")

    infrared = {}
    for path in sorted(ir_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        if path.stem in infrared:
            raise ValueError(f"发现重复 IR 文件 ID：{path.stem}")
        infrared[path.stem] = path

    split_names = ("train", "val", "test", "smoke_train", "smoke_val")
    source_lists = {}
    for name in split_names:
        list_path = rgb_root / f"{name}.txt"
        if not list_path.is_file():
            if name.startswith("smoke_"):
                continue
            raise FileNotFoundError(f"缺少 RGB 划分文件：{list_path}")
        rows = [line.strip().strip('"') for line in list_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"RGB 列表为空：{list_path}")
        source_lists[name] = rows

    required_ids = {Path(row).stem for rows in source_lists.values() for row in rows}
    missing = sorted(required_ids - set(infrared))
    if missing:
        raise FileNotFoundError(f"缺少 {len(missing)} 张对应 IR 图像，示例：{missing[:5]}")

    prepare_output(output, args.overwrite)
    for name, rows in source_lists.items():
        split = name.removeprefix("smoke_")
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        output_paths = []
        for row in rows:
            sample_id = Path(row).stem
            source = infrared[sample_id]
            target = image_dir / source.name
            if not target.exists():
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copy2(source, target)
            rgb_label = rgb_root / "labels" / split / f"{sample_id}.txt"
            if not rgb_label.is_file():
                raise FileNotFoundError(f"缺少 RGB 对应标签：{rgb_label}")
            shutil.copyfile(rgb_label, label_dir / rgb_label.name)
            output_paths.append(target.resolve().as_posix())
        (output / f"{name}.txt").write_text("\n".join(output_paths) + "\n", encoding="utf-8")

    for config, prefix in (("ir", ""), ("ir_smoke", "smoke_")):
        (output / f"{config}.yaml").write_text(
            f"path: {json.dumps(output.as_posix(), ensure_ascii=False)}\n"
            f"train: {prefix}train.txt\nval: {prefix}val.txt\n"
            "test: test.txt\nnames: {0: person}\n",
            encoding="utf-8",
        )

    print("LLVIP IR 数据准备完成")
    print(f"数据划分：{ {name: len(rows) for name, rows in source_lists.items()} }")
    print(f"输出目录：{output}")
    print(f"训练配置：{output / 'ir.yaml'}")


if __name__ == "__main__":
    main()
