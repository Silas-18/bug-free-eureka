
from pathlib import Path
import math

PROJECT = Path(r"E:\LLVIP_Project")

DATASETS = {
    "IR": PROJECT / "dataset" / "llvip_ir_721",
    "RGB": PROJECT / "dataset" / "llvip_rgb_721",
}

SPLITS = ["train", "val", "test"]


def check_dataset(name, root):
    print(f"\n{'=' * 60}")
    print(f"检查 {name} 数据集")
    print(f"{'=' * 60}")

    stats = {
        "images": 0,
        "labels": 0,
        "empty": 0,
        "objects": 0,
        "invalid": 0,
        "missing": 0,
    }

    label_files = set()
    list_entries = {}

    for split in SPLITS:
        txt = root / f"{split}.txt"
        if not txt.exists():
            print(f"[错误] 缺少 {txt}")
            continue

        lines = [
            x.strip() for x in
            txt.read_text(encoding="utf-8-sig").splitlines()
            if x.strip()
        ]

        list_entries[split] = []

        for line in lines:
            image = Path(line.replace("/", "\\"))
            key = f"{split}/{image.name}"
            list_entries[split].append(key)
            stats["images"] += 1

            label = root / "labels" / split / (
                image.stem + ".txt"
            )

            if not label.is_file():
                stats["missing"] += 1
                continue

            label_files.add(f"{split}/{label.name}")
            stats["labels"] += 1

            content = label.read_text(
                encoding="utf-8-sig"
            ).strip()

            if not content:
                stats["empty"] += 1
                continue

            for line_no, row in enumerate(
                content.splitlines(), 1
            ):
                fields = row.split()

                if len(fields) != 5:
                    stats["invalid"] += 1
                    print(
                        f"[格式错误] {label}, 第{line_no}行"
                    )
                    continue

                try:
                    cls = float(fields[0])
                    x, y, w, h = map(float, fields[1:])

                    values = [cls, x, y, w, h]

                    if not all(map(math.isfinite, values)):
                        raise ValueError("非有限数值")

                    if cls != 0:
                        raise ValueError("类别编号不是0")

                    if not all(
                        0 <= v <= 1 for v in [x, y, w, h]
                    ):
                        raise ValueError("坐标超出0~1")

                    if w <= 0 or h <= 0:
                        raise ValueError("宽度或高度不合法")

                    if (
                        x - w / 2 < -1e-6
                        or y - h / 2 < -1e-6
                        or x + w / 2 > 1 + 1e-6
                        or y + h / 2 > 1 + 1e-6
                    ):
                        raise ValueError("边界框超出图像范围")

                    stats["objects"] += 1

                except ValueError as e:
                    stats["invalid"] += 1
                    print(f"[坐标错误] {label}: {e}")

    print(f"图片数量：{stats['images']}")
    print(f"找到标签：{stats['labels']}")
    print(f"空标签数量：{stats['empty']}")
    print(f"行人标注总数：{stats['objects']}")
    print(f"缺失标签：{stats['missing']}")
    print(f"错误标签：{stats['invalid']}")

    return stats, list_entries, label_files


results = {}

for name, root in DATASETS.items():
    results[name] = check_dataset(name, root)

print("\n" + "=" * 60)
print("IR / RGB 一致性检查")
print("=" * 60)

ir_stats, ir_lists, ir_files = results["IR"]
rgb_stats, rgb_lists, rgb_files = results["RGB"]

for split in SPLITS:
    a = ir_lists.get(split, [])
    b = rgb_lists.get(split, [])

    print(f"\n{split}:")
    print("IR图片数：", len(a))
    print("RGB图片数：", len(b))
    print("图片划分一致：", a == b)
    print(
        "标签文件集合一致：",
        {
            p for p in ir_files if p.startswith(split + "/")
        } == {
            p for p in rgb_files if p.startswith(split + "/")
        }
    )

print("\n检查结束！")

# ===============================
# IR / RGB 全部标签内容逐一比较
# ===============================

print("\n" + "=" * 60)
print("IR / RGB 标签内容逐一比较")
print("=" * 60)

ir_root = DATASETS["IR"] / "labels"
rgb_root = DATASETS["RGB"] / "labels"

ir_label_paths = {
    p.relative_to(ir_root).as_posix(): p
    for p in ir_root.rglob("*.txt")
}

rgb_label_paths = {
    p.relative_to(rgb_root).as_posix(): p
    for p in rgb_root.rglob("*.txt")
}

ir_keys = set(ir_label_paths)
rgb_keys = set(rgb_label_paths)

print("IR 标签文件数：", len(ir_keys))
print("RGB 标签文件数：", len(rgb_keys))
print("文件集合一致：", ir_keys == rgb_keys)

different = []

for key in sorted(ir_keys & rgb_keys):
    ir_content = ir_label_paths[key].read_text(
        encoding="utf-8-sig"
    ).strip()

    rgb_content = rgb_label_paths[key].read_text(
        encoding="utf-8-sig"
    ).strip()

    if ir_content != rgb_content:
        different.append(key)

print("标签内容不同数量：", len(different))

if different:
    print("前10个不一致文件：")
    for item in different[:10]:
        print(item)
else:
    print("所有 IR / RGB 标签内容完全一致！")

print("\n全部检查结束！")
