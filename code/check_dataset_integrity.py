
from pathlib import Path

# 项目数据集根目录
ROOT = Path(r"E:\LLVIP_Project\dataset")

DATASETS = [
    "llvip_ir_721",
    "llvip_rgb_721"
]

SPLITS = ["train", "val", "test"]

for dataset_name in DATASETS:
    dataset_dir = ROOT / dataset_name

    print("\n" + "=" * 55)
    print(f"检查数据集：{dataset_name}")
    print("=" * 55)

    for split in SPLITS:
        txt_file = dataset_dir / f"{split}.txt"

        if not txt_file.exists():
            print(f"[错误] 找不到 {split}.txt")
            continue

        lines = txt_file.read_text(
            encoding="utf-8-sig"
        ).splitlines()

        missing_images = []
        missing_labels = []
        empty_labels = []

        for line in lines:
            if not line.strip():
                continue

            img_path = Path(line.strip())

            if not img_path.is_absolute():
                img_path = dataset_dir / img_path

            if not img_path.is_file():
                missing_images.append(str(img_path))
                continue

            # images/train/xxx.jpg
            # 对应 labels/train/xxx.txt
            label_path = (
              img_path.parent.parent.parent
              / "labels"
              / img_path.parent.name
              / f"{img_path.stem}.txt"
             )

            if not label_path.is_file():
                missing_labels.append(str(label_path))
            elif label_path.stat().st_size == 0:
                empty_labels.append(str(label_path))

        print(f"\n{split}:")
        print(f"  图片清单数量：{len([x for x in lines if x.strip()])}")
        print(f"  缺失图片：{len(missing_images)}")
        print(f"  缺失标签：{len(missing_labels)}")
        print(f"  空标签：{len(empty_labels)}")

        for item in missing_images[:3]:
            print("  缺失图片示例：", item)

        for item in missing_labels[:3]:
            print("  缺失标签示例：", item)

print("\n数据集检查结束！")
