
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "dataset" / "yolo_ready_v1"

print("=" * 55)
print("LLVIP YOLO 标签完整性检查")
print("=" * 55)

extensions = {".jpg", ".jpeg", ".png"}

for modality in ["visible", "infrared"]:
    print(f"\n【{modality}】")

    for split in ["train", "val", "test"]:
        image_dir = root / "images" / modality / split
        label_dir = root / "labels" / modality / split

        images = {
            p.stem for p in image_dir.iterdir()
            if p.suffix.lower() in extensions
        }

        labels = {
            p.stem for p in label_dir.glob("*.txt")
        }

        missing = images - labels
        extra = labels - images

        print(f"\n{split}:")
        print(f"图片数量：{len(images)}")
        print(f"标签数量：{len(labels)}")
        print(f"缺少标签：{len(missing)}")
        print(f"多余标签：{len(extra)}")

        if missing:
            print("缺少标签示例：", list(missing)[:5])

        if extra:
            print("多余标签示例：", list(extra)[:5])

print("\n检查完成！")
