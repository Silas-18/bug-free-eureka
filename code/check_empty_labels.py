
from pathlib import Path

ROOT = Path(r"E:\LLVIP_Project\dataset")

datasets = {
    "IR": ROOT / "llvip_ir_721",
    "RGB": ROOT / "llvip_rgb_721",
}

empty_results = {}

for mode, dataset_dir in datasets.items():
    print(f"\n========== {mode} ==========")
    empty_results[mode] = {}

    for split in ["train", "val", "test"]:
        label_dir = dataset_dir / "labels" / split
        empty_files = []

        for file in label_dir.glob("*.txt"):
            content = file.read_text(
                encoding="utf-8-sig"
            ).strip()

            if not content:
                empty_files.append(file)

        empty_results[mode][split] = {
            file.stem for file in empty_files
        }

        print(f"\n{split}: 空标签数量 = {len(empty_files)}")

        for file in empty_files:
            print("空标签:", file)

            image = (
                dataset_dir / "images" / split
                / f"{file.stem}.jpg"
            )

            print("对应图片:", image)
            print("图片是否存在:", image.exists())

print("\n========== IR/RGB 对比 ==========")

for split in ["train", "val", "test"]:
    ir = empty_results["IR"][split]
    rgb = empty_results["RGB"][split]

    print(f"\n{split}:")
    print("IR 空标签文件名:", ir)
    print("RGB 空标签文件名:", rgb)
    print("是否一致:", ir == rgb)
