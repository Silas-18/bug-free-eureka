
from pathlib import Path
import shutil

# 两个模态的数据集根目录
datasets = {
    "IR": {
        "root": Path(r"E:\LLVIP_Project\dataset\llvip_ir_721"),
        "old": "E:/LoadDown/LLVIP/datasets/llvip_ir_721",
    },
    "RGB": {
        "root": Path(r"E:\LLVIP_Project\dataset\llvip_rgb_721"),
        "old": "E:/LoadDown/LLVIP/datasets/llvip_rgb_721",
    },
}

for modality, info in datasets.items():
    root = info["root"]
    old_prefix = info["old"]
    new_prefix = root.as_posix()

    print(f"\n正在检查 {modality} 数据集...")

    for filename in ["train.txt", "val.txt", "test.txt"]:
        txt_file = root / filename

        if not txt_file.exists():
            print(f"[缺失] {txt_file}")
            continue

        original = txt_file.read_text(encoding="utf-8-sig")
        lines = original.splitlines()

        updated_lines = []
        missing = []

        for line in lines:
            path_text = line.strip()

            if not path_text:
                updated_lines.append(line)
                continue

            # 替换旧数据集路径
            path_text = path_text.replace(
                old_prefix, new_prefix
            )
            path_text = path_text.replace(
                old_prefix.replace("/", "\\"),
                new_prefix
            )

            updated_lines.append(path_text)

            # 检查图片是否真实存在
            image_path = Path(path_text.replace("/", "\\"))
            if not image_path.is_absolute():
                image_path = root / image_path

            if not image_path.is_file():
                missing.append(path_text)

        if missing:
            print(f"[异常] {filename} 有 {len(missing)} 个路径不存在")
            for item in missing[:5]:
                print("   ", item)
            print("该文件暂不修改，请检查路径。")
            continue

        # 修改前自动备份
        backup = txt_file.with_suffix(".txt.bak")
        shutil.copy2(txt_file, backup)

        # 写入新路径
        txt_file.write_text(
            "\n".join(updated_lines) + "\n",
            encoding="utf-8"
        )

        print(f"[成功] {filename}")
        print(f"       图片路径全部检查通过")
        print(f"       原文件备份：{backup.name}")

print("\n路径检查结束！")
