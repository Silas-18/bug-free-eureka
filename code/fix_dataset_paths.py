from pathlib import Path, PurePosixPath
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "IR": PROJECT_ROOT / "dataset" / "llvip_ir_721",
    "RGB": PROJECT_ROOT / "dataset" / "llvip_rgb_721",
}

def resolve_image(path_text, root):
    normalized = path_text.strip().strip('"').replace("\\", "/")
    if not normalized:
        return None

    parts = PurePosixPath(normalized).parts

    if root.name in parts:
        index = parts.index(root.name)
        relative = Path(*parts[index + 1:])
        candidates = [root / relative]
    elif "images" in parts:
        index = parts.index("images")
        relative = Path(*parts[index:])
        candidates = [root / relative]
    else:
        relative = Path(normalized)
        candidates = [root / relative, root / "images" / relative]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    return None

for modality, root in DATASETS.items():
    print(f"\n正在检查 {modality} 数据集：{root}")

    if not root.is_dir():
        print("[跳过] 数据集目录不存在，请先准备数据。")
        continue

    for split in ["train", "val", "test"]:
        txt_file = root / f"{split}.txt"

        if not txt_file.exists():
            print(f"[缺失] {txt_file}")
            continue

        lines = txt_file.read_text(encoding="utf-8-sig").splitlines()
        updated = []
        missing = []

        for line in lines:
            if not line.strip():
                updated.append(line)
                continue

            image = resolve_image(line, root)

            if image is None:
                missing.append(line)
            else:
                updated.append(image.as_posix())

        if missing:
            print(f"[异常] {split}.txt 缺少 {len(missing)} 张图片，未修改。")
            for item in missing[:5]:
                print("  ", item)
            continue

        backup = txt_file.with_suffix(".txt.bak")
        shutil.copy2(txt_file, backup)
        txt_file.write_text("\n".join(updated) + "\n", encoding="utf-8")

        print(f"[成功] {split}.txt，路径检查通过。")

print("\n路径检查结束！")
