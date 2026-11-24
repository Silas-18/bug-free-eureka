"""Create a small RGB/IR paired-image demo from the raw LLVIP dataset."""
import argparse
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="LLVIP paired-modality sample demo")
    p.add_argument("--raw-root", type=Path, default=root / "dataset/LLVIP")
    p.add_argument("--output", type=Path, default=root / "results/paired_demo.png")
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    visible_root = args.raw_root / "visible"
    ir_root = args.raw_root / "infrared"
    if not visible_root.is_dir() or not ir_root.is_dir():
        raise FileNotFoundError(
            f"找不到 visible/infrared。请先下载并解压 LLVIP 到：{args.raw_root}"
        )

    def index(folder):
        result = {}
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in EXTS:
                key = path.relative_to(folder).with_suffix("").as_posix().lower()
                result[key] = path
        return result

    visible, infrared = index(visible_root), index(ir_root)
    common = sorted(set(visible) & set(infrared))
    if not common:
        raise RuntimeError("未找到可见光/红外配对图像，请检查数据目录层级。")
    random.Random(args.seed).shuffle(common)
    selected = common[:min(args.num_samples, len(common))]

    font = ImageFont.load_default()
    cell_w, img_h, header_h, gap = 640, 420, 54, 16
    canvas = Image.new("RGB", (cell_w * 2 + gap, (img_h + header_h + gap) * len(selected)), "white")
    draw = ImageDraw.Draw(canvas)

    for i, key in enumerate(selected):
        y = i * (img_h + header_h + gap)
        for j, (folder, label) in enumerate(((visible, "Visible / RGB"), (infrared, "Infrared / IR"))):
            x = j * (cell_w + gap)
            draw.text((x + 8, y + 8), f"{label} | {Path(key).name}", fill="black", font=font)
            with Image.open(folder[key]) as im:
                im = im.convert("RGB")
                im.thumbnail((cell_w - 16, img_h - 12), Image.Resampling.LANCZOS)
                px = x + (cell_w - im.width) // 2
                py = y + header_h + (img_h - im.height) // 2
                canvas.paste(im, (px, py))
            draw.rectangle((x, y + header_h, x + cell_w - 1, y + header_h + img_h - 1), outline="#999999", width=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"配对样本演示图已保存：{args.output.resolve()}")
    print(f"显示样本对：{len(selected)} / 可配对样本总数：{len(common)}")


if __name__ == "__main__":
    main()
