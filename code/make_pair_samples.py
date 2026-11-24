from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

DATA_ROOT = Path(r"E:\LLVIP_Project\dataset\LLVIP")
VISIBLE_DIR = DATA_ROOT / "visible"
INFRARED_DIR = DATA_ROOT / "infrared"
OUT_DIR = Path(r"E:\LLVIP_Project\results")

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMG_W, IMG_H = 480, 384
CELL_H = 470


def find_images(folder):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTS
    )


def make_key(path, folder):
    return str(path.relative_to(folder).with_suffix("")).replace(
        "\\", "/"
    ).lower()


def load_font(size):
    for font_path in [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]:
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    visible = {
        make_key(p, VISIBLE_DIR): p
        for p in find_images(VISIBLE_DIR)
    }
    infrared = {
        make_key(p, INFRARED_DIR): p
        for p in find_images(INFRARED_DIR)
    }

    common = sorted(set(visible) & set(infrared))

    if not common:
        print("没有找到匹配图像，请检查数据集目录和文件名。")
        return

    # 从整个数据集中均匀选取最多3对样本
    n = min(3, len(common))
    indices = [round(i * (len(common) - 1) / (n - 1))
               for i in range(n)] if n > 1 else [0]
    selected = [common[i] for i in indices]

    canvas = Image.new("RGB", (IMG_W * 2 + 40, CELL_H * n), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(22)
    text_font = load_font(15)

    for row, key in enumerate(selected):
        y = row * CELL_H
        vp, ip = visible[key], infrared[key]

        draw.text((10, y + 5), f"Sample {row + 1}: {vp.stem}",
                  fill="black", font=title_font)

        for col, (path, label) in enumerate([
            (vp, "Visible"),
            (ip, "Infrared")
        ]):
            x = 10 + col * (IMG_W + 20)

            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((IMG_W, IMG_H), Image.Resampling.LANCZOS)

            draw.text((x, y + 38), label, fill="black", font=text_font)

            img_y = y + 60
            canvas.paste(im, (x, img_y))

            draw.rectangle(
                [x, img_y, x + im.width - 1, img_y + im.height - 1],
                outline="#888888", width=1
            )

    output = OUT_DIR / "paired_samples.png"
    canvas.save(output)
    print(f"配对样本图已保存：{output}")
    print(f"展示样本数：{len(selected)} 对")
    print("左侧为可见光图像，右侧为对应红外图像。")


if __name__ == "__main__":
    main()
