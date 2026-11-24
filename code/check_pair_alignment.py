
import random
import csv
from pathlib import Path
from collections import Counter

from PIL import Image, ImageDraw, ImageFont, ImageOps


# =========================
# 1. 项目路径配置
# =========================

PROJECT_ROOT = Path(r"E:\LLVIP_Project")

RGB_ROOT = PROJECT_ROOT / "dataset" / "llvip_rgb_721"
IR_ROOT = PROJECT_ROOT / "dataset" / "llvip_ir_721"

OUTPUT_DIR = PROJECT_ROOT / "results"

SAMPLE_NUM = 20
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# =========================
# 2. 获取图像目录
# =========================

def get_image_root(root):
    """
    优先读取 YOLO 数据集中的 images 文件夹。
    """

    image_dir = root / "images"

    if image_dir.exists():
        return image_dir

    return root


# =========================
# 3. 遍历图像
# =========================

def collect_images(root):

    image_root = get_image_root(root)

    if not image_root.exists():
        raise FileNotFoundError(
            f"找不到图像目录：{image_root}"
        )

    images = {}

    for path in image_root.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        relative_path = path.relative_to(image_root)

        key = relative_path.as_posix().lower()

        if key in images:
            print("重复文件：", key)
            continue

        images[key] = path

    return images


# =========================
# 4. 图像配对
# =========================

def get_split(path):

    for part in path.parts:

        if part.lower() in ("train", "val", "test"):
            return part.lower()

    return ""


def match_images(rgb_images, ir_images):

    pairs = []

    used_ir = set()

    ir_by_stem = {}

    for ir_path in ir_images.values():

        stem = ir_path.stem.lower()
        split = get_split(ir_path)

        key = (split, stem)

        ir_by_stem.setdefault(key, []).append(ir_path)

    for rgb_key, rgb_path in rgb_images.items():

        # 优先按照相对路径匹配
        ir_path = ir_images.get(rgb_key)

        if ir_path is not None:

            if str(ir_path) in used_ir:
                continue

            pairs.append((rgb_path, ir_path))
            used_ir.add(str(ir_path))
            continue

        # 如果目录结构不同，则按照文件名和数据划分匹配
        key = (get_split(rgb_path), rgb_path.stem.lower())

        candidates = ir_by_stem.get(key, [])

        candidates = [
            p for p in candidates
            if str(p) not in used_ir
        ]

        if len(candidates) == 1:

            ir_path = candidates[0]

            pairs.append((rgb_path, ir_path))
            used_ir.add(str(ir_path))

    return pairs


# =========================
# 5. 检查图像是否正常
# =========================

def inspect_image(path):

    try:

        with Image.open(path) as img:

            img.verify()

        with Image.open(path) as img:

            return img.size, True

    except Exception as e:

        print("异常图像：", path, e)

        return None, False


# =========================
# 6. 随机样本可视化
# =========================

def create_visualization(pairs):

    if not pairs:
        print("没有可用于可视化的配对图像！")
        return

    random.seed(RANDOM_SEED)

    samples = random.sample(
        pairs,
        min(SAMPLE_NUM, len(pairs))
    )

    columns = 2

    card_width = 700
    card_height = 300

    rows = (len(samples) + columns - 1) // columns

    canvas = Image.new(
        "RGB",
        (columns * card_width, rows * card_height),
        "white"
    )

    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    for i, (rgb_path, ir_path) in enumerate(samples):

        x = (i % columns) * card_width
        y = (i // columns) * card_height

        with Image.open(rgb_path) as img:
            rgb = img.convert("RGB")

        with Image.open(ir_path) as img:
            ir = img.convert("RGB")

        rgb_thumb = ImageOps.contain(
            rgb,
            (330, 235)
        )

        ir_thumb = ImageOps.contain(
            ir,
            (330, 235)
        )

        draw.text(
            (x + 10, y + 5),
            f"Sample {i + 1}: {rgb_path.stem}",
            fill="black",
            font=font
        )

        rgb_x = x + 10
        ir_x = x + 350
        image_y = y + 40

        canvas.paste(rgb_thumb, (rgb_x, image_y))
        canvas.paste(ir_thumb, (ir_x, image_y))

        draw.text(
            (rgb_x, y + 275),
            "Visible / RGB",
            fill="black",
            font=font
        )

        draw.text(
            (ir_x, y + 275),
            "Infrared / IR",
            fill="black",
            font=font
        )

    save_path = OUTPUT_DIR / "pair_alignment_samples.png"

    canvas.save(save_path)

    print("\n可视化图片已保存：")
    print(save_path)


# =========================
# 7. 主程序
# =========================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 50)
    print("LLVIP RGB-IR 数据配对检查")
    print("=" * 50)

    print("\n正在读取 RGB 图像...")

    rgb_images = collect_images(RGB_ROOT)

    print("RGB 图像数量：", len(rgb_images))

    print("\n正在读取 IR 图像...")

    ir_images = collect_images(IR_ROOT)

    print("IR 图像数量：", len(ir_images))

    print("\n正在进行图像配对...")

    pairs = match_images(rgb_images, ir_images)

    print("成功配对数量：", len(pairs))

    print(
        "RGB 未配对数量：",
        len(rgb_images) - len(pairs)
    )

    print(
        "IR 未配对数量：",
        len(ir_images) - len(pairs)
    )

    print("\n正在检查图像尺寸和完整性...")

    size_mismatch = 0
    invalid_images = 0

    size_counter = Counter()

    report_path = OUTPUT_DIR / "pair_alignment_report.csv"

    with open(
        report_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "文件名",
            "RGB路径",
            "IR路径",
            "RGB尺寸",
            "IR尺寸",
            "尺寸是否一致",
            "图像是否正常"
        ])

        valid_pairs = []

        for rgb_path, ir_path in pairs:

            rgb_size, rgb_ok = inspect_image(rgb_path)

            ir_size, ir_ok = inspect_image(ir_path)

            valid = rgb_ok and ir_ok

            same_size = (
                rgb_size == ir_size
                if valid else False
            )

            if valid:
                valid_pairs.append((rgb_path, ir_path))

                size_counter[
                    (rgb_size, ir_size)
                ] += 1

            else:
                invalid_images += 1

            if valid and not same_size:
                size_mismatch += 1

            writer.writerow([
                rgb_path.stem,
                str(rgb_path),
                str(ir_path),
                rgb_size,
                ir_size,
                same_size,
                valid
            ])

    print("\n========== 检查结果 ==========")

    print("RGB 图像总数：", len(rgb_images))
    print("IR 图像总数：", len(ir_images))
    print("成功配对数量：", len(pairs))
    print("有效配对数量：", len(valid_pairs))
    print("尺寸不一致数量：", size_mismatch)
    print("异常图像数量：", invalid_images)

    print("\n常见图像尺寸组合：")

    for size, count in size_counter.most_common(10):

        print(
            f"RGB: {size[0]} | IR: {size[1]} | 数量: {count}"
        )

    print("\nCSV 检查报告：")
    print(report_path)

    print("\n正在生成随机样本可视化...")

    create_visualization(valid_pairs)

    print("\n全部检查完成！")


if __name__ == "__main__":
    main()
