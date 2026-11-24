from pathlib import Path
from collections import Counter
import csv
import xml.etree.ElementTree as ET
from PIL import Image

# 1. 数据集路径：根据你的实际文件夹设置
DATA_ROOT = Path(r"E:\LLVIP_Project\dataset\LLVIP")
VISIBLE_DIR = DATA_ROOT / "visible"
INFRARED_DIR = DATA_ROOT / "infrared"
ANNOTATIONS_DIR = DATA_ROOT / "Annotations"
RESULT_DIR = Path(r"E:\LLVIP_Project\results")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_images(folder):
    """递归查找目录中的所有图像。"""
    if not folder.exists():
        raise FileNotFoundError(f"目录不存在：{folder}")

    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def get_key(path, root):
    """使用相对路径和文件名匹配两种模态。"""
    return str(path.relative_to(root).with_suffix("")).replace("\\", "/").lower()


def analyze_images(paths, modality, root, writer):
    """读取图像信息，统计尺寸、格式和文件大小。"""
    resolutions = Counter()
    formats = Counter()
    total_bytes = 0
    valid_count = 0
    failed_files = []

    for path in paths:
        try:
            with Image.open(path) as img:
                width, height = img.size
                img_format = img.format or path.suffix.lstrip(".").upper()
                img_mode = img.mode

                # verify() 前先读取元数据；这里实际解码图像以检查可读性
                img.load()

            size_bytes = path.stat().st_size
            resolutions[(width, height)] += 1
            formats[img_format] += 1
            total_bytes += size_bytes
            valid_count += 1

            writer.writerow([
                modality,
                str(path.relative_to(root)),
                width,
                height,
                img_format,
                img_mode,
                round(size_bytes / 1024, 2)
            ])

        except Exception as exc:
            failed_files.append((str(path), str(exc)))

    avg_kb = total_bytes / valid_count / 1024 if valid_count else 0

    return {
        "valid_count": valid_count,
        "resolutions": resolutions,
        "formats": formats,
        "avg_kb": avg_kb,
        "failed_files": failed_files
    }


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. 扫描两种模态
    visible = get_images(VISIBLE_DIR)
    infrared = get_images(INFRARED_DIR)

    # 3. 根据相对路径和文件名匹配图像对
    visible_keys = {get_key(p, VISIBLE_DIR) for p in visible}
    infrared_keys = {get_key(p, INFRARED_DIR) for p in infrared}

    paired_keys = visible_keys & infrared_keys
    visible_only = visible_keys - infrared_keys
    infrared_only = infrared_keys - visible_keys

    # 4. 输出每张图像的详细信息到 CSV
    csv_path = RESULT_DIR / "image_statistics.csv"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "模态", "相对路径", "宽度(px)", "高度(px)",
            "格式", "颜色模式", "文件大小(KB)"
        ])

        visible_stats = analyze_images(
            visible, "可见光", VISIBLE_DIR, writer
        )
        infrared_stats = analyze_images(
            infrared, "红外", INFRARED_DIR, writer
        )

    # 5. 统计标注文件
    annotation_files = []
    annotation_exts = Counter()
    xml_boxes = 0
    xml_count = 0
    xml_errors = 0

    if ANNOTATIONS_DIR.exists():
        annotation_files = [
            p for p in ANNOTATIONS_DIR.rglob("*") if p.is_file()
        ]

        for path in annotation_files:
            annotation_exts[path.suffix.lower() or "[无扩展名]"] += 1

            if path.suffix.lower() == ".xml":
                xml_count += 1
                try:
                    root = ET.parse(path).getroot()
                    xml_boxes += len(root.findall(".//object"))
                except (ET.ParseError, OSError):
                    xml_errors += 1

    # 6. 生成文字报告
    lines = [
        "LLVIP 数据集读取与统计结果",
        "=" * 40,
        f"数据集路径：{DATA_ROOT}",
        f"可见光图像文件数：{len(visible)} 张",
        f"红外图像文件数：{len(infrared)} 张",
        f"匹配图像对数：{len(paired_keys)} 对",
        f"仅有可见光的匹配键数：{len(visible_only)}",
        f"仅有红外的匹配键数：{len(infrared_only)}",
        "",
        f"可见光成功读取：{visible_stats['valid_count']} 张",
        f"红外成功读取：{infrared_stats['valid_count']} 张",
        f"可见光平均文件大小：{visible_stats['avg_kb']:.2f} KB/张",
        f"红外平均文件大小：{infrared_stats['avg_kb']:.2f} KB/张",
        "",
        "可见光分辨率统计（前10项）："
    ]

    for (w, h), count in visible_stats["resolutions"].most_common(10):
        lines.append(f"  {w} × {h} 像素：{count} 张")

    lines.append("")
    lines.append("红外分辨率统计（前10项）：")

    for (w, h), count in infrared_stats["resolutions"].most_common(10):
        lines.append(f"  {w} × {h} 像素：{count} 张")

    lines.extend([
        "",
        f"标注文件总数：{len(annotation_files)} 个",
        f"标注文件类型统计：{dict(annotation_exts)}",
        f"XML 标注文件数：{xml_count} 个",
        f"XML 中的 object 标注框总数：{xml_boxes} 个",
        f"XML 解析失败数：{xml_errors} 个",
        "",
        "注意：图像对按相对路径和文件名匹配，需结合 README 核实。",
        "XML 标注框统计仅适用于包含 object 节点的 XML 格式。"
    ])

    all_failures = (
        visible_stats["failed_files"] + infrared_stats["failed_files"]
    )
    lines.append(f"无法读取的图像数：{len(all_failures)}")

    report_path = RESULT_DIR / "dataset_summary.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\n详细统计表已保存：{csv_path}")
    print(f"文字统计报告已保存：{report_path}")

    if all_failures:
        print("\n无法读取的图像示例：")
        for path, error in all_failures[:5]:
            print(path, error)
            
if __name__ == "__main__":
    main()
