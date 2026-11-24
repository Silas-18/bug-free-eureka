
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET

# 数据集根目录
root = Path(r"E:\LLVIP_Project\dataset\LLVIP")

visible_dir = root / "visible"
infrared_dir = root / "infrared"
annotation_dir = root / "Annotations"

# 1. 统计文件数量
def get_files(folder, suffixes):
    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in suffixes
    ]

visible = get_files(visible_dir, {".jpg", ".jpeg", ".png"})
infrared = get_files(infrared_dir, {".jpg", ".jpeg", ".png"})
xml_files = get_files(annotation_dir, {".xml"})

print("=" * 55)
print("LLVIP 数据集结构检查")
print("=" * 55)

print("可见光图像数量：", len(visible))
print("红外图像数量：", len(infrared))
print("XML 标注数量：", len(xml_files))

# 2. 检查图像配对
def get_keys(files, folder):
    return {
        p.relative_to(folder).with_suffix("").as_posix()
        for p in files
    }

v_keys = get_keys(visible, visible_dir)
i_keys = get_keys(infrared, infrared_dir)
a_keys = get_keys(xml_files, annotation_dir)

print("\n图像配对检查：")
print("可见光与红外匹配：", len(v_keys & i_keys))
print("可见光与标注匹配：", len(v_keys & a_keys))
print("仅有可见光：", len(v_keys - i_keys))
print("仅有红外：", len(i_keys - v_keys))
print("缺少标注：", len(v_keys - a_keys))

# 3. 统计 XML 中的目标类别
class_counter = Counter()
object_count = 0
error_count = 0

for xml_path in xml_files:
    try:
        tree = ET.parse(xml_path)
        xml_root = tree.getroot()

        for obj in xml_root.findall("object"):
            name = obj.findtext("name", default="unknown")
            class_counter[name] += 1
            object_count += 1

    except Exception as e:
        error_count += 1

print("\n目标类别统计：")
for name, count in class_counter.items():
    print(f"{name}: {count}")

print("目标标注总数：", object_count)
print("XML 解析失败数：", error_count)

# 4. 展示一个 XML 样本
print("\nXML 标注样本：")

if xml_files:
    sample = ET.parse(xml_files[0]).getroot()

    print("文件名：", xml_files[0].name)
    print("图片名称：", sample.findtext("filename"))
    print("图片宽度：", sample.findtext("size/width"))
    print("图片高度：", sample.findtext("size/height"))

    for obj in sample.findall("object")[:5]:
        print("类别：", obj.findtext("name"))
        print("边界框：",
              [obj.findtext(f"bndbox/{x}")
               for x in ["xmin", "ymin", "xmax", "ymax"]])

print("\n数据集检查完成！")
