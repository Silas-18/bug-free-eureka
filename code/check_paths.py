
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter

root = Path(r"E:\LLVIP_Project\dataset\LLVIP")

visible_dir = root / "visible"
infrared_dir = root / "infrared"
annotation_dir = root / "Annotations"

def get_files(folder, suffix):
    return list(folder.rglob(f"*{suffix}"))

visible = get_files(visible_dir, ".jpg")
infrared = get_files(infrared_dir, ".jpg")
annotations = get_files(annotation_dir, ".xml")

print("=" * 55)
print("一、文件夹实际路径")
print("=" * 55)

for title, files, folder in [
    ("可见光", visible, visible_dir),
    ("红外", infrared, infrared_dir),
    ("XML标注", annotations, annotation_dir)
]:
    print(f"\n{title}前10个文件：")
    for p in files[:10]:
        print(p.relative_to(folder))

print("\n" + "=" * 55)
print("二、按文件名匹配")
print("=" * 55)

def stems(files):
    return [p.stem for p in files]

v = stems(visible)
i = stems(infrared)
a = stems(annotations)

print("可见光与红外文件名匹配：",
      len(set(v) & set(i)))

print("可见光与XML文件名匹配：",
      len(set(v) & set(a)))

print("红外与XML文件名匹配：",
      len(set(i) & set(a)))

print("\n文件名重复检查：")
print("可见光重复：", len(v) - len(set(v)))
print("红外重复：", len(i) - len(set(i)))
print("XML重复：", len(a) - len(set(a)))

print("\n" + "=" * 55)
print("三、XML内部文件名检查")
print("=" * 55)

for p in annotations[:5]:
    tree = ET.parse(p)
    filename = tree.getroot().findtext("filename")
    print("XML路径：", p.relative_to(annotation_dir))
    print("XML内部filename：", filename)
    print("-" * 30)

print("\n检查完成！")
