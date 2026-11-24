
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image
import random
import shutil

# ==========================
# 1. 路径配置
# ==========================

ROOT = Path(r"E:\LLVIP_Project\dataset\LLVIP")
OUT = Path(r"E:\LLVIP_Project\dataset\yolo_ready_v1")

VISIBLE = ROOT / "visible"
INFRARED = ROOT / "infrared"
ANNOTATIONS = ROOT / "Annotations"

SEED = 42
VAL_RATIO = 0.1

# 防止覆盖之前的实验数据
if OUT.exists():
    raise RuntimeError(
        f"输出目录已经存在：{OUT}\n"
        "请先确认目录内容，必要时换一个新的输出目录名称。"
    )

# ==========================
# 2. 建立文件索引
# ==========================

def index_images(folder):
    result = {}

    for p in folder.rglob("*"):
        if p.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        stem = p.stem

        if stem in result:
            raise ValueError(f"发现重复图片文件名：{stem}")

        parts = [x.lower() for x in p.parts]

        if "train" in parts:
            split = "train"
        elif "test" in parts:
            split = "test"
        else:
            raise ValueError(f"无法判断图片所属数据集：{p}")

        result[stem] = (p, split)

    return result


visible_map = index_images(VISIBLE)
infrared_map = index_images(INFRARED)

xml_map = {}

for p in ANNOTATIONS.rglob("*.xml"):
    if p.stem in xml_map:
        raise ValueError(f"发现重复XML：{p.stem}")
    xml_map[p.stem] = p

# ==========================
# 3. 检查文件匹配
# ==========================

vset = set(visible_map)
iset = set(infrared_map)
aset = set(xml_map)

assert vset == iset, "可见光与红外文件名不一致"
assert vset == aset, "图片与XML文件名不一致"

for stem in vset:
    if visible_map[stem][1] != infrared_map[stem][1]:
        raise ValueError(f"模态划分不一致：{stem}")

print("文件匹配检查通过！")
print("总样本数：", len(vset))

# ==========================
# 4. 划分训练集、验证集、测试集
# ==========================

train_stems = [
    s for s in vset
    if visible_map[s][1] == "train"
]

test_stems = [
    s for s in vset
    if visible_map[s][1] == "test"
]

random.seed(SEED)
random.shuffle(train_stems)

val_count = max(1, round(len(train_stems) * VAL_RATIO))

val_stems = train_stems[:val_count]
new_train_stems = train_stems[val_count:]

splits = {
    "train": new_train_stems,
    "val": val_stems,
    "test": test_stems
}

print("\n数据集划分：")
for name, items in splits.items():
    print(f"{name}: {len(items)} 张")

# ==========================
# 5. XML转YOLO格式
# ==========================

def convert_xml(xml_path, image_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    with Image.open(image_path) as img:
        width, height = img.size

    lines = []

    for obj in root.findall("object"):

        name = obj.findtext("name", "").strip()

        # LLVIP 当前只使用 person 类
        if name != "person":
            continue

        box = obj.find("bndbox")

        if box is None:
            continue

        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))

        # 限制边界框在图像范围内
        xmin = max(0, min(xmin, width))
        xmax = max(0, min(xmax, width))
        ymin = max(0, min(ymin, height))
        ymax = max(0, min(ymax, height))

        if xmax <= xmin or ymax <= ymin:
            continue

        # 转换为YOLO归一化坐标
        xc = (xmin + xmax) / 2 / width
        yc = (ymin + ymax) / 2 / height

        w = (xmax - xmin) / width
        h = (ymax - ymin) / height

        lines.append(
            f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
        )

    return lines


# ==========================
# 6. 生成两种模态的数据
# ==========================

total_boxes = 0

for modality, image_map in [
    ("visible", visible_map),
    ("infrared", infrared_map)
]:

    for split, stems in splits.items():

        image_out = OUT / "images" / modality / split
        label_out = OUT / "labels" / modality / split

        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        for stem in stems:

            image_path = image_map[stem][0]
            xml_path = xml_map[stem]

            # 转换XML
            lines = convert_xml(xml_path, image_path)

            # 复制图片
            shutil.copy2(
                image_path,
                image_out / image_path.name
            )

            # 保存YOLO标签
            label_path = label_out / f"{stem}.txt"
            label_path.write_text(
                "\n".join(lines),
                encoding="utf-8"
            )

            total_boxes += len(lines)

    print(f"{modality} 数据处理完成！")

# ==========================
# 7. 生成数据集配置文件
# ==========================

for modality in ["visible", "infrared"]:

    yaml_path = OUT / f"dataset_{modality}.yaml"

    content = f"""path: {OUT.as_posix()}
train: images/{modality}/train
val: images/{modality}/val
test: images/{modality}/test

names:
  0: person
"""

    yaml_path.write_text(content, encoding="utf-8")

# ==========================
# 8. 输出统计
# ==========================

print("\n" + "=" * 50)
print("LLVIP YOLO数据集转换完成！")
print("=" * 50)

for name, items in splits.items():
    print(f"{name}: {len(items)} 张")

print("每种模态的目标框数量：", total_boxes // 2)
print("两种模态合计目标框数量：", total_boxes)
print("输出目录：", OUT)
print("可见光配置：dataset_visible.yaml")
print("红外配置：dataset_infrared.yaml")

print("\n下一步：检查标签并进行YOLO基线训练。")
