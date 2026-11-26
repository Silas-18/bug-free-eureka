
from pathlib import Path
from ultralytics import YOLO

# 1. 数据集路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
dataset_dir = PROJECT_ROOT / "dataset" / "LLVIP"
visible_dir = dataset_dir / "visible"

# 2. 检查数据集
if not visible_dir.exists():
    raise FileNotFoundError(f"找不到可见光文件夹：{visible_dir}")

# 3. 自动寻找一张可见光图片
image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

images = sorted(
    p for p in visible_dir.rglob("*")
    if p.is_file() and p.suffix.lower() in image_extensions
)

if not images:
    raise FileNotFoundError("没有找到可见光图像")

image_path = images[0]

print("测试图片：", image_path)

# 4. 加载YOLOv8n
model = YOLO("yolov8n.pt")

# 5. 使用GPU进行目标检测
results = model.predict(
    source=str(image_path),
    device=0,
    imgsz=640,
    conf=0.25,
    save=True,
    project=str(PROJECT_ROOT / "results"),
    name="yolo_visible_test",
    exist_ok=True
)

# 6. 输出检测结果
result = results[0]

print("\n========== 检测结果 ==========")
print("图片尺寸：", result.orig_shape)
print("检测目标数量：", len(result.boxes))
print("检测类别：", result.boxes.cls.tolist())
print("检测置信度：", result.boxes.conf.tolist())
print("结果已保存至：", PROJECT_ROOT / "results" / "yolo_visible_test")
print("YOLO GPU推理测试完成！")
