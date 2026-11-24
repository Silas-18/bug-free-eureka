
from ultralytics import YOLO

# IR 测试
print("========== IR TEST ==========")

ir_model = YOLO(
    r"experiments/baseline/v8_ir_baseline/weights/best.pt"
)

ir_metrics = ir_model.val(
    data=r"dataset/llvip_ir_721/ir.yaml",
    split="test",
    imgsz=640,
    batch=4,
    device=0,
    workers=0,
    project="results",
    name="ir_test",
    plots=True
)

# RGB 测试
print("========== RGB TEST ==========")

rgb_model = YOLO(
    r"experiments/baseline/v8_rgb_baseline/weights/best.pt"
)

rgb_metrics = rgb_model.val(
    data=r"dataset/llvip_rgb_721/rgb.yaml",
    split="test",
    imgsz=640,
    batch=4,
    device=0,
    workers=0,
    project="results",
    name="rgb_test",
    plots=True
)

print("两个单模态测试集评估完成！")
