
import torch
from fusion_model import DualModalYOLOv8

# 加载之前训练好的两个模型
rgb_weights = (
    r"experiments/baseline/v8_rgb_baseline/weights/best.pt"
)

ir_weights = (
    r"experiments/baseline/v8_ir_baseline/weights/best.pt"
)

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

print("使用设备：", device)

# 创建融合模型
model = DualModalYOLOv8(
    rgb_weights=rgb_weights,
    ir_weights=ir_weights
).to(device)

model.eval()

# 创建两张模拟输入图片
rgb = torch.randn(1, 3, 640, 640).to(device)
ir = torch.randn(1, 3, 640, 640).to(device)

print("开始测试双模态前向传播...")

with torch.no_grad():
    output = model(rgb, ir)

print("前向传播成功！")

if isinstance(output, (tuple, list)):
    print("输出类型：", type(output))
    if torch.is_tensor(output[0]):
        print("预测结果尺寸：", output[0].shape)
else:
    print("预测结果尺寸：", output.shape)

print("双模态融合模型测试完成！")
