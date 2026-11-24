
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from fusion_model import DualModalYOLOv8
from fusion_dataset import (
    LLVIPFusionDataset,
    fusion_collate_fn
)

ROOT = Path(__file__).resolve().parents[1]

device = torch.device("cuda:0")

# 加载真实 RGB-IR 配对数据
dataset = LLVIPFusionDataset(
    root=ROOT / "dataset",
    split="train",
    imgsz=640
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    collate_fn=fusion_collate_fn
)

# 加载双模态融合模型
model = DualModalYOLOv8(
    rgb_weights=str(
        ROOT / "experiments/baseline/v8_rgb_baseline/weights/best.pt"
    ),
    ir_weights=str(
        ROOT / "experiments/baseline/v8_ir_baseline/weights/best.pt"
    )
).to(device)

model.train()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0001,
    weight_decay=0.0005
)

# 只测试一个真实训练批次
rgb, ir, targets = next(iter(loader))

rgb = rgb.to(device).float() / 255.0
ir = ir.to(device).float() / 255.0

targets = {
    k: v.to(device)
    for k, v in targets.items()
}

print("RGB 输入尺寸：", rgb.shape)
print("IR 输入尺寸：", ir.shape)
print("行人标注数量：", len(targets["cls"]))

optimizer.zero_grad(set_to_none=True)

# 双模态前向传播
preds = model(rgb, ir)

# YOLO 检测损失
loss, loss_items = model.loss(targets, preds)

# 将三个损失分量求和，得到总损失
total_loss = loss.sum()

print("训练总损失：", total_loss.item())
print("损失分量：", loss.detach().cpu().tolist())

# 反向传播
total_loss.backward()

# 更新模型参数
optimizer.step()

print("反向传播成功！")
print("参数更新成功！")
print("真实数据训练步骤测试完成！")
