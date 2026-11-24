import torch
from pathlib import Path
from torch.utils.data import DataLoader

from fusion_model import DualModalYOLOv8
from fusion_dataset import LLVIPFusionDataset, fusion_collate_fn

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def report_gradients(name, module):
    params = [(n, p) for n, p in module.named_parameters()]
    trainable = [(n, p) for n, p in params if p.requires_grad]
    with_grad = [(n, p) for n, p in trainable if p.grad is not None]
    nonzero = [
        (n, p) for n, p in with_grad
        if torch.count_nonzero(p.grad.detach()).item() > 0
    ]

    total_sq = 0.0
    max_abs = 0.0
    for _, p in with_grad:
        g = p.grad.detach()
        total_sq += float(torch.sum(g.float() ** 2).item())
        if g.numel():
            max_abs = max(max_abs, float(g.abs().max().item()))

    global_norm = total_sq ** 0.5

    print(f"\n===== {name} 梯度统计 =====")
    print(f"参数张量数量：{len(params)}")
    print(f"requires_grad=True：{len(trainable)}")
    print(f"grad 非 None：{len(with_grad)}")
    print(f"梯度含非零元素：{len(nonzero)}")
    print(f"全模块梯度 L2 norm：{global_norm:.8e}")
    print(f"最大梯度绝对值：{max_abs:.8e}")

    # 按 YOLO 层查看，帮助定位梯度是否只在部分层传播
    layer_stats = {}
    for name_i, p in with_grad:
        layer = name_i.split(".", 1)[0]
        if layer not in layer_stats:
            layer_stats[layer] = [0.0, 0]
        layer_stats[layer][0] += float(torch.sum(p.grad.detach().float() ** 2).item())
        if torch.count_nonzero(p.grad.detach()).item() > 0:
            layer_stats[layer][1] += 1

    print("分层梯度（层名：L2 norm / 有非零梯度的参数张量数）：")
    for layer, (sq, count) in layer_stats.items():
        print(f"  {layer}: {(sq ** 0.5):.8e} / {count}")

    if not trainable:
        print("警告：该模块所有参数都被冻结（requires_grad=False）。")
    elif not with_grad:
        print("警告：该模块没有获得梯度（所有 grad 都是 None）。")
    elif not nonzero:
        print("警告：该模块获得了梯度张量，但所有梯度元素均为 0。")
    else:
        print("结论：该模块至少有一部分参数获得了非零梯度。")


def main():
    print(f"设备：{DEVICE}")
    dataset = LLVIPFusionDataset(
        root=ROOT / "dataset",
        split="train",
        imgsz=640,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=fusion_collate_fn,
    )

    model = DualModalYOLOv8(
        rgb_weights=str(
            ROOT / "experiments/baseline/v8_rgb_baseline/weights/best.pt"
        ),
        ir_weights=str(
            ROOT / "experiments/baseline/v8_ir_baseline/weights/best.pt"
        ),
    ).to(DEVICE)

    model.train()
    rgb, ir, targets = next(iter(loader))
    rgb = rgb.to(DEVICE).float() / 255.0
    ir = ir.to(DEVICE).float() / 255.0
    targets = {k: v.to(DEVICE) for k, v in targets.items()}

    model.zero_grad(set_to_none=True)
    preds = model(rgb, ir)
    loss, loss_items = model.loss(targets, preds)
    total_loss = loss.sum()

    print(f"\n总损失：{total_loss.item():.8f}")
    print(f"损失分量：{loss.detach().cpu().tolist()}")
    total_loss.backward()

    report_gradients("RGB Backbone", model.rgb_backbone)
    report_gradients("IR Backbone", model.ir_backbone)
    report_gradients("Fusion Layers", model.fusions)
    report_gradients("YOLO Neck + Detect Head", model.detector.model[10:])

    print("\n梯度诊断完成。请把从 RGB Backbone 到 YOLO Neck 的完整输出发回来。")


if __name__ == "__main__":
    main()
