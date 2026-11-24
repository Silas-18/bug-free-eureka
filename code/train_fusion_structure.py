import argparse
import csv
import random
import time
from contextlib import nullcontext
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 允许从 code 文件夹直接导入同目录模块
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from fusion_model_structure import DualModalYOLOv8
from fusion_dataset import LLVIPFusionDataset, fusion_collate_fn


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLVIP RGB-IR 双模态 YOLOv8 融合训练"
    )
    parser.add_argument("--data-root", type=str,
                        default=str(PROJECT_ROOT / "dataset"))
    parser.add_argument("--rgb-weights", type=str,
                        default=str(PROJECT_ROOT / "experiments/baseline/v8_rgb_baseline/weights/best.pt"))
    parser.add_argument("--ir-weights", type=str,
                        default=str(PROJECT_ROOT / "experiments/baseline/v8_ir_baseline/weights/best.pt"))
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "experiments/fusion/stage1_fusion_neck"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--amp", action="store_true",
                        help="启用 CUDA 混合精度；默认关闭，先确保基础训练稳定")
    parser.add_argument("--init-weights", type=str, default="",
                        help="仅从已有融合 checkpoint 加载模型参数，不恢复优化器")
    parser.add_argument("--resume", type=str, default="",
                        help="可选：从本脚本保存的 last.pt 恢复训练")
    parser.add_argument("--max-train-steps", type=int, default=0,
                        help="调试用：每个 epoch 最多训练多少个 batch；0 表示不限制")
    parser.add_argument("--max-val-steps", type=int, default=0,
                        help="调试用：每个 epoch 最多验证多少个 batch；0 表示不限制")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-geo-aug", action="store_true",
                        help="消融：关闭 RGB-IR 同步几何增强")
    parser.add_argument("--no-color-aug", action="store_true",
                        help="消融：关闭 RGB 颜色/亮度/对比度增强")
    parser.add_argument('--fusion-mode', choices=['concat', 'avg'], default='concat', help='concat: original CBR; avg: elementwise mean')
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_targets_to_device(targets, device):
    return {key: value.to(device, non_blocking=True)
            for key, value in targets.items()}


def autocast_context(use_amp):
    if use_amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def get_grad_norm(parameter):
    if parameter is None or parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().norm().item())



def get_module_grad_norm(module):
    """计算整个模块所有参数梯度的全局 L2 norm。"""
    squared_sum = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared_sum += float(
                parameter.grad.detach().float().pow(2).sum().item()
            )
    return squared_sum ** 0.5


def save_checkpoint(path, model, optimizer, scheduler, scaler,
                    epoch, best_val_loss, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_val_loss": best_val_loss,
        "config": vars(args),
    }
    torch.save(checkpoint, path)


def validate_loss(model, loader, device, use_amp, max_steps=0):
    """
    计算验证集检测 loss。
    这里让检测头保持训练输出格式，同时冻结 BN 的运行统计；
    本函数不计算 mAP，mAP 需要后续单独评估。
    """
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()

    total_loss = 0.0
    batches = 0

    with torch.no_grad():
        for step, (rgb, ir, targets) in enumerate(loader):
            if max_steps > 0 and step >= max_steps:
                break

            rgb = rgb.to(device, non_blocking=True).float().div_(255.0)
            ir = ir.to(device, non_blocking=True).float().div_(255.0)
            targets = move_targets_to_device(targets, device)

            with autocast_context(use_amp):
                preds = model(rgb, ir)
                loss, _ = model.loss(targets, preds)
                batch_loss = loss.sum()

            if not torch.isfinite(batch_loss):
                raise FloatingPointError(
                    f"验证阶段出现非有限 loss，batch={step}"
                )

            total_loss += float(batch_loss.item())
            batches += 1

    # 恢复所有模块的训练状态，下一轮训练前会重新进入 train 模式
    model.train()

    if batches == 0:
        raise RuntimeError("验证集没有可用 batch，请检查 val.txt 和数据路径。")

    return total_loss / batches


def update_loss_plot(csv_path, output_dir):
    try:
        import matplotlib.pyplot as plt

        epochs = []
        train_losses = []
        val_losses = []

        with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                epochs.append(int(row["epoch"]))
                train_losses.append(float(row["train_loss"]))
                val_losses.append(float(row["val_loss"]))

        if not epochs:
            return

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("LLVIP RGB-IR Fusion Training")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "loss_curve.png", dpi=160)
        plt.close()
    except Exception as exc:
        print(f"提示：损失曲线绘制失败，但 CSV 已保存。原因：{exc}")


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.epochs < 1:
        raise ValueError("--epochs 必须大于等于 1")
    if args.batch_size < 1 or args.accum_steps < 1:
        raise ValueError("--batch-size 和 --accum-steps 必须大于等于 1")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "指定了 CUDA 设备，但 PyTorch 当前无法使用 CUDA。"
            "请先检查 conda 环境中的 torch 和显卡驱动。"
        )

    use_amp = bool(args.amp and device.type == "cuda")
    output_dir = Path(args.output)
    weights_dir = output_dir / "weights"
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LLVIP RGB-IR 双模态融合训练")
    print("=" * 60)
    print(f"设备：{device}")
    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(device)}")
        print(f"显存：{torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")
    print(f"Epoch：{args.epochs}")
    print(f"Batch size：{args.batch_size}")
    print(f"梯度累积步数：{args.accum_steps}")
    print(f"AMP：{use_amp}")
    print(f"输出目录：{output_dir}")

    for required_path in [args.rgb_weights, args.ir_weights]:
        if not Path(required_path).is_file():
            raise FileNotFoundError(f"找不到预训练权重：{required_path}")

    # 读取训练集和验证集
    train_dataset = LLVIPFusionDataset(
        root=args.data_root, split="train", imgsz=args.imgsz,
        geo_aug=not args.no_geo_aug, color_aug=not args.no_color_aug
    )
    val_dataset = LLVIPFusionDataset(
        root=args.data_root, split="val", imgsz=args.imgsz
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("训练集或验证集为空，请检查数据集列表文件。")

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": fusion_collate_fn,
    }
    if args.workers > 0:
        loader_options["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=False, **loader_options
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, drop_last=False, **loader_options
    )

    print(f"训练集配对数：{len(train_dataset)}")
    print(f"验证集配对数：{len(val_dataset)}")

    # 初始化双模态模型
    model = DualModalYOLOv8(rgb_weights=args.rgb_weights, ir_weights=args.ir_weights, fusion_mode=args.fusion_mode).to(device)

    # 从上一轮融合 best.pt 初始化模型，只加载模型参数
    if args.init_weights:
        init_path = Path(args.init_weights)

        if not init_path.is_file():
            raise FileNotFoundError(
                f"找不到融合初始化权重：{init_path}"
            )

        init_checkpoint = torch.load(
            init_path,
            map_location="cpu",
            weights_only=False
        )

        if "model" not in init_checkpoint:
            raise KeyError("融合 checkpoint 中不存在 model 字段")

        model.load_state_dict(
            init_checkpoint["model"],
            strict=True
        )

        print("\\n========== 融合模型初始化 ==========")
        print(f"初始化权重：{init_path}")
        print(f"来源 Epoch：{init_checkpoint['epoch'] + 1}")
        print(f"来源 Best Val Loss：{init_checkpoint['best_val_loss']:.6f}")
        print("模型参数加载成功，优化器将重新初始化。")

    # 第一阶段训练策略：
    # 1. 冻结 RGB / IR Backbone，保留单模态预训练特征；
    # 2. 训练三层融合模块；
    # 3. 解冻 RGB YOLO Neck + Detect Head，让检测器适应融合后的特征。
    # 这比“只训练融合层”更合理，也比一开始就训练两个完整 Backbone
    # 更节省 6GB 显存。后续稳定后再考虑第二阶段解冻 Backbone 后段。
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for parameter in model.fusions.parameters():
        parameter.requires_grad_(True)

    for parameter in model.detector.model[10:].parameters():
        parameter.requires_grad_(True)

    trainable_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    trainable_count = sum(p.numel() for p in trainable_parameters)
    total_count = sum(p.numel() for p in model.parameters())

    print("\\n========== 参数训练状态 ==========")
    print("RGB Backbone：冻结")
    print("IR Backbone：冻结")
    print("Fusion Layers：训练")
    print("YOLO Neck + Detect Head：训练")
    print(
        f"可训练参数：{trainable_count:,} / "
        f"{total_count:,} ({trainable_count / total_count:.2%})"
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.lr * 0.01,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"找不到恢复训练文件：{resume_path}")
        checkpoint = torch.load(
            resume_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        print(f"已从 epoch {start_epoch} 继续训练。")

    csv_path = output_dir / "metrics.csv"
    if not csv_path.exists() or start_epoch == 0:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "val_loss", "learning_rate",
                "epoch_seconds"
            ])

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        model.train()

         # 冻结两个 Backbone 的 BatchNorm 运行状态
        model.rgb_backbone.eval()
        model.ir_backbone.eval()

        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        train_batches = 0
        optimizer_updates = 0
        first_grad_reported = False

        effective_train_steps = len(train_loader)
        if args.max_train_steps > 0:
            effective_train_steps = min(
                effective_train_steps, args.max_train_steps
            )

        print(f"\n========== Epoch {epoch + 1}/{args.epochs} ==========")

        for step, (rgb, ir, targets) in enumerate(train_loader):
            if step >= effective_train_steps:
                break

            rgb = rgb.to(device, non_blocking=True).float().div_(255.0)
            ir = ir.to(device, non_blocking=True).float().div_(255.0)
            targets = move_targets_to_device(targets, device)

            # 处理最后一个不完整的梯度累积组
            group_start = (step // args.accum_steps) * args.accum_steps
            group_size = min(
                args.accum_steps, effective_train_steps - group_start
            )

            with autocast_context(use_amp):
                preds = model(rgb, ir)
                loss, _ = model.loss(targets, preds)
                total_loss = loss.sum()

            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    f"训练阶段出现非有限 loss，epoch={epoch + 1}, batch={step + 1}"
                )

            scaled_loss = total_loss / group_size
            scaler.scale(scaled_loss).backward()

            should_update = (
                (step + 1) % args.accum_steps == 0
                or step + 1 == effective_train_steps
            )

            if should_update:
                scaler.unscale_(optimizer)

                if not first_grad_reported:
                    print(
                        "梯度检查（模块全局 L2 norm）| "
                        f"RGB Backbone（冻结，预期 0）: "
                        f"{get_module_grad_norm(model.rgb_backbone):.6e} | "
                        f"IR Backbone（冻结，预期 0）: "
                        f"{get_module_grad_norm(model.ir_backbone):.6e} | "
                        f"Fusion: {get_module_grad_norm(model.fusions):.6e} | "
                        f"YOLO Neck+Head: "
                        f"{get_module_grad_norm(model.detector.model[10:]):.6e}"
                    )
                    first_grad_reported = True

                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, max_norm=10.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_updates += 1

            running_loss += float(total_loss.detach().item())
            train_batches += 1

            if (step + 1) % 100 == 0 or step + 1 == effective_train_steps:
                average_so_far = running_loss / train_batches
                print(
                    f"Epoch {epoch + 1} | "
                    f"Batch {step + 1}/{effective_train_steps} | "
                    f"Loss {average_so_far:.4f}"
                )

        if train_batches == 0:
            raise RuntimeError("本轮没有执行任何训练 batch。")

        train_loss = running_loss / train_batches

        # 验证阶段只计算 validation loss，不计算 mAP
        val_loss = validate_loss(
            model=model,
            loader=val_loader,
            device=device,
            use_amp=use_amp,
            max_steps=args.max_val_steps,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        epoch_seconds = time.time() - epoch_start

        print(
            f"Epoch {epoch + 1} 完成 | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Optimizer Updates: {optimizer_updates} | "
            f"耗时: {epoch_seconds / 60:.1f} 分钟"
        )

        with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                f"{train_loss:.6f}",
                f"{val_loss:.6f}",
                f"{current_lr:.8g}",
                f"{epoch_seconds:.2f}",
            ])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                weights_dir / "best.pt",
                model, optimizer, scheduler, scaler,
                epoch, best_val_loss, args,
            )
            print(f"验证集 loss 刷新，已保存 best.pt（{best_val_loss:.4f}）")

        save_checkpoint(
            weights_dir / "last.pt",
            model, optimizer, scheduler, scaler,
            epoch, best_val_loss, args,
        )

        update_loss_plot(csv_path, output_dir)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n训练流程结束。")
    print(f"Last 权重：{weights_dir / 'last.pt'}")
    print(f"Best 权重：{weights_dir / 'best.pt'}")
    print(f"训练记录：{csv_path}")
    print(f"损失曲线：{output_dir / 'loss_curve.png'}")
    print("注意：当前脚本的验证指标是 loss，不是 mAP。")


if __name__ == "__main__":
    main()
