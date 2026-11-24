#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLVIP RGB-IR 双模态 YOLOv8 融合模型测试脚本。

放在 LLVIP_Project/code/eval_fusion.py 下运行。
指标：Precision/Recall（指定置信度、IoU=0.5）、mAP@0.5、mAP@0.5:0.95。
AP 使用 101-point interpolation；所有待比较模型必须使用相同评估口径。
"""
import argparse
import csv
import json
import time
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader
from ultralytics.utils.ops import non_max_suppression

from fusion_model import DualModalYOLOv8
from fusion_dataset import LLVIPFusionDataset, fusion_collate_fn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10, dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser("LLVIP RGB-IR fusion evaluation")
    p.add_argument("--data-root", default=str(PROJECT_ROOT / "dataset"))
    p.add_argument("--rgb-weights", default=str(PROJECT_ROOT / "experiments/baseline/v8_rgb_baseline/weights/best.pt"))
    p.add_argument("--ir-weights", default=str(PROJECT_ROOT / "experiments/baseline/v8_ir_baseline/weights/best.pt"))
    p.add_argument("--fusion-weights", default=str(PROJECT_ROOT / "experiments/fusion/stage2_bs4_fp32/weights/best.pt"))
    p.add_argument("--output", default=str(PROJECT_ROOT / "experiments/fusion/stage2_bs4_fp32/evaluation_test"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--nms-conf", type=float, default=0.001, help="NMS候选置信度阈值；低阈值用于保留AP计算所需预测")
    p.add_argument("--nms-iou", type=float, default=0.70)
    p.add_argument("--report-conf", type=float, default=0.25, help="Precision/Recall汇报使用的置信度阈值")
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--amp", action="store_true", help="启用CUDA AMP推理；默认FP32")
    p.add_argument("--max-batches", type=int, default=0, help="调试用；0表示完整测试集")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_prediction_tensor(output, batch_size):
    """从模型输出（Tensor或嵌套tuple/list）中提取[B,4+nc,N]解码预测。"""
    candidates = []

    def visit(obj):
        if torch.is_tensor(obj):
            if obj.ndim == 3 and obj.shape[0] == batch_size:
                if 5 <= obj.shape[1] <= 256 and obj.shape[2] > obj.shape[1]:
                    candidates.append((obj, False))       # [B,C,N]
                elif 5 <= obj.shape[2] <= 256 and obj.shape[1] > obj.shape[2]:
                    candidates.append((obj, True))        # [B,N,C]
        elif isinstance(obj, (tuple, list)):
            for x in obj:
                visit(x)
        elif isinstance(obj, dict):
            for x in obj.values():
                visit(x)

    visit(output)
    if not candidates:
        shape = tuple(output.shape) if torch.is_tensor(output) else type(output).__name__
        raise RuntimeError(
            f"无法从模型输出中识别[B,4+nc,N]预测张量，输出为：{shape}。"
            "请检查 DualModalYOLOv8.eval() 下的 forward 输出。"
        )
    pred, transpose = max(
        candidates,
        key=lambda z: z[0].shape[1] if z[1] else z[0].shape[2],
    )
    if transpose:
        pred = pred.transpose(1, 2).contiguous()
    if pred.shape[1] < 5:
        raise RuntimeError(f"预测通道数不足：{tuple(pred.shape)}")
    return pred


def xywhn_to_xyxy_pixels(boxes, imgsz):
    """数据集标签已经过letterbox；这里只将归一化xywh转为输入图像像素xyxy。"""
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    b = boxes.float() * float(imgsz)
    x, y, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return torch.stack((x-w/2, y-h/2, x+w/2, y+h/2), dim=1)


def box_iou_numpy(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(rb - lt, 0.0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.maximum(a[:, 2]-a[:, 0], 0) * np.maximum(a[:, 3]-a[:, 1], 0)
    area_b = np.maximum(b[:, 2]-b[:, 0], 0) * np.maximum(b[:, 3]-b[:, 1], 0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def match_predictions(pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes):
    """按置信度优先进行一对一匹配，返回[N预测框,10个IoU阈值] TP矩阵。"""
    tp = np.zeros((len(pred_boxes), len(IOU_THRESHOLDS)), dtype=bool)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return tp
    ious = box_iou_numpy(pred_boxes, gt_boxes)
    ious *= (pred_classes[:, None] == gt_classes[None, :]).astype(np.float32)
    pred_order = np.argsort(-pred_scores)
    for j, threshold in enumerate(IOU_THRESHOLDS):
        pi, gi = np.where(ious >= threshold)
        if len(pi) == 0:
            continue
        matches = np.stack((pi, gi, ious[pi, gi]), axis=1)
        matches = matches[np.argsort(-matches[:, 2])]
        _, keep = np.unique(matches[:, 0], return_index=True)
        matches = matches[np.sort(keep)]
        _, keep = np.unique(matches[:, 1], return_index=True)
        matches = matches[np.sort(keep)]
        tp[matches[:, 0].astype(np.int64), j] = True
    # 调用者后续按全局置信度排序；这里排序不改变tp与预测框的对应关系。
    return tp


def compute_ap_101(recall, precision):
    if len(recall) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    x = np.linspace(0.0, 1.0, 101)
    return float(np.trapz(np.interp(x, mrec, mpre), x))


def calculate_metrics(tp, conf, pred_cls, target_cls, report_conf):
    if len(target_cls) == 0:
        raise RuntimeError("测试集中没有真实标注框，请检查 labels 路径和标签内容。")
    classes = np.unique(target_cls)
    class_rows = []
    ap_all, p_all, r_all = [], [], []
    for class_id in classes:
        n_gt = int(np.sum(target_cls == class_id))
        mask = pred_cls == class_id
        c_conf, c_tp = conf[mask], tp[mask]
        if len(c_conf):
            order = np.argsort(-c_conf)
            c_conf, c_tp = c_conf[order], c_tp[order]
            cum_tp = np.cumsum(c_tp, axis=0)
            cum_fp = np.cumsum(~c_tp, axis=0)
            rec_curve = cum_tp / max(n_gt, 1)
            pre_curve = cum_tp / np.maximum(cum_tp + cum_fp, 1)
            ap = np.array([
                compute_ap_101(rec_curve[:, j], pre_curve[:, j])
                for j in range(len(IOU_THRESHOLDS))
            ], dtype=np.float32)
        else:
            ap = np.zeros(len(IOU_THRESHOLDS), dtype=np.float32)

        selected = c_conf >= report_conf
        selected_tp = c_tp[selected] if len(c_conf) else np.zeros((0, len(IOU_THRESHOLDS)), dtype=bool)
        tp50 = int(selected_tp[:, 0].sum()) if len(selected_tp) else 0
        fp50 = int(len(selected_tp) - tp50)
        fn50 = max(n_gt - tp50, 0)
        precision = tp50 / max(tp50 + fp50, 1)
        recall = tp50 / max(tp50 + fn50, 1)

        class_rows.append({
            "class_id": int(class_id),
            "num_ground_truth": n_gt,
            "num_predictions_at_report_conf": int(selected.sum()),
            "precision": float(precision),
            "recall": float(recall),
            "ap50": float(ap[0]),
            "ap50_95": float(ap.mean()),
        })
        ap_all.append(ap)
        p_all.append(precision)
        r_all.append(recall)

    ap_all = np.asarray(ap_all, dtype=np.float32)
    return {
        "precision": float(np.mean(p_all)),
        "recall": float(np.mean(r_all)),
        "map50": float(np.mean(ap_all[:, 0])),
        "map50_95": float(np.mean(ap_all)),
        "classes": class_rows,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.batch_size < 1:
        raise ValueError("--batch-size 必须大于等于1")
    if not 0 <= args.nms_conf < 1 or not 0 < args.nms_iou < 1 or not 0 <= args.report_conf < 1:
        raise ValueError("请检查 nms-conf、nms-iou、report-conf 的取值范围")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查 AutoDL GPU 和 PyTorch 环境。")
    use_amp = bool(args.amp and device.type == "cuda")

    data_root = Path(args.data_root)
    rgb_weights, ir_weights = Path(args.rgb_weights), Path(args.ir_weights)
    fusion_weights, output_dir = Path(args.fusion_weights), Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for f in (rgb_weights, ir_weights, fusion_weights):
        if not f.is_file():
            raise FileNotFoundError(f"权重文件不存在：{f}")

    print("=" * 68)
    print("LLVIP RGB-IR 双模态融合模型测试")
    print(f"设备：{device}")
    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(device)}")
        print(f"显存：{torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")
    print(f"融合权重：{fusion_weights}")
    print(f"Batch Size：{args.batch_size} | imgsz：{args.imgsz} | AMP：{use_amp}")
    print(f"NMS conf：{args.nms_conf} | NMS IoU：{args.nms_iou} | P/R conf：{args.report_conf}")

    # split=test：按当前 fusion_dataset.py 逻辑不会启用训练增强。
    dataset = LLVIPFusionDataset(root=str(data_root), split="test", imgsz=args.imgsz)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=fusion_collate_fn,
    )
    if args.workers > 0:
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kwargs)
    print(f"测试集配对数量：{len(dataset)}")

    print("\n========== 初始化双模态模型 ==========")
    model = DualModalYOLOv8(rgb_weights=str(rgb_weights), ir_weights=str(ir_weights))
    checkpoint = torch.load(str(fusion_weights), map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        source_epoch = checkpoint.get("epoch")
        source_best_loss = checkpoint.get("best_val_loss")
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict, source_epoch, source_best_loss = checkpoint["state_dict"], None, None
    else:
        raise RuntimeError("无法识别 checkpoint：预期包含 model 或 state_dict。")
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    print("融合权重严格匹配加载成功！")
    if source_epoch is not None:
        print(f"权重来源 Epoch：{int(source_epoch) + 1}")
    if source_best_loss is not None:
        print(f"来源 Best Val Loss：{float(source_best_loss):.6f}")

    stat_tp, stat_conf, stat_pred_cls, stat_target_cls = [], [], [], []
    total_images = total_preds = total_gt = 0
    start = time.time()
    print("\n========== 开始测试集推理 ==========")

    with torch.inference_mode():
        for batch_i, (rgb, ir, targets) in enumerate(loader):
            if args.max_batches > 0 and batch_i >= args.max_batches:
                break
            rgb = rgb.to(device, non_blocking=True).float().div_(255.0)
            ir = ir.to(device, non_blocking=True).float().div_(255.0)
            bs = rgb.shape[0]
            amp_ctx = torch.cuda.amp.autocast(enabled=True) if use_amp else nullcontext()
            with amp_ctx:
                output = model(rgb, ir)
            pred = extract_prediction_tensor(output, bs)
            detections = non_max_suppression(
                pred.float(),
                conf_thres=args.nms_conf,
                iou_thres=args.nms_iou,
                max_det=args.max_det,
            )

            batch_idx = targets["batch_idx"].long()
            boxes_all = targets["bboxes"]
            cls_all = targets["cls"].view(-1)
            for image_i in range(bs):
                gt_mask = batch_idx == image_i
                gt_boxes = xywhn_to_xyxy_pixels(boxes_all[gt_mask].to(device), args.imgsz).cpu().numpy()
                gt_cls = cls_all[gt_mask].to(device).cpu().numpy().astype(np.int64)
                det = detections[image_i]
                if det is None or len(det) == 0:
                    p_boxes = np.zeros((0, 4), dtype=np.float32)
                    p_scores = np.zeros((0,), dtype=np.float32)
                    p_cls = np.zeros((0,), dtype=np.int64)
                else:
                    det = det.cpu().numpy()
                    p_boxes = det[:, :4].astype(np.float32)
                    p_scores = det[:, 4].astype(np.float32)
                    p_cls = det[:, 5].astype(np.int64)
                tp = match_predictions(p_boxes, p_scores, p_cls, gt_boxes, gt_cls)
                stat_tp.append(tp)
                stat_conf.append(p_scores)
                stat_pred_cls.append(p_cls)
                stat_target_cls.append(gt_cls)
                total_images += 1
                total_preds += len(p_boxes)
                total_gt += len(gt_boxes)

            if (batch_i + 1) % 50 == 0 or batch_i + 1 == len(loader):
                done = min((batch_i + 1) * args.batch_size, len(dataset))
                print(f"Batch {batch_i+1}/{len(loader)} | Images {done}/{len(dataset)} | Predictions {total_preds} | {(time.time()-start)/60:.1f} min")

    if total_images == 0:
        raise RuntimeError("没有完成任何测试图像推理。")

    tp_all = np.concatenate(stat_tp, axis=0) if stat_tp else np.zeros((0, 10), dtype=bool)
    conf_all = np.concatenate(stat_conf) if stat_conf else np.zeros((0,), dtype=np.float32)
    pred_cls_all = np.concatenate(stat_pred_cls) if stat_pred_cls else np.zeros((0,), dtype=np.int64)
    target_cls_all = np.concatenate(stat_target_cls) if stat_target_cls else np.zeros((0,), dtype=np.int64)
    metrics = calculate_metrics(tp_all, conf_all, pred_cls_all, target_cls_all, args.report_conf)
    elapsed = time.time() - start

    result = {
        "model": "DualModalYOLOv8",
        "fusion_weights": str(fusion_weights),
        "split": "test",
        "images_evaluated": total_images,
        "dataset_pairs": len(dataset),
        "ground_truth_boxes": total_gt,
        "predictions_after_nms": total_preds,
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "nms_conf": args.nms_conf,
        "nms_iou": args.nms_iou,
        "report_conf": args.report_conf,
        "iou_thresholds": [float(x) for x in IOU_THRESHOLDS],
        "precision_at_report_conf_iou50": metrics["precision"],
        "recall_at_report_conf_iou50": metrics["recall"],
        "map50": metrics["map50"],
        "map50_95": metrics["map50_95"],
        "elapsed_seconds": elapsed,
        "partial_evaluation": bool(args.max_batches > 0),
        "ap_method": "101-point interpolated AP",
        "classes": metrics["classes"],
    }

    json_path = output_dir / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    csv_path = output_dir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["images", "GT_boxes", "predictions", "report_conf", "Precision", "Recall", "mAP50", "mAP50_95", "elapsed_seconds", "partial"])
        w.writerow([total_images, total_gt, total_preds, args.report_conf, f"{metrics['precision']:.6f}", f"{metrics['recall']:.6f}", f"{metrics['map50']:.6f}", f"{metrics['map50_95']:.6f}", f"{elapsed:.2f}", args.max_batches > 0])

    class_csv = output_dir / "per_class_metrics.csv"
    with open(class_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "GT_boxes", "predictions_at_report_conf", "Precision", "Recall", "AP50", "AP50_95"])
        for row in metrics["classes"]:
            w.writerow([row["class_id"], row["num_ground_truth"], row["num_predictions_at_report_conf"], f"{row['precision']:.6f}", f"{row['recall']:.6f}", f"{row['ap50']:.6f}", f"{row['ap50_95']:.6f}"])

    print("\n" + "=" * 68)
    print("测试完成")
    print(f"测试图像：{total_images} | GT框：{total_gt} | NMS预测框：{total_preds}")
    print(f"Precision (conf={args.report_conf}, IoU=0.5)：{metrics['precision']:.4f}")
    print(f"Recall    (conf={args.report_conf}, IoU=0.5)：{metrics['recall']:.4f}")
    print(f"mAP@0.5：{metrics['map50']:.4f}")
    print(f"mAP@0.5:0.95：{metrics['map50_95']:.4f}")
    print(f"耗时：{elapsed/60:.2f} 分钟")
    print(f"JSON：{json_path}")
    print(f"CSV：{csv_path}")
    print(f"逐类别结果：{class_csv}")
    if args.max_batches > 0:
        print("警告：当前是部分 batch 调试结果，不能作为正式测试指标。")
    else:
        print("已完成完整测试集评估。")


if __name__ == "__main__":
    main()
