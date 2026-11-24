#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLVIP single-modal RGB/IR baselines, evaluated with eval_fusion.py metrics.

Place beside eval_fusion.py and fusion_dataset.py in LLVIP_Project/code/.
Requires the SAME eval_fusion.py used for the fusion evaluations.

By default the LLVIP paired test loader uses RGB annotation boxes for both
modalities, identical to the current fusion evaluation. Inspect RGB/IR labels
alignment before drawing conclusions about IR-native annotation results.

The P/R definition matches eval_fusion.py: confidence >= report_conf, IoU=.5.
AP uses the same custom 101-point interpolation and matching policy; it may
not be numerically identical to Ultralytics' built-in validation metrics.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression

from fusion_dataset import LLVIPFusionDataset, fusion_collate_fn
from eval_fusion import (
    IOU_THRESHOLDS,
    calculate_metrics,
    extract_prediction_tensor,
    match_predictions,
    xywhn_to_xyxy_pixels,
)

ROOT = Path(__file__).resolve().parents[1]


def cli():
    p = argparse.ArgumentParser(description="LLVIP RGB + IR 单模态统一评估")
    p.add_argument("--data-root", default=str(ROOT / "dataset"))
    p.add_argument("--rgb-weights", default=str(ROOT / "experiments/baseline/v8_rgb_baseline/weights/best.pt"))
    p.add_argument("--ir-weights", default=str(ROOT / "experiments/baseline/v8_ir_baseline/weights/best.pt"))
    p.add_argument("--output", default=str(ROOT / "experiments/comparison/llvip_unified"))
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--nms-conf", type=float, default=0.001)
    p.add_argument("--nms-iou", type=float, default=0.70)
    p.add_argument("--report-conf", type=float, default=0.25)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--max-batches", type=int, default=0,
                   help="0=全部1549张；大于0仅作 smoke test")
    p.add_argument("--mode", choices=("both", "rgb", "ir"), default="both")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def validate_args(args):
    if args.batch_size < 1 or args.imgsz < 32 or args.workers < 0:
        raise ValueError("batch-size、imgsz 必须为正数，workers 不能为负数")
    if not 0 <= args.nms_conf < 1 or not 0 < args.nms_iou < 1:
        raise ValueError("NMS 参数超出合法范围")
    if not 0 <= args.report_conf < 1:
        raise ValueError("report-conf 必须在 [0,1) 范围内")
    if args.max_batches < 0:
        raise ValueError("max-batches 不能为负数")


def finish_metrics(state, args, model_name, weight_path, dataset_size):
    if not state["images"]:
        raise RuntimeError(f"{model_name} 无可用测试图像")
    tp = np.concatenate(state["tp"], axis=0)
    conf = np.concatenate(state["conf"])
    pred_cls = np.concatenate(state["pred_cls"])
    target_cls = np.concatenate(state["target_cls"])
    metrics = calculate_metrics(tp, conf, pred_cls, target_cls, args.report_conf)
    return {
        "model": model_name,
        "weights": str(weight_path),
        "split": "test",
        "images_evaluated": state["images"],
        "dataset_pairs": dataset_size,
        "ground_truth_boxes": state["gt"],
        "predictions_after_nms": state["predictions"],
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "nms_conf": args.nms_conf,
        "nms_iou": args.nms_iou,
        "report_conf": args.report_conf,
        "iou_thresholds": [float(t) for t in IOU_THRESHOLDS],
        "precision_at_report_conf_iou50": metrics["precision"],
        "recall_at_report_conf_iou50": metrics["recall"],
        "map50": metrics["map50"],
        "map50_95": metrics["map50_95"],
        "ap_method": "101-point interpolated AP",
        "partial_evaluation": args.max_batches > 0,
        "classes": metrics["classes"],
    }


def save_result(path, result):
    path.mkdir(parents=True, exist_ok=True)
    with (path / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with (path / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["model", "images", "GT_boxes", "predictions", "report_conf",
                    "Precision", "Recall", "mAP50", "mAP50_95", "partial"])
        w.writerow([
            result["model"], result["images_evaluated"],
            result["ground_truth_boxes"], result["predictions_after_nms"],
            result["report_conf"],
            f"{result['precision_at_report_conf_iou50']:.6f}",
            f"{result['recall_at_report_conf_iou50']:.6f}",
            f"{result['map50']:.6f}", f"{result['map50_95']:.6f}",
            result["partial_evaluation"],
        ])
    with (path / "per_class_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "GT_boxes", "predictions_at_report_conf",
                    "Precision", "Recall", "AP50", "AP50_95"])
        for row in result["classes"]:
            w.writerow([
                row["class_id"], row["num_ground_truth"],
                row["num_predictions_at_report_conf"],
                f"{row['precision']:.6f}", f"{row['recall']:.6f}",
                f"{row['ap50']:.6f}", f"{row['ap50_95']:.6f}",
            ])


def fusion_results_if_compatible(args, baseline_result):
    """Read finished fusion results only when settings/GT counts really match."""
    entries = [
        ("Fusion Stage1", ROOT / "experiments/fusion/stage1_full_check/evaluation_test/metrics.json"),
        ("Fusion Stage2", ROOT / "experiments/fusion/stage2_bs4_fp32/evaluation_test/metrics.json"),
    ]
    loaded = []
    for name, path in entries:
        if not path.is_file():
            print(f"提示：缺少 {path}，四模型汇总暂不包含 {name}")
            continue
        with path.open("r", encoding="utf-8") as f:
            value = json.load(f)
        keys = ("imgsz", "nms_conf", "nms_iou", "report_conf")
        if any(abs(float(value.get(k, -999)) - float(baseline_result[k])) > 1e-7 for k in keys):
            print(f"警告：{name} 的评估参数不一致，跳过四模型汇总")
            continue
        if (value.get("partial_evaluation") or
            value.get("images_evaluated") != baseline_result["images_evaluated"] or
            value.get("ground_truth_boxes") != baseline_result["ground_truth_boxes"] or
            value.get("ap_method") != baseline_result["ap_method"]):
            print(f"警告：{name} 的样本/GT/AP口径不一致，跳过四模型汇总")
            continue
        loaded.append((name, value))
    return loaded


def comparison_csv(out_dir, baseline_results, args):
    if args.max_batches > 0:
        print("调试模式：不生成正式模型对比表。")
        return
    entries = [(name, baseline_results[name]) for name in ("RGB", "IR") if name in baseline_results]
    if not entries:
        return
    entries.extend(fusion_results_if_compatible(args, entries[0][1]))
    path = out_dir / "comparison_all_models.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["model", "images", "GT_boxes", "Precision", "Recall", "mAP50", "mAP50_95"])
        for name, result in entries:
            w.writerow([
                name, result["images_evaluated"], result["ground_truth_boxes"],
                f"{result['precision_at_report_conf_iou50']:.6f}",
                f"{result['recall_at_report_conf_iou50']:.6f}",
                f"{result['map50']:.6f}", f"{result['map50_95']:.6f}",
            ])
    print(f"\n模型对比表：{path}")
    if len(entries) < 4:
        print("提示：目前不是完整四模型对比，请查看上方缺失/配置警告。")


def main():
    args = cli()
    validate_args(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查 AutoDL GPU 是否已启动")

    selected = ("rgb", "ir") if args.mode == "both" else (args.mode,)
    paths = {
        "rgb": Path(args.rgb_weights),
        "ir": Path(args.ir_weights),
    }
    for mode in selected:
        if not paths[mode].is_file():
            raise FileNotFoundError(f"{mode} 权重不存在：{paths[mode]}")

    print("=" * 70)
    print("LLVIP RGB / IR 单模态统一测试（与 eval_fusion 共用评估函数）")
    print("=" * 70)
    print(f"设备：{device}；Batch Size：{args.batch_size}；imgsz：{args.imgsz}")
    print(f"NMS conf={args.nms_conf}, IoU={args.nms_iou}, P/R conf={args.report_conf}")
    print("标注：沿用 fusion_dataset 的 RGB 标注，确保与融合模型共用同一 GT")
    print("增强：关闭（split=test）；AMP：关闭（FP32）")
    if args.max_batches:
        print(f"警告：当前仅调试 {args.max_batches} 个 Batch，结果不可用于论文。")

    dataset = LLVIPFusionDataset(root=args.data_root, split="test", imgsz=args.imgsz)
    loader_opts = dict(
        batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        collate_fn=fusion_collate_fn,
    )
    if args.workers > 0:
        loader_opts["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_opts)

    models = {}
    for mode in selected:
        model = YOLO(str(paths[mode]))
        if model.task != "detect" or len(model.names) != 1:
            raise RuntimeError(f"{mode} 应是单类别 detect 模型，实际 {model.task}, {model.names}")
        models[mode] = model.model.to(device).eval()
        print(f"{mode.upper()} 权重加载成功：{paths[mode]}")

    stats = {
        mode: {"tp": [], "conf": [], "pred_cls": [], "target_cls": [],
               "images": 0, "gt": 0, "predictions": 0}
        for mode in selected
    }
    start = time.time()

    with torch.inference_mode():
        for batch_i, (rgb, ir, targets) in enumerate(loader):
            if args.max_batches and batch_i >= args.max_batches:
                break
            batch = {
                "rgb": rgb.to(device, non_blocking=True).float().div_(255),
                "ir": ir.to(device, non_blocking=True).float().div_(255),
            }
            bs = batch["rgb"].shape[0]
            gt_masks = targets["batch_idx"].long()
            gt_boxes_all = targets["bboxes"]
            gt_classes_all = targets["cls"].view(-1)

            # The same GT/letterbox coordinates as eval_fusion.py.
            gt_batch = []
            for image_i in range(bs):
                mask = gt_masks == image_i
                gt_boxes = xywhn_to_xyxy_pixels(
                    gt_boxes_all[mask], args.imgsz
                ).numpy().astype(np.float32)
                gt_cls = gt_classes_all[mask].numpy().astype(np.int64)
                gt_batch.append((gt_boxes, gt_cls))

            for mode in selected:
                raw = models[mode](batch[mode])
                prediction = extract_prediction_tensor(raw, bs)
                dets = non_max_suppression(
                    prediction.float(),
                    conf_thres=args.nms_conf,
                    iou_thres=args.nms_iou,
                    max_det=args.max_det,
                )
                state = stats[mode]
                for image_i, det in enumerate(dets):
                    gt_boxes, gt_cls = gt_batch[image_i]
                    if det is None or len(det) == 0:
                        pred_boxes = np.zeros((0, 4), np.float32)
                        pred_scores = np.zeros(0, np.float32)
                        pred_classes = np.zeros(0, np.int64)
                    else:
                        a = det.detach().cpu().numpy()
                        pred_boxes = a[:, :4].astype(np.float32)
                        pred_scores = a[:, 4].astype(np.float32)
                        pred_classes = a[:, 5].astype(np.int64)
                    tp = match_predictions(
                        pred_boxes, pred_scores, pred_classes, gt_boxes, gt_cls
                    )
                    state["tp"].append(tp)
                    state["conf"].append(pred_scores)
                    state["pred_cls"].append(pred_classes)
                    state["target_cls"].append(gt_cls)
                    state["images"] += 1
                    state["gt"] += len(gt_boxes)
                    state["predictions"] += len(pred_boxes)

            if (batch_i + 1) % 50 == 0 or batch_i + 1 == len(loader) or (args.max_batches and batch_i+1 == args.max_batches):
                print(f"Batch {batch_i+1}/{len(loader)} | images={stats[selected[0]]['images']}/{len(dataset)} | {((time.time()-start)/60):.2f} min")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for mode in selected:
        result = finish_metrics(stats[mode], args, mode.upper(), paths[mode], len(dataset))
        result["elapsed_seconds_total"] = round(time.time() - start, 2)
        save_result(out_dir / mode, result)
        results[mode.upper()] = result
        print(f"\n[{mode.upper()}] {result['images_evaluated']} images; "
              f"GT={result['ground_truth_boxes']}; "
              f"NMS predictions={result['predictions_after_nms']}")
        print(f"Precision={result['precision_at_report_conf_iou50']:.4f} | "
              f"Recall={result['recall_at_report_conf_iou50']:.4f} | "
              f"mAP50={result['map50']:.4f} | "
              f"mAP50-95={result['map50_95']:.4f}")

    comparison_csv(out_dir, results, args)
    print("\n单模态评估完成。")


if __name__ == "__main__":
    main()
