#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visual comparison of four LLVIP person detectors on identical RGB-IR test pairs.

Put in LLVIP_Project/code/ next to eval_fusion.py, fusion_dataset.py and
fusion_model.py. The checkpoint names, NMS settings and image coordinate frame
match the user's existing unified evaluation workflow.

Each output PNG contains 8 panels (2 x 4):
    RGB input | IR input | GT on RGB | GT on IR
    RGB baseline | IR baseline | Fusion stage 1 | Fusion stage 2

Note: GT in this pipeline comes from RGB labels, for all four detectors.
Do not interpret qualitative examples as complete-test performance estimates.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression

from eval_fusion import extract_prediction_tensor, xywhn_to_xyxy_pixels
from fusion_dataset import LLVIPFusionDataset
from fusion_model import DualModalYOLOv8

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="同一批 LLVIP RGB-IR 测试图像的 RGB、IR、Fusion S1、Fusion S2 检测对比可视化"
    )
    parser.add_argument("--data-root", type=str, default=str(PROJECT_ROOT / "dataset"))
    parser.add_argument("--rgb-weights", type=str, default=str(
        PROJECT_ROOT / "experiments/baseline/v8_rgb_baseline/weights/best.pt"))
    parser.add_argument("--ir-weights", type=str, default=str(
        PROJECT_ROOT / "experiments/baseline/v8_ir_baseline/weights/best.pt"))
    parser.add_argument("--fusion1-weights", type=str, default=str(
        PROJECT_ROOT / "experiments/fusion/stage1_full_check/weights/best.pt"))
    parser.add_argument("--fusion2-weights", type=str, default=str(
        PROJECT_ROOT / "experiments/fusion/stage2_bs4_fp32/weights/best.pt"))
    parser.add_argument("--output", type=str, default=str(
        PROJECT_ROOT / "experiments/comparison/visualizations_stage1_vs_stage2"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--num-samples", type=int, default=6,
                        help="未指定 --indices/--names 时随机抽取的测试图片数，默认6")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--indices", nargs="+", type=int, default=None,
                        help="直接指定测试集索引，0起始，例如 --indices 0 50 100")
    parser.add_argument("--names", nargs="+", default=None,
                        help="按图片文件名（去扩展名）选择，例如 --names 010007 010010")
    parser.add_argument("--nms-conf", type=float, default=0.001,
                        help="NMS 候选置信度门槛，与统一评估一致")
    parser.add_argument("--nms-iou", type=float, default=0.70,
                        help="NMS IoU，与统一评估一致")
    parser.add_argument("--display-conf", type=float, default=0.25,
                        help="画图时显示的置信度门槛，默认0.25")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--max-draw", type=int, default=40,
                        help="每张图每个模型最多画多少框，以避免图片过密")
    return parser.parse_args()


def check_options(args):
    if args.indices is not None and args.names is not None:
        raise ValueError("--indices 和 --names 不能同时指定。")
    if args.num_samples < 1 or args.max_draw < 1 or args.imgsz < 32:
        raise ValueError("num-samples、max-draw 必须 >= 1，imgsz 必须 >= 32。")
    if not (0 <= args.nms_conf < 1 and 0 <= args.display_conf < 1):
        raise ValueError("置信度阈值必须在 [0, 1) 内。")
    if not (0 < args.nms_iou < 1):
        raise ValueError("NMS IoU 必须在 (0, 1) 内。")
    if not (1 <= args.max_det <= 3000):
        raise ValueError("max-det 必须在 [1, 3000] 内。")
    for key in ("rgb_weights", "ir_weights", "fusion1_weights", "fusion2_weights"):
        path = Path(getattr(args, key))
        if not path.is_file():
            raise FileNotFoundError(f"权重文件不存在：{path}")


def sample_indices(dataset, args):
    n = len(dataset)
    if args.indices is not None:
        invalid = [i for i in args.indices if not 0 <= i < n]
        if invalid:
            raise IndexError(f"测试集索引超出 [0, {n - 1}]：{invalid}")
        return list(dict.fromkeys(args.indices))

    if args.names is not None:
        name_to_index = {rgb.stem: idx for idx, (rgb, _) in enumerate(dataset.pairs)}
        missing = [s for s in args.names if Path(s).stem not in name_to_index]
        if missing:
            raise ValueError(
                f"未找到测试集图片 ID：{missing}。可先用 --num-samples 2 随机试运行。"
            )
        return list(dict.fromkeys(name_to_index[Path(s).stem] for s in args.names))

    rng = random.Random(args.seed)
    return sorted(rng.sample(range(n), min(args.num_samples, n)))


def load_fusion(rgb_path, ir_path, fusion_ckpt, device):
    model = DualModalYOLOv8(rgb_weights=str(rgb_path), ir_weights=str(ir_path))
    checkpoint = torch.load(str(fusion_ckpt), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError(f"融合权重不符合当前 checkpoint 格式：{fusion_ckpt}")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    epoch = checkpoint.get("epoch")
    print(f"融合权重已加载：{fusion_ckpt} | epoch={int(epoch)+1 if epoch is not None else 'unknown'}")
    return model


def infer_one(model, rgb_tensor, ir_tensor, mode, args):
    if mode == "rgb":
        raw = model(rgb_tensor)
    elif mode == "ir":
        raw = model(ir_tensor)
    elif mode == "fusion":
        raw = model(rgb_tensor, ir_tensor)
    else:
        raise ValueError(f"未知模型模式：{mode}")

    decoded = extract_prediction_tensor(raw, batch_size=1)
    detections = non_max_suppression(
        decoded.float(),
        conf_thres=args.nms_conf,
        iou_thres=args.nms_iou,
        max_det=args.max_det,
    )[0]
    if detections is None or len(detections) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    return detections.detach().cpu().numpy().astype(np.float32)


def to_bgr(image_tensor):
    rgb_uint8 = image_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)


def draw_detections(img, detections, display_conf, max_draw, color):
    canvas = img.copy()
    kept = detections[detections[:, 4] >= display_conf]
    kept = kept[np.argsort(-kept[:, 4])]
    full_count = len(kept)
    for row in kept[:max_draw]:
        x1, y1, x2, y2, confidence, class_id = row.tolist()
        h, w = canvas.shape[:2]
        x1 = max(0, min(int(round(x1)), w - 1))
        y1 = max(0, min(int(round(y1)), h - 1))
        x2 = max(0, min(int(round(x2)), w - 1))
        y2 = max(0, min(int(round(y2)), h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"person {confidence:.2f}" if int(class_id) == 0 else f"cls{int(class_id)} {confidence:.2f}"
        size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        rect_y1 = max(0, y1 - size[1] - baseline - 6)
        rect_y2 = max(size[1] + baseline + 6, y1)
        cv2.rectangle(canvas, (x1, rect_y1), (min(w - 1, x1 + size[0] + 5), rect_y2), color, -1)
        text_y = max(size[1] + 2, rect_y2 - baseline - 3)
        cv2.putText(canvas, label, (x1 + 2, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas, full_count, min(full_count, max_draw)


def draw_gt(img, boxes):
    canvas = img.copy()
    h, w = canvas.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = box.tolist()
        x1 = max(0, min(int(round(x1)), w-1))
        y1 = max(0, min(int(round(y1)), h-1))
        x2 = max(0, min(int(round(x2)), w-1))
        y2 = max(0, min(int(round(y2)), h-1))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 190, 55), 2)
    return canvas


def tile(image, title, color=(50, 60, 80), header=44):
    h, w = image.shape[:2]
    result = np.full((h + header, w, 3), 247, dtype=np.uint8)
    result[:header] = (245, 245, 245)
    result[header:] = image
    cv2.rectangle(result, (0, 0), (w - 1, h + header - 1), (208, 208, 208), 2)
    cv2.rectangle(result, (0, 0), (8, header - 1), color, -1)
    cv2.putText(result, title, (18, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (25, 35, 45), 2, cv2.LINE_AA)
    return result


def make_comparison(rgb, ir, gt_boxes, outputs, display_conf, max_draw):
    # OpenCV colors are BGR, not RGB.
    rgb_color = (255, 135, 40)
    ir_color = (50, 140, 245)
    fusion1_color = (150, 80, 210)
    fusion2_color = (90, 160, 30)

    rgb_baseline, rgb_count, rgb_shown = draw_detections(
        rgb, outputs["rgb"], display_conf, max_draw, rgb_color)
    ir_baseline, ir_count, ir_shown = draw_detections(
        ir, outputs["ir"], display_conf, max_draw, ir_color)
    fusion1, f1_count, f1_shown = draw_detections(
        rgb, outputs["fusion1"], display_conf, max_draw, fusion1_color)
    fusion2, f2_count, f2_shown = draw_detections(
        rgb, outputs["fusion2"], display_conf, max_draw, fusion2_color)

    panels = [
        tile(rgb, "RGB input"),
        tile(ir, "IR input"),
        tile(draw_gt(rgb, gt_boxes), f"Ground truth on RGB ({len(gt_boxes)})", (60, 190, 55)),
        tile(draw_gt(ir, gt_boxes), f"Ground truth on IR ({len(gt_boxes)})", (60, 190, 55)),
        tile(rgb_baseline, f"RGB baseline ({rgb_shown}/{rgb_count})", rgb_color),
        tile(ir_baseline, f"IR baseline ({ir_shown}/{ir_count})", ir_color),
        tile(fusion1, f"Fusion stage 1 ({f1_shown}/{f1_count})", fusion1_color),
        tile(fusion2, f"Fusion stage 2 ({f2_shown}/{f2_count})", fusion2_color),
    ]
    upper = np.concatenate(panels[:4], axis=1)
    lower = np.concatenate(panels[4:], axis=1)
    return np.concatenate((upper, lower), axis=0), {
        "gt_count": int(len(gt_boxes)),
        "rgb_predictions_display_conf": rgb_count,
        "ir_predictions_display_conf": ir_count,
        "fusion1_predictions_display_conf": f1_count,
        "fusion2_predictions_display_conf": f2_count,
    }


def main():
    args = parse_args()
    check_options(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 PyTorch 不能访问 GPU。")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    test = LLVIPFusionDataset(root=args.data_root, split="test", imgsz=args.imgsz)
    if getattr(test, "augment", False):
        raise RuntimeError("可视化测试集不能开启数据增强，请检查 fusion_dataset.py。")
    indices = sample_indices(test, args)
    print(f"测试集总数：{len(test)}，本次可视化样本：{len(indices)}")
    print(f"选择索引：{indices}")

    models = {}
    for name, weight in (("rgb", args.rgb_weights), ("ir", args.ir_weights)):
        wrapper = YOLO(weight)
        if wrapper.task != "detect":
            raise RuntimeError(f"{name} 不是 detect 模型：{weight}")
        models[name] = wrapper.model.to(device).eval()
        print(f"{name.upper()} 单模态已加载：{weight}")

    models["fusion1"] = load_fusion(
        args.rgb_weights, args.ir_weights, args.fusion1_weights, device)
    models["fusion2"] = load_fusion(
        args.rgb_weights, args.ir_weights, args.fusion2_weights, device)

    rows = []
    print(f"检测展示阈值：{args.display_conf}; NMS 阈值：conf={args.nms_conf}, IoU={args.nms_iou}")
    with torch.inference_mode():
        for order, index in enumerate(indices, 1):
            rgb_tensor, ir_tensor, labels = test[index]
            image_id = test.pairs[index][0].stem
            rgb_image = to_bgr(rgb_tensor)
            ir_image = to_bgr(ir_tensor)
            rgb_in = rgb_tensor.unsqueeze(0).to(device).float().div_(255.0)
            ir_in = ir_tensor.unsqueeze(0).to(device).float().div_(255.0)

            gt_boxes = xywhn_to_xyxy_pixels(labels[:, 1:5], args.imgsz).numpy()
            outputs = {
                "rgb": infer_one(models["rgb"], rgb_in, ir_in, "rgb", args),
                "ir": infer_one(models["ir"], rgb_in, ir_in, "ir", args),
                "fusion1": infer_one(models["fusion1"], rgb_in, ir_in, "fusion", args),
                "fusion2": infer_one(models["fusion2"], rgb_in, ir_in, "fusion", args),
            }

            panel, counts = make_comparison(
                rgb_image, ir_image, gt_boxes, outputs, args.display_conf, args.max_draw)
            file_path = output / f"comparison_{image_id}.png"
            if not cv2.imwrite(str(file_path), panel):
                raise OSError(f"图片保存失败：{file_path}")
            rows.append({"index": index, "image_id": image_id, "file": file_path.name, **counts})
            print(f"[{order}/{len(indices)}] 保存：{file_path.name} | GT={counts['gt_count']} | "
                  f"RGB/IR/F1/F2={counts['rgb_predictions_display_conf']}/"
                  f"{counts['ir_predictions_display_conf']}/"
                  f"{counts['fusion1_predictions_display_conf']}/"
                  f"{counts['fusion2_predictions_display_conf']}")

    with (output / "samples.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    settings = {
        "purpose": "Qualitative comparison only. Not a substitute for full test set metrics.",
        "split": "test",
        "gt_source": "RGB labels, transformed by fusion_dataset. Ground truth is shown over RGB and IR letterboxed inputs.",
        "imgsz": args.imgsz,
        "device": str(device),
        "seed": args.seed,
        "indices": indices,
        "nms_conf": args.nms_conf,
        "nms_iou": args.nms_iou,
        "display_conf": args.display_conf,
        "max_det": args.max_det,
        "max_draw": args.max_draw,
        "weights": {k: str(getattr(args, k)) for k in (
            "rgb_weights", "ir_weights", "fusion1_weights", "fusion2_weights")},
    }
    with (output / "visualization_settings.json").open("w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)

    print("\n全部可视化完成！")
    print(f"输出目录：{output}")
    print(f"生成 PNG：{len(rows)} 张")
    print("文件：samples.csv、visualization_settings.json、comparison_*.png")


if __name__ == "__main__":
    main()
