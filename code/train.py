"""Train the visible-light YOLOv8n baseline."""
import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "dataset/llvip_rgb_721/rgb.yaml"))
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--name", default="v8_rgb_baseline")
    args = p.parse_args()
    if not Path(args.data).is_file():
        raise FileNotFoundError(f"数据配置不存在：{args.data}；请先运行数据准备脚本。")
    model = YOLO(args.weights)
    model.train(data=args.data, project=str(ROOT / "experiments/baseline"),
                epochs=args.epochs, imgsz=args.imgsz, device=args.device,
                workers=args.workers, batch=args.batch, amp=False, name=args.name)

if __name__ == "__main__":
    main()
