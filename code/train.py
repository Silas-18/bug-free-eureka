import time
from ultralytics import YOLO

#训练rgb图像
model = YOLO("yolov8n.pt") # load a pretrained model (recommended for training)
results = model.train(data=r'E:/LLVIP_Project/dataset/llvip_rgb_721/rgb.yaml', project="./runs", epochs=100, imgsz=640, device=0, workers=0, batch=4, amp=False, name="v8_rgb_baseline") # CPU 开始训练
