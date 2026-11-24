
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


def read_image_list(txt_path):
    with open(txt_path, "r", encoding="utf-8-sig") as f:
        return [
            Path(line.strip().strip('"'))
            for line in f if line.strip()
        ]


def letterbox(image, labels, size=640):
    h, w = image.shape[:2]

    ratio = min(size / h, size / w)
    nw, nh = round(w * ratio), round(h * ratio)

    image = cv2.resize(image, (nw, nh))

    dw = (size - nw) / 2
    dh = (size - nh) / 2

    left = round(dw - 0.1)
    right = round(dw + 0.1)
    top = round(dh - 0.1)
    bottom = round(dh + 0.1)

    image = cv2.copyMakeBorder(
        image, top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )

    labels = labels.copy()

    if len(labels):
        labels[:, 1] = (
            labels[:, 1] * w * ratio + left
        ) / size

        labels[:, 2] = (
            labels[:, 2] * h * ratio + top
        ) / size

        labels[:, 3] = (
            labels[:, 3] * w * ratio
        ) / size

        labels[:, 4] = (
            labels[:, 4] * h * ratio
        ) / size

    return image, labels


class LLVIPFusionDataset(Dataset):

    def __init__(self, root, split="train", imgsz=640):
        self.root = Path(root)
        self.imgsz = imgsz

        rgb_root = self.root / "llvip_rgb_721"
        ir_root = self.root / "llvip_ir_721"

        rgb_list = read_image_list(
            rgb_root / f"{split}.txt"
        )

        ir_list = read_image_list(
            ir_root / f"{split}.txt"
        )

        rgb_dict = {p.stem: p for p in rgb_list}
        ir_dict = {p.stem: p for p in ir_list}

        if set(rgb_dict) != set(ir_dict):
            raise ValueError("RGB 和 IR 图片无法完全配对！")

        self.pairs = [
            (rgb_dict[name], ir_dict[name])
            for name in sorted(rgb_dict)
        ]

        print(f"{split} 集配对数量：{len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):

        rgb_path, ir_path = self.pairs[index]

        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        ir = cv2.imread(str(ir_path), cv2.IMREAD_COLOR)

        if rgb is None or ir is None:
            raise FileNotFoundError(
                f"图像读取失败：{rgb_path} 或 {ir_path}"
            )

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        ir = cv2.cvtColor(ir, cv2.COLOR_BGR2RGB)

        # 使用 RGB 对应的行人标注
        label_path = (
            rgb_path.parent.parent.parent
            / "labels"
            / rgb_path.parent.name
            / f"{rgb_path.stem}.txt"
        )

        if not label_path.exists():
            raise FileNotFoundError(
                f"标签文件不存在：{label_path}"
            )

        if label_path.stat().st_size:
            labels = np.loadtxt(
                label_path, dtype=np.float32
            ).reshape(-1, 5)
        else:
            labels = np.zeros((0, 5), dtype=np.float32)

        rgb, labels = letterbox(
            rgb, labels, self.imgsz
        )

        ir, _ = letterbox(
            ir,
            np.zeros((0, 5), dtype=np.float32),
            self.imgsz
        )

        rgb = torch.from_numpy(
            rgb.transpose(2, 0, 1).copy()
        )

        ir = torch.from_numpy(
            ir.transpose(2, 0, 1).copy()
        )

        labels = torch.from_numpy(labels)

        return rgb, ir, labels


def fusion_collate_fn(batch):

    rgbs, irs, labels = zip(*batch)

    cls_list = []
    box_list = []
    index_list = []

    for i, label in enumerate(labels):
        if len(label):
            cls_list.append(label[:, 0:1])
            box_list.append(label[:, 1:5])

            index_list.append(
                torch.full(
                    (len(label),),
                    i,
                    dtype=torch.float32
                )
            )

    if cls_list:
        cls = torch.cat(cls_list, dim=0)
        boxes = torch.cat(box_list, dim=0)
        batch_idx = torch.cat(index_list, dim=0)
    else:
        cls = torch.zeros((0, 1))
        boxes = torch.zeros((0, 4))
        batch_idx = torch.zeros((0,))

    targets = {
        "cls": cls,
        "bboxes": boxes,
        "batch_idx": batch_idx
    }

    return (
        torch.stack(rgbs),
        torch.stack(irs),
        targets
    )
