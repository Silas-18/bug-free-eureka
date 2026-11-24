"""Synchronized RGB-IR augmentation for paired pedestrian detection."""
import cv2
import numpy as np


def _transform_boxes(labels, matrix, width, height):
    """Apply a 2x3 affine transform to normalized xywh boxes."""
    if labels.size == 0:
        return labels.reshape(0, 5).astype(np.float32)
    output = []
    for row in labels:
        cls, xc, yc, bw, bh = row.tolist()
        x1, x2 = (xc - bw / 2) * width, (xc + bw / 2) * width
        y1, y2 = (yc - bh / 2) * height, (yc + bh / 2) * height
        corners = np.array(
            [[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]],
            dtype=np.float32,
        )
        transformed = corners @ matrix.T
        nx1 = float(np.clip(transformed[:, 0].min(), 0, width))
        ny1 = float(np.clip(transformed[:, 1].min(), 0, height))
        nx2 = float(np.clip(transformed[:, 0].max(), 0, width))
        ny2 = float(np.clip(transformed[:, 1].max(), 0, height))
        if nx2 <= nx1 or ny2 <= ny1:
            continue
        output.append([
            cls,
            (nx1 + nx2) / (2 * width),
            (ny1 + ny2) / (2 * height),
            (nx2 - nx1) / width,
            (ny2 - ny1) / height,
        ])
    return np.asarray(output, dtype=np.float32).reshape(-1, 5)


def augment_pair(rgb, ir, labels, geo_aug=True, color_aug=True):
    """Augment aligned image pairs; geometry is shared, photometric jitter is RGB-only.

    Inputs are RGB-order uint8 arrays and normalized YOLO labels [class, x, y, w, h].
    """
    if rgb.shape[:2] != ir.shape[:2]:
        raise ValueError(
            f"RGB/IR 尺寸不一致，不能执行同步增强：{rgb.shape[:2]} vs {ir.shape[:2]}"
        )
    labels = np.asarray(labels, dtype=np.float32).reshape(-1, 5).copy()
    height, width = rgb.shape[:2]

    if geo_aug:
        # Synchronized horizontal flip.
        if np.random.random() < 0.5:
            rgb = cv2.flip(rgb, 1)
            ir = cv2.flip(ir, 1)
            if len(labels):
                labels[:, 1] = 1.0 - labels[:, 1]

        # Synchronized mild scale/translation. Boxes receive the same transform.
        if np.random.random() < 0.30:
            scale = float(np.random.uniform(0.90, 1.10))
            tx = float(np.random.uniform(-0.04, 0.04) * width)
            ty = float(np.random.uniform(-0.04, 0.04) * height)
            matrix = np.array(
                [
                    [scale, 0.0, (1.0 - scale) * width / 2.0 + tx],
                    [0.0, scale, (1.0 - scale) * height / 2.0 + ty],
                ],
                dtype=np.float32,
            )
            rgb = cv2.warpAffine(
                rgb, matrix, (width, height), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114),
            )
            ir = cv2.warpAffine(
                ir, matrix, (width, height), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114),
            )
            labels = _transform_boxes(labels, matrix, width, height)

    if color_aug:
        # Visible-light color augmentation only; thermal intensity is not hue-jittered.
        if np.random.random() < 0.80:
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
            gains = np.random.uniform(-1.0, 1.0, size=3).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + gains[0] * 8.0) % 180.0
            hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + gains[1] * 0.25), 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] * (1.0 + gains[2] * 0.25), 0, 255)
            rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        if np.random.random() < 0.30:
            alpha = float(np.random.uniform(0.85, 1.15))
            beta = float(np.random.uniform(-12.0, 12.0))
            rgb = np.clip(rgb.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    return rgb, ir, labels
