
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG, get_cfg


class DualModalYOLOv8(nn.Module):

    def __init__(self, rgb_weights, ir_weights, fusion_mode='concat'):
        super().__init__()
        if fusion_mode not in ('concat', 'avg'):
            raise ValueError(f'Unknown fusion_mode: {fusion_mode}')
        self.fusion_mode = fusion_mode

        print("正在加载 RGB 预训练权重...")
        rgb_detector = YOLO(rgb_weights).model

        print("正在加载 IR 预训练权重...")
        ir_detector = YOLO(ir_weights).model

        # 转换配置格式，兼容 Ultralytics
        # 加载 YOLO 默认训练配置，补齐损失计算所需参数
        rgb_detector.args = get_cfg(DEFAULT_CFG)

        print("YOLO 损失配置加载完成！")
        print("box:", rgb_detector.args.box)
        print("cls:", rgb_detector.args.cls)
        print("dfl:", rgb_detector.args.dfl)

        # RGB Backbone
        self.rgb_backbone = nn.ModuleList(
            list(rgb_detector.model[:10])
        )

        # IR Backbone
        self.ir_backbone = nn.ModuleList(
            list(ir_detector.model[:10])
        )

        # 保留 RGB Neck 和 Detect Head
        rgb_head = list(rgb_detector.model[10:])

        rgb_detector.model = nn.Sequential(
            *(
                [nn.Identity() for _ in range(10)]
                + rgb_head
            )
        )

        self.detector = rgb_detector

        # P3、P4、P5 融合层
        channels = [64, 128, 256]

        self.fusions = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    c * 2,
                    c,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(c),
                nn.SiLU(inplace=True)
            )
            for c in channels
        ])

        # avg requires no learned fusion module; keep the .fusions attribute
        # for compatibility with the existing parameter-freezing code.
        if self.fusion_mode == 'avg':
            self.fusions = nn.ModuleList([nn.Identity() for _ in channels])

        print(f'融合方式: {self.fusion_mode}')
        print("双模态融合模型初始化完成！")

    def forward_backbone(self, modules, image):

        outputs = []
        x = image

        for m in modules:

            if m.f != -1:
                if isinstance(m.f, int):
                    x = outputs[m.f]
                else:
                    x = [
                        x if j == -1 else outputs[j]
                        for j in m.f
                    ]

            x = m(x)
            outputs.append(x)

        return outputs[4], outputs[6], outputs[9]

    def forward(self, rgb, ir):

        rgb_features = self.forward_backbone(
            self.rgb_backbone, rgb
        )

        ir_features = self.forward_backbone(
            self.ir_backbone, ir
        )

        fused_features = []

        for i in range(3):
            if self.fusion_mode == 'avg':
                # Same geometry and channel count as the original features.
                feature = (rgb_features[i] + ir_features[i]) * 0.5
            else:
                feature = torch.cat([rgb_features[i], ir_features[i]], dim=1)
                feature = self.fusions[i](feature)
            fused_features.append(feature)

        # 构建 YOLO Neck 所需的特征索引
        y = [None] * 10

        y[4] = fused_features[0]
        y[6] = fused_features[1]
        y[9] = fused_features[2]

        x = fused_features[2]

        # 执行 Neck 和 Detect Head
        for m in self.detector.model[10:]:

            if m.f != -1:
                if isinstance(m.f, int):
                    x = y[m.f]
                else:
                    x = [
                        x if j == -1 else y[j]
                        for j in m.f
                    ]

            x = m(x)
            y.append(x)

        return x

    def loss(self, batch, preds):
        return self.detector.loss(batch, preds)
