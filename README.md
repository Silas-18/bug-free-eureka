# LLVIP：可见光—红外双模态行人检测

基于 LLVIP 可见光（RGB）与红外（IR）配对数据集，使用 YOLOv8n 单模态基线和双分支融合检测模型，完成基线对比、融合实验与消融分析。

> 本仓库不包含原始 LLVIP 数据集、预训练权重和训练 checkpoint。请先按本文件下载数据并准备权重。实验结果目录中的指标是已有实验记录，不代表新环境已自动复现。

## 1. 项目任务

- 数据集：LLVIP Visible-Infrared Paired Dataset。
- 任务：单类别行人目标检测（person）。
- 模态：可见光 RGB 与红外 IR。
- 单模态基线：RGB YOLOv8n、IR YOLOv8n。
- 融合：双分支特征提取，多尺度融合；另比较逐元素平均（avg）与拼接后卷积（concat）。
- 实验：Stage1、Stage2、几何增强/颜色增强消融、融合结构对比。

## 2. 仓库结构

```text
bug-free-eureka/
├── code/
│   ├── prepare_rgb.py              # 原始 XML 标注转 RGB YOLO 数据
│   ├── prepare_ir.py               # 按 RGB 划分构建 IR 数据
│   ├── fusion_augment.py            # RGB-IR 同步增强
│   ├── fusion_dataset.py            # 双模态数据读取与配对
│   ├── fusion_model.py              # 双分支融合模型
│   ├── fusion_model_structure.py    # avg / concat 结构对比模型
│   ├── train.py                     # RGB 基线训练
│   ├── train_ir_only.py             # IR 基线训练
│   ├── train_fusion_stage1.py       # 第一阶段融合训练
│   ├── train_fusion.py              # 融合训练入口
│   ├── train_fusion_structure.py    # avg / concat 与增强消融训练
│   ├── eval_baselines.py            # RGB/IR 统一测试集评估
│   ├── eval_fusion.py               # 融合模型测试集评估
│   ├── eval_fusion_structure.py     # 结构融合模型测试集评估
│   ├── check_dataset_integrity.py   # 数据完整性检查（旧版专用目录结构）
│   └── demo_paired_samples.py       # RGB/IR 配对样本小样例
├── dataset/                         # 本地数据，不提交 Git
│   ├── LLVIP/                       # 下载并解压的原始数据
│   ├── llvip_rgb_721/               # 自动生成的 RGB YOLO 数据
│   └── llvip_ir_721/                # 自动生成的 IR YOLO 数据
├── experiments/                     # 训练日志、评估指标、可视化
├── results/                         # 数据检查和 Demo 输出
├── deliverables/                    # 汇总 CSV 和实验图
├── requirements.txt
└── README.md
```

## 3. 环境配置

建议使用 Python 3.10 或 3.11、64 位 Windows/Linux，以及可用 NVIDIA GPU。具体 PyTorch/CUDA 版本应根据本机驱动选择；不要盲目安装与显卡环境不匹配的 CUDA wheel。

### 3.1 创建环境

```bash
conda create -n llvip python=3.10 -y
conda activate llvip
```

### 3.2 安装 PyTorch

先根据 PyTorch 官方安装页面选择适配本机 CUDA 的命令：

https://pytorch.org/get-started/locally/

安装完成后检查：

```bash
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 3.3 安装项目依赖

```bash
pip install -r requirements.txt
```

检查关键包：

```bash
python -c "import torch, torchvision, ultralytics, cv2, numpy, PIL, yaml; print('environment OK')"
```

如网络或 CUDA 版本导致安装失败，请先单独按 PyTorch 官网安装 torch/torchvision，再安装 requirements.txt 中其余依赖。

## 4. 数据集下载与准备

### 4.1 下载 LLVIP

请从数据集作者提供的官方页面下载，不要从来历不明的网盘镜像获取：

- LLVIP 官方项目：https://github.com/bupt-ai-cz/LLVIP
- 官方下载说明：https://github.com/bupt-ai-cz/LLVIP/blob/main/download_dataset.md
- 项目主页：https://bupt-ai-cz.github.io/LLVIP/

官方说明列出 Google Drive 与百度网盘下载方式。百度网盘提取码以官方页面当前公布内容为准。请阅读并遵守数据集许可：该数据集面向学术研究、教学和个人实验等非商业用途，使用时应引用 LLVIP 论文。

### 4.2 解压目录

下载并解压后，将数据放到仓库根目录的 `dataset/LLVIP/`，目录至少应包含：

```text
dataset/LLVIP/
├── visible/
│   ├── train/
│   └── test/
├── infrared/
│   ├── train/
│   └── test/
└── Annotations/
    ├── *.xml
    └── ...
```

注意：文件夹名称大小写应与示例一致。若下载包多套了一层目录，请把包含 visible、infrared、Annotations 的那一层作为 LLVIP 根目录。

### 4.3 生成 YOLO 格式数据

在仓库根目录执行：

```bash
python code/prepare_rgb.py
python code/prepare_ir.py
```

脚本会检查 RGB、IR 与 XML 的文件 ID 是否匹配，然后生成：

```text
dataset/
├── LLVIP/
├── llvip_rgb_721/
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   ├── train.txt
│   ├── val.txt
│   ├── test.txt
│   ├── rgb.yaml
│   └── rgb_smoke.yaml
└── llvip_ir_721/
    ├── images/{train,val,test}/
    ├── labels/{train,val,test}/
    ├── train.txt
    ├── val.txt
    ├── test.txt
    ├── ir.yaml
    └── ir_smoke.yaml
```

默认划分设置为随机种子 42、训练集 70%、验证集 20%、测试集 10%。RGB 脚本生成划分清单，IR 脚本严格复用同一份清单，确保两种模态的样本划分一致。该划分方式是本项目处理脚本当前采用的随机划分，不等同于直接沿用 LLVIP 原始 train/test 官方划分。已有实验结果应结合其当时使用的数据划分和日志解释，不要把不同划分的结果直接混为一谈。

若需要重新生成非空输出目录，先确认目录内没有需要保留的文件，再分别添加 `--overwrite`：

```bash
python code/prepare_rgb.py --overwrite
python code/prepare_ir.py --overwrite
```

## 5. 小样本 Demo

准备好原始数据后，在仓库根目录运行：

```bash
python code/demo_paired_samples.py --num-samples 4 --seed 42
```

输出文件：

```text
results/paired_demo.png
```

该 Demo 从数据集中抽取若干样本，将同一文件 ID 的 RGB 与 IR 图像并排展示，用于快速检查双模态配对和数据读取。它是数据可视化 Demo，不等于模型检测推理 Demo。

## 6. 训练与复现命令

以下命令均从仓库根目录运行。首次完整训练前，请确认第 4 节数据准备成功，并且磁盘空间、GPU 显存足够。训练会消耗较长时间。

### 6.1 RGB 单模态基线

```bash
python code/train.py --epochs 100 --batch 4 --imgsz 640 --device 0
```

CPU 调试可将 `--device 0` 改成 `--device cpu`，并将 epochs 降低；CPU 调试结果不作为正式性能结果。

### 6.2 IR 单模态基线

```bash
python code/train_ir_only.py --epochs 100 --batch 4 --imgsz 640 --device 0
```

训练完成后，默认权重位置：

```text
experiments/baseline/v8_rgb_baseline/weights/best.pt
experiments/baseline/v8_ir_baseline/weights/best.pt
```

### 6.3 双模态融合训练

```bash
python code/train_fusion.py --epochs 100 --batch-size 1 --imgsz 640 --device cuda:0
```

融合训练需要先完成 RGB、IR 两个单模态基线训练，以提供初始化权重。若使用其他权重位置，请通过 `--rgb-weights`、`--ir-weights` 指定。

### 6.4 第一阶段融合训练

```bash
python code/train_fusion_stage1.py --epochs 30 --batch-size 1 --imgsz 640 --device cuda:0
```

Stage1 默认使用同步几何增强与 RGB 颜色增强；具体训练参数以脚本的 argparse 配置为准。

### 6.5 融合结构对比：Concat / Avg

Concat：

```bash
python code/train_fusion_structure.py --fusion-mode concat --epochs 100 --batch-size 1 --imgsz 640 --device cuda:0 --output experiments/structure/F_concat_reproduction
```

Avg：

```bash
python code/train_fusion_structure.py --fusion-mode avg --epochs 100 --batch-size 1 --imgsz 640 --device cuda:0 --output experiments/structure/E_avg_reproduction
```

关闭几何增强：

```bash
python code/train_fusion_structure.py --fusion-mode concat --no-geo-aug --output experiments/ablation/B_no_geo_reproduction
```

关闭颜色增强：

```bash
python code/train_fusion_structure.py --fusion-mode concat --no-color-aug --output experiments/ablation/C_no_color_reproduction
```

同时关闭两类增强：

```bash
python code/train_fusion_structure.py --fusion-mode concat --no-geo-aug --no-color-aug --output experiments/ablation/D_no_aug_reproduction
```

### 6.6 单模态基线统一评估

```bash
python code/eval_baselines.py --data-root dataset --mode both --device cuda:0 --batch-size 4
```

### 6.7 融合模型测试集评估

```bash
python code/eval_fusion.py --data-root dataset --device cuda:0 --batch-size 4
```

如需评估结构对比模型，请指定对应 checkpoint 和融合模式：

```bash
python code/eval_fusion_structure.py --data-root dataset --fusion-mode concat --fusion-weights experiments/structure/F_concat_reproduction/weights/best.pt --output experiments/structure/F_concat_reproduction/evaluation_test --device cuda:0
```

### 6.8 调试模式与正式评估的区别

部分脚本提供 `--max-batches`，用于快速检查模型是否能前向运行、数据能否加载。只要设置为大于 0，就是部分测试，不可将其指标写成完整测试集结果。正式测试应省略该参数或显式设置为 0。

## 7. 结果文件与复现核验

训练日志和模型通常保存在 `experiments/` 下。评估脚本会输出 `metrics.json`、`metrics.csv` 和逐类别指标文件。最终汇总表位于：

- `deliverables/final_metrics_all9.csv`
- `deliverables/final_metrics.csv`

绘图脚本：

```bash
python deliverables/generate_metric_figures.py
python deliverables/generate_loss_curves.py
```

建议复现时记录以下信息：

1. 数据集版本、下载日期、原始文件数量和数据划分随机种子；
2. Python、PyTorch、torchvision、Ultralytics 版本；
3. GPU 型号、显存、CUDA 是否可用；
4. 训练命令、权重初始化来源、epoch、batch size、imgsz、seed；
5. 测试集图像数、GT 数、NMS 参数、P/R 置信度阈值和 AP 计算口径；
6. `best.pt`、`last.pt`、训练 CSV、测试 JSON/CSV 和可视化输出。

## 8. 当前复现边界与注意事项

- 原始 LLVIP 数据集和模型权重不在 Git 中，必须自行下载/生成/训练或取得对应 checkpoint。
- README 中的命令是项目入口说明；完整复现仍依赖本地数据目录和训练得到的权重文件。
- 现有仓库中部分历史脚本曾使用 Windows 绝对路径；本次整理的数据准备脚本和 RGB/IR 基线训练入口改为相对仓库路径。其他历史脚本若有自定义路径，应使用命令行参数覆盖。
- 当前项目的随机划分设置与 LLVIP 官方 train/test 划分并不相同。重新划分会改变训练/测试样本，因此新跑出的指标不一定与仓库保存的历史指标完全相同。
- 单模态与融合评估应使用相同的测试样本、标注、图像尺寸、NMS 参数与指标计算口径。
- 数据集许可限制非商业使用。请在报告和 PPT 中引用原始 LLVIP 论文。

## 9. 参考文献与数据集引用

Jia, X., Zhu, C., Li, M., Tang, W., & Zhou, W. (2021). LLVIP: A visible-infrared paired dataset for low-light vision. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 3496–3504.

BibTeX：

```bibtex
@inproceedings{jia2021llvip,
  title={LLVIP: A visible-infrared paired dataset for low-light vision},
  author={Jia, Xinyu and Zhu, Chuang and Li, Minzhen and Tang, Wenqi and Zhou, Wenli},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={3496--3504},
  year={2021}
}
```
