# LLVIP: Visible-Infrared Multimodal Pedestrian Detection

基于 LLVIP 数据集的可见光（RGB）与红外（IR）双模态行人检测项目。

## 1. 项目概述

本项目以 YOLOv8 为检测基础，研究可见光和红外图像的特征融合方法，并通过基线实验、数据增强消融实验以及融合结构对比实验评估检测效果。

## 2. 数据集

使用 LLVIP（Low-Light Visible-Infrared Paired）数据集。

原始数据包括：
- 15,488 张可见光图像
- 15,488 张红外图像
- 对应的目标检测标注

数据划分：
- Train：10,842 对
- Validation：3,097 对
- Test：1,549 对

原始数据集未包含在本仓库中。

## 3. 实验方案

共整理九组实验：

1. RGB Baseline
2. IR Baseline
3. Stage1 Full
4. Stage2 Full
5. B No Geo
6. C No Color
7. D No Aug
8. E Avg
9. F Concat

融合模型采用可见光与红外双分支特征提取，在多个特征尺度上进行融合，并通过检测头输出预测结果。

## 4. 实验结果

以下为统一测试集评估结果。

| Experiment | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| RGB Baseline | 0.790654 | 0.802395 | 0.855300 | 0.549503 |
| IR Baseline | 0.863393 | 0.926467 | 0.951333 | 0.689412 |
| Stage1 Full | 0.864518 | 0.886467 | 0.931583 | 0.654536 |
| Stage2 Full | 0.896688 | 0.927186 | 0.962336 | 0.696736 |
| B No Geo | 0.882137 | 0.921437 | 0.959045 | 0.697307 |
| C No Color | 0.886228 | 0.921677 | 0.958782 | 0.693343 |
| D No Aug | 0.875544 | 0.914970 | 0.954958 | 0.694987 |
| E Avg | 0.885935 | 0.933892 | 0.962291 | 0.692862 |
| F Concat | 0.902852 | 0.932695 | 0.965219 | 0.699173 |

在当前实验设置下，F Concat 取得最高 Precision、mAP50 和 mAP50-95。

详细实验数据位于：
`deliverables/final_metrics_all9.csv`

## 5. 仓库结构

- `code/`：数据准备、融合模型、训练和评估代码
- `experiments/`：各实验的训练记录与评估结果
- `results/`：数据集检查、统计及配对可视化
- `deliverables/figures/`：最终实验图表
- `deliverables/`：指标汇总及绘图脚本

## 6. 环境与复现

实验基于 Python、PyTorch 和 Ultralytics YOLO 实现。

依赖项参考 `requirements.txt`。

运行前需自行准备 LLVIP 数据集与模型权重，并按照实际环境调整数据路径和配置。

各脚本的具体运行参数以代码实现为准。

## 7. 说明

本仓库保存课程项目代码、实验记录与可视化结果。
原始数据、模型权重及大型备份文件未纳入 Git 仓库。

Report 和 PPT 将在后续补充。
