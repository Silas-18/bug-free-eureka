
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np

# 项目根目录
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "deliverables" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# 完整测试集实验数据
data = [
    ["Stage1 Full", 0.864518, 0.886467, 0.931583, 0.654536],
    ["Stage2 Full", 0.896688, 0.927186, 0.962336, 0.696736],
    ["B No Geo", 0.882137, 0.921437, 0.959045, 0.697307],
    ["C No Color", 0.886228, 0.921677, 0.958782, 0.693343],
    ["D No Aug", 0.875544, 0.914970, 0.954958, 0.694987],
    ["E Avg", 0.885935, 0.933892, 0.962291, 0.692862],
    ["F Concat", 0.902852, 0.932695, 0.965219, 0.699173],
]

# 保存 CSV
csv_path = ROOT / "deliverables" / "final_metrics.csv"

with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Experiment", "Precision", "Recall",
        "mAP50", "mAP50-95"
    ])
    writer.writerows(data)

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False

names = [x[0] for x in data]
map50 = [x[3] for x in data]
map5095 = [x[4] for x in data]

# 图1：所有实验组 mAP 对比
x = np.arange(len(names))
width = 0.36

fig, ax = plt.subplots(figsize=(12, 6))
b1 = ax.bar(x - width/2, map50, width, label="mAP@0.5")
b2 = ax.bar(x + width/2, map5095, width, label="mAP@0.5:0.95")

ax.set_ylabel("mAP")
ax.set_title("Performance Comparison of All Experiments")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=20, ha="right")
ax.set_ylim(0, 1.0)
ax.legend()
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(OUT / "all_experiments_map.png", dpi=300)
plt.close(fig)

# 图2：数据增强消融
aug_names = ["Stage2 Full", "B No Geo", "C No Color", "D No Aug"]
aug_map50 = map50[1:5]
aug_map5095 = map5095[1:5]

x = np.arange(4)
fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(x - width/2, aug_map50, width, label="mAP@0.5")
ax.bar(x + width/2, aug_map5095, width, label="mAP@0.5:0.95")

ax.set_title("Data Augmentation Ablation")
ax.set_xticks(x)
ax.set_xticklabels(aug_names)
ax.set_ylim(0, 1.0)
ax.legend()
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(OUT / "augmentation_ablation.png", dpi=300)
plt.close(fig)

# 图3：融合方式对比
fig, ax = plt.subplots(figsize=(7, 5))

fusion_names = ["Avg", "Concat"]
fusion_map50 = [data[5][3], data[6][3]]
fusion_map5095 = [data[5][4], data[6][4]]

x = np.arange(2)
ax.bar(x - width/2, fusion_map50, width, label="mAP@0.5")
ax.bar(x + width/2, fusion_map5095, width, label="mAP@0.5:0.95")

ax.set_title("Fusion Strategy Comparison")
ax.set_xticks(x)
ax.set_xticklabels(fusion_names)
ax.set_ylim(0, 1.0)
ax.legend()
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(OUT / "fusion_comparison.png", dpi=300)
plt.close(fig)

print("结果整理完成！")
print("CSV:", csv_path)
print("图表目录:", OUT)
for p in OUT.glob("*.png"):
    print(p.name)
