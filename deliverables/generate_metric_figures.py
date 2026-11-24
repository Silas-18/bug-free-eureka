import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent.parent
csv_path = root / "deliverables" / "final_metrics_all9.csv"
out_dir = root / "deliverables" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

with open(csv_path, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

names = [r["Experiment"] for r in rows]
precision = [float(r["Precision"]) for r in rows]
recall = [float(r["Recall"]) for r in rows]
map5095 = [float(r["mAP50_95"]) for r in rows]

plt.rcParams["axes.unicode_minus"] = False

# Figure 1: Precision and Recall
x = np.arange(len(names))
width = 0.36

fig, ax = plt.subplots(figsize=(12, 6))

bars1 = ax.bar(x - width/2, precision, width, label="Precision")
bars2 = ax.bar(x + width/2, recall, width, label="Recall")

ax.set_title("Precision and Recall Comparison", fontsize=15)
ax.set_ylabel("Score")
ax.set_xlabel("Experiment")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=25, ha="right")
ax.set_ylim(0, 1.08)
ax.legend(loc="upper left")
ax.grid(axis="y", alpha=0.25)

# 兼容 Matplotlib 3.3.4：手动添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.008,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

fig.tight_layout()
fig.savefig(
    out_dir / "precision_recall_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

# Figure 2: mAP50-95
fig, ax = plt.subplots(figsize=(12, 6))

bars = ax.bar(names, map5095)

ax.set_title("mAP50-95 Comparison", fontsize=15)
ax.set_ylabel("mAP50-95")
ax.set_xlabel("Experiment")
ax.set_ylim(0, 1.08)
ax.tick_params(axis="x", rotation=25)
ax.grid(axis="y", alpha=0.25)

# 手动添加数值标签
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.008,
        f"{height:.4f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

fig.tight_layout()
fig.savefig(
    out_dir / "map5095_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

print("\n图表生成成功！")
print("输出目录：", out_dir)
print("1. precision_recall_comparison.png")
print("2. map5095_comparison.png")
