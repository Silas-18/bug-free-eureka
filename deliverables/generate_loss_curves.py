import csv
from pathlib import Path
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent.parent
out_dir = root / "deliverables" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

experiments = {
    "Stage2 Full": {
        "file": root / "experiments/fusion/stage2_bs4_fp32/metrics.csv",
        "name": "stage2_full_loss.png"
    },
    "E Avg": {
        "file": root / "experiments/structure/E_avg_full/metrics.csv",
        "name": "e_avg_loss.png"
    },
    "F Concat": {
        "file": root / "experiments/structure/F_concat_full/metrics.csv",
        "name": "f_concat_loss.png"
    }
}

data = {}

for name, config in experiments.items():
    with open(config["file"], encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    val_loss = [float(r["val_loss"]) for r in rows]

    data[name] = {
        "epochs": epochs,
        "train_loss": train_loss,
        "val_loss": val_loss
    }

    print(name, "训练轮数:", len(rows))

    # 单组实验损失曲线
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, train_loss, label="Train Loss", linewidth=1.8)
    ax.plot(epochs, val_loss, label="Val Loss", linewidth=1.8)

    ax.set_title(name + " Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / config["name"], dpi=300)
    plt.close(fig)

# 三组实验验证损失对比
fig, ax = plt.subplots(figsize=(11, 6))

for name, values in data.items():
    ax.plot(
        values["epochs"],
        values["val_loss"],
        label=name,
        linewidth=1.8
    )

ax.set_title("Validation Loss Comparison")
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Loss")
ax.legend()
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(out_dir / "validation_loss_comparison.png", dpi=300)
plt.close(fig)

print("\n四张训练损失曲线生成成功！")
print("保存目录:", out_dir)

for f in out_dir.glob("*loss*.png"):
    print(f.name, f.stat().st_size, "bytes")
