"""Regenerate plots from the v7 sft_log.json + evals.json."""

import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

# ---------- SFT loss curve ----------
log = json.loads(Path("outputs/sft_log_v7.json").read_text())
points = [(e["epoch"], e["loss"]) for e in log if "loss" in e and "epoch" in e]
epochs, losses = zip(*points)

fig, ax = plt.subplots(figsize=(8, 4.4))
ax.plot(epochs, losses, color="#2a6df4", linewidth=1.5, alpha=0.9)
ax.scatter(epochs, losses, color="#2a6df4", s=8, zorder=3)
ax.set_xlabel("Training progress (fraction of 1 epoch over 16,000 samples)")
ax.set_ylabel("Cross-entropy loss")
ax.set_title("SFT loss — Qwen 2.5 3B + LoRA r=16 on LeniencyBench (v7 onsite run)")
ax.set_yscale("log")
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(OUT / "sft_loss.png", dpi=150)
plt.close(fig)
print(f"wrote outputs/sft_loss.png  ({len(points)} points, {losses[0]:.3f} -> {losses[-1]:.3f})")

# ---------- direction-split bar chart ----------
evals = json.loads(Path("outputs/evals_v7.json").read_text())
def acc(d, dirn):
    return d["drift_acc_by_direction"].get(dirn) or 0.0

stages = [("Pre-training", evals["pre"], "#d5342a"),
          ("Post-SFT",    evals["post_sft"], "#2a6df4")]

dirs = ["tightening", "loosening"]
labels = ["Tightening", "Loosening"]
import numpy as np
x = np.arange(len(labels))
width = 0.36

fig, ax = plt.subplots(figsize=(8, 4.6))
for i, (label, ev, color) in enumerate(stages):
    vals = [acc(ev, d) * 100 for d in dirs]
    bars = ax.bar(x + (i - 0.5) * width, vals, width, label=label, color=color)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.6,
                f"{b.get_height():.1f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_ylabel("Drift-sensitive accuracy")
ax.set_ylim(0, 105)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_yticklabels([f"{v}%" for v in [0, 20, 40, 60, 80, 100]])
ax.set_title("Qwen 2.5 3B on LeniencyBench: leniency bias before vs after SFT (v7)")
ax.legend(loc="upper left", framealpha=0.95)
ax.grid(alpha=0.2, axis="y")
fig.tight_layout()
fig.savefig(OUT / "direction_split.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote outputs/direction_split.png")

print("done")
