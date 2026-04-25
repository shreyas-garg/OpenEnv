"""Generate plots from the v6 partial run logs + Llama baseline.

Outputs (all 150 dpi PNG, axis-labeled, captioned):
  outputs/sft_loss_v6.png             SFT loss curve over 1000 steps (3B run)
  outputs/direction_split_v6.png      Pre vs post-SFT direction-split accuracy
  outputs/cross_model_baseline.png    Llama 3.1 8B vs Qwen 2.5 3B leniency bias
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = Path("outputs/v6_full_logs.txt")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Parse SFT loss curve from v6 logs
# ---------------------------------------------------------------------------
def parse_sft_losses() -> list[tuple[float, float]]:
    """Return list of (epoch, loss) parsed from `{'loss': X, ..., 'epoch': Y}` lines."""
    text = LOG.read_text()
    pattern = re.compile(r"\{'loss': '([\d\.e\-]+)'.*?'epoch': '([\d\.]+)'\}")
    out: list[tuple[float, float]] = []
    for m in pattern.finditer(text):
        try:
            loss = float(m.group(1))
            epoch = float(m.group(2))
        except ValueError:
            continue
        if epoch <= 1.0:                          # only SFT (epoch \in [0,1])
            out.append((epoch, loss))
    return out


# ---------------------------------------------------------------------------
# Plot 1: SFT loss curve
# ---------------------------------------------------------------------------
def plot_sft_loss():
    pts = parse_sft_losses()
    if not pts:
        print("[skip] no SFT loss points found in v6 logs")
        return
    epochs, losses = zip(*pts)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(epochs, losses, color="#2a6df4", linewidth=1.5, alpha=0.85)
    ax.scatter(epochs, losses, color="#2a6df4", s=10, zorder=3)
    ax.set_xlabel("Training progress (fraction of 1 epoch over 16,000 samples)")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("SFT loss over training (Qwen 2.5 3B + LoRA r=16, lr=2e-4)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = OUT / "sft_loss_v6.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"wrote {p}  ({len(pts)} points, loss {losses[0]:.3f} -> {losses[-1]:.3f})")


# ---------------------------------------------------------------------------
# Plot 2: Pre vs post-SFT direction-split accuracy on Qwen 2.5 3B
# ---------------------------------------------------------------------------
def plot_direction_split_v6():
    # From v6 logs, hardcoded (already extracted):
    pre  = {"tightening": (0,  23), "loosening": (3,  14)}
    post = {"tightening": (21, 23), "loosening": (10, 14)}

    labels = ["Tightening", "Loosening"]
    pre_acc  = [pre[k.lower()][0]  / pre[k.lower()][1]  * 100 for k in labels]
    post_acc = [post[k.lower()][0] / post[k.lower()][1] * 100 for k in labels]

    x = range(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.6))
    b1 = ax.bar([i - width/2 for i in x], pre_acc,  width, label="Pre-training",
                color="#d5342a")
    b2 = ax.bar([i + width/2 for i in x], post_acc, width, label="Post-SFT (1 epoch)",
                color="#2a6df4")

    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.6,
                    f"{b.get_height():.1f}%",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Annotate with raw counts under each bar
    for i, lbl in enumerate(labels):
        c_pre, t_pre   = pre[lbl.lower()]
        c_post, t_post = post[lbl.lower()]
        ax.text(i - width/2, -7, f"{c_pre}/{t_pre}",  ha="center", fontsize=9, color="#666")
        ax.text(i + width/2, -7, f"{c_post}/{t_post}", ha="center", fontsize=9, color="#666")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Drift-sensitive accuracy")
    ax.set_ylim(-10, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels([f"{v}%" for v in [0, 20, 40, 60, 80, 100]])
    ax.set_title("Qwen 2.5 3B: leniency bias before vs after one epoch of SFT")
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(alpha=0.2, axis="y")
    ax.axhline(0, color="#888", linewidth=0.6)

    fig.tight_layout()
    p = OUT / "direction_split_v6.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


# ---------------------------------------------------------------------------
# Plot 3: Cross-model baseline (Llama 3.1 8B vs Qwen 2.5 3B, both untrained)
# ---------------------------------------------------------------------------
def plot_cross_model():
    # Llama 3.1 8B from eval_results.json
    with open("eval_results.json") as f:
        data = json.load(f)
    from drift_env.policy import drift_direction
    llama = {"tightening": [0, 0], "loosening": [0, 0]}
    for name, stats in data["summary"]["per_drift"].items():
        d = drift_direction(name)
        if d in llama:
            llama[d][0] += stats["correct"]
            llama[d][1] += stats["total"]

    qwen = {"tightening": (0, 23), "loosening": (3, 14)}  # from v6 logs

    labels = ["Tightening", "Loosening"]
    llama_acc = [llama[k.lower()][0] / llama[k.lower()][1] * 100 for k in labels]
    qwen_acc  = [qwen[k.lower()][0]  / qwen[k.lower()][1]  * 100 for k in labels]

    x = range(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.6))
    b1 = ax.bar([i - width/2 for i in x], llama_acc, width,
                label="Llama 3.1 8B (untrained, 8 episodes)", color="#7b3fbf")
    b2 = ax.bar([i + width/2 for i in x], qwen_acc, width,
                label="Qwen 2.5 3B (untrained, 200 samples)", color="#1f8a3b")

    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.6,
                    f"{b.get_height():.1f}%",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Drift-sensitive accuracy")
    ax.set_ylim(0, 60)
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_yticklabels([f"{v}%" for v in [0, 10, 20, 30, 40, 50]])
    ax.set_title("Leniency bias is direction-asymmetric across model families")
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(alpha=0.2, axis="y")

    fig.tight_layout()
    p = OUT / "cross_model_baseline.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    plot_sft_loss()
    plot_direction_split_v6()
    plot_cross_model()
    print("done")
