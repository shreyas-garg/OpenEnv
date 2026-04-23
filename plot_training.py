"""Plot training curves from the logs `train.py` saved.

Usage:
    python plot_training.py ./outputs

Expects these files (any subset — plotter skips what's missing):
    outputs/sft_log.json     <- trainer.state.log_history from SFT
    outputs/grpo_log.json    <- trainer.state.log_history from GRPO
    outputs/evals.json       <- {pre, post_sft, post_grpo} snapshots

Produces:
    outputs/reward_curve.png   <- GRPO reward + components over steps
    outputs/sft_loss.png       <- SFT loss curve
    outputs/drift_acc_bars.png <- pre / post-SFT / post-GRPO drift-sensitive accuracy
    outputs/summary.png        <- combined 1x3 figure suitable for a pitch slide
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def _load(path: str) -> Optional[object]:
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _extract_series(log: list[dict], key: str) -> tuple[list[int], list[float]]:
    """Pull a (step, value) time series from trainer.state.log_history."""
    xs, ys = [], []
    for entry in log:
        if key not in entry or "step" not in entry:
            continue
        try:
            ys.append(float(entry[key]))
            xs.append(int(entry["step"]))
        except (TypeError, ValueError):
            continue
    return xs, ys


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_sft_loss(log: list[dict], out_path: str) -> None:
    steps, losses = _extract_series(log, "loss")
    if not steps:
        print(f"[skip] no loss series in sft_log")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, losses, marker="o", markersize=3, linewidth=1.5, color="#2a6df4")
    ax.set_xlabel("SFT step")
    ax.set_ylabel("Loss")
    ax.set_title("SFT warm-up — loss over training steps")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[ok] wrote {out_path}")


def plot_grpo_reward_curve(log: list[dict], out_path: str) -> None:
    steps_r, total = _extract_series(log, "reward")
    _, comp = _extract_series(log, "rewards/reward_compliance/mean")
    _, appr = _extract_series(log, "rewards/reward_appropriateness/mean")
    _, bonus = _extract_series(log, "rewards/reward_drift_bonus/mean")

    if not steps_r:
        print(f"[skip] no reward series in grpo_log")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    # Total as a bold line; components as thinner stacked lines.
    if total:
        ax.plot(steps_r, total, label="total", linewidth=2.2, color="#111")
    if comp:
        ax.plot(steps_r[:len(comp)], comp, label="compliance",
                linewidth=1.5, color="#2a6df4")
    if appr:
        ax.plot(steps_r[:len(appr)], appr, label="appropriateness",
                linewidth=1.5, color="#f29e2e")
    if bonus:
        ax.plot(steps_r[:len(bonus)], bonus, label="drift_bonus",
                linewidth=1.5, color="#d5342a")

    ax.set_xlabel("GRPO step")
    ax.set_ylabel("Mean reward (per completion)")
    ax.set_title("GRPO — reward and components over training")
    ax.set_ylim(bottom=0)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[ok] wrote {out_path}")


def plot_drift_acc_bars(evals: dict, out_path: str) -> None:
    labels = ["pre", "post-SFT", "post-GRPO"]
    keys = ["pre", "post_sft", "post_grpo"]
    accs = []
    for k in keys:
        a = evals.get(k, {}).get("drift_acc")
        accs.append(a if isinstance(a, (int, float)) else 0.0)
    colors = ["#d5342a", "#f29e2e", "#2a6df4"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, [a * 100 for a in accs], color=colors, width=0.5)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                f"{a:.0%}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Drift-sensitive accuracy")
    ax.set_title(f"Drift-sensitive accuracy — {evals.get('model_name', 'model')}")
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels([f"{v}%" for v in [0, 20, 40, 60, 80, 100]])
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[ok] wrote {out_path}")


def plot_summary(sft_log: list[dict] | None, grpo_log: list[dict] | None,
                 evals: dict | None, out_path: str) -> None:
    """Combined 1x3 figure for the pitch slide."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    # Panel 1: SFT loss
    if sft_log:
        steps, losses = _extract_series(sft_log, "loss")
        if steps:
            axes[0].plot(steps, losses, marker="o", markersize=3, color="#2a6df4")
            axes[0].set_title("SFT loss")
            axes[0].set_xlabel("step"); axes[0].set_ylabel("loss")
            axes[0].grid(alpha=0.3)

    # Panel 2: GRPO reward curve
    if grpo_log:
        steps_r, total = _extract_series(grpo_log, "reward")
        _, comp = _extract_series(grpo_log, "rewards/reward_compliance/mean")
        _, appr = _extract_series(grpo_log, "rewards/reward_appropriateness/mean")
        _, bonus = _extract_series(grpo_log, "rewards/reward_drift_bonus/mean")
        if steps_r:
            axes[1].plot(steps_r, total, label="total", linewidth=2.2, color="#111")
            if comp: axes[1].plot(steps_r[:len(comp)], comp, label="comp", color="#2a6df4")
            if appr: axes[1].plot(steps_r[:len(appr)], appr, label="appr", color="#f29e2e")
            if bonus: axes[1].plot(steps_r[:len(bonus)], bonus, label="drift", color="#d5342a")
            axes[1].set_title("GRPO reward")
            axes[1].set_xlabel("step"); axes[1].set_ylabel("reward")
            axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    # Panel 3: drift acc bars
    if evals:
        labels = ["pre", "post-SFT", "post-GRPO"]
        keys = ["pre", "post_sft", "post_grpo"]
        accs = [evals.get(k, {}).get("drift_acc") or 0.0 for k in keys]
        colors = ["#d5342a", "#f29e2e", "#2a6df4"]
        bars = axes[2].bar(labels, [a * 100 for a in accs], color=colors, width=0.55)
        for b, a in zip(bars, accs):
            axes[2].text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                         f"{a:.0%}", ha="center", va="bottom",
                         fontsize=10, fontweight="bold")
        axes[2].set_ylim(0, 105)
        axes[2].set_title("Drift-sensitive accuracy")
        axes[2].grid(alpha=0.2, axis="y")

    fig.suptitle("Policy-Drift env — training run summary", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out_path}")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("outputs_dir", nargs="?", default="./outputs",
                    help="Directory containing sft_log.json, grpo_log.json, evals.json")
    args = ap.parse_args()

    d = args.outputs_dir
    sft_log = _load(os.path.join(d, "sft_log.json"))
    grpo_log = _load(os.path.join(d, "grpo_log.json"))
    evals = _load(os.path.join(d, "evals.json"))

    missing = [n for n, v in [("sft_log", sft_log), ("grpo_log", grpo_log), ("evals", evals)] if v is None]
    if missing:
        print(f"[warn] missing files (will skip corresponding plots): {missing}")

    if sft_log:
        plot_sft_loss(sft_log, os.path.join(d, "sft_loss.png"))
    if grpo_log:
        plot_grpo_reward_curve(grpo_log, os.path.join(d, "reward_curve.png"))
    if evals:
        plot_drift_acc_bars(evals, os.path.join(d, "drift_acc_bars.png"))

    plot_summary(sft_log, grpo_log, evals, os.path.join(d, "summary.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
