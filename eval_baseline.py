"""Pre-training evaluation harness.

Runs a base LLM agent against N episodes of DriftEnv and reports:
  - mean episode reward
  - mean fraction of max (max = 30.0 per episode)
  - drift-sensitive-step accuracy  <-- the KEY metric
  - breakdown by drift event
  - sample trajectories

The whole point is to confirm the base model's drift-sensitive-step accuracy
sits in the 20%-40% training-headroom zone. Above 60% means our task is too
easy and training won't show improvement. Below 10% means our task is too
hard for any RL within the time budget.

Usage:
    API_BASE_URL=https://api.groq.com/openai/v1 \\
    HF_TOKEN=gsk_... \\
    MODEL_NAME=llama-3.1-8b-instant \\
    PYTHONPATH=. python3 eval_baseline.py --episodes 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any

from dotenv import load_dotenv

from drift_env.environment import DriftEnv
from drift_env.llm_agent import LLMAgent
from drift_env.episodes import generate_episode

load_dotenv()


def run_one_episode(
    agent: LLMAgent, seed: int, verbose: bool = False,
) -> dict[str, Any]:
    env = DriftEnv()
    obs = env.reset(seed=seed, episode_id=f"eval_{seed}")
    ep_plan = generate_episode(seed=seed, episode_id=f"eval_{seed}")

    total_reward = 0.0
    breakdown_totals = defaultdict(float)
    drift_sensitive_total = 0
    drift_sensitive_correct = 0
    per_drift: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0}
    )
    trajectory = []

    for i, step_plan in enumerate(ep_plan.steps):
        action, raw = agent.act(obs)
        result = env.step(action)

        total_reward += result.reward
        for k, v in result.info["breakdown"].items():
            breakdown_totals[k] += v

        sensitive_to = step_plan.drift_sensitive_to
        is_correct = result.info["breakdown"]["compliance"] >= 1.0
        if sensitive_to is not None:
            drift_sensitive_total += 1
            per_drift[sensitive_to]["total"] += 1
            if is_correct:
                drift_sensitive_correct += 1
                per_drift[sensitive_to]["correct"] += 1

        trajectory.append({
            "step": i,
            "email_kind": step_plan.email.kind.value,
            "email_id": step_plan.email.id,
            "action": action.model_dump(exclude_none=True),
            "correct_hint": step_plan.correct_action_hint,
            "drift_sensitive_to": sensitive_to,
            "reward": result.reward,
            "compliance": result.info["breakdown"]["compliance"],
        })

        if verbose:
            sens = f" [SENS→{sensitive_to}]" if sensitive_to else ""
            print(f"  step {i:>2} [{step_plan.email.kind.value:<8}] "
                  f"reward={result.reward:.2f} comp={result.info['breakdown']['compliance']:.2f}{sens}")

        if result.done:
            break
        if result.observation is not None:
            obs = result.observation

    return {
        "seed": seed,
        "total_reward": round(total_reward, 4),
        "max_reward": 30.0,
        "frac_of_max": round(total_reward / 30.0, 4),
        "breakdown_totals": {k: round(v, 4) for k, v in breakdown_totals.items()},
        "drift_sensitive_total": drift_sensitive_total,
        "drift_sensitive_correct": drift_sensitive_correct,
        "drift_sensitive_acc": (
            round(drift_sensitive_correct / drift_sensitive_total, 4)
            if drift_sensitive_total else None
        ),
        "per_drift": {k: dict(v) for k, v in per_drift.items()},
        "trajectory": trajectory,
    }


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    mean_reward = sum(r["total_reward"] for r in results) / n
    mean_frac = sum(r["frac_of_max"] for r in results) / n
    dst = sum(r["drift_sensitive_total"] for r in results)
    dsc = sum(r["drift_sensitive_correct"] for r in results)

    per_drift_agg: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0}
    )
    for r in results:
        for name, stats in r["per_drift"].items():
            per_drift_agg[name]["total"] += stats["total"]
            per_drift_agg[name]["correct"] += stats["correct"]

    return {
        "episodes": n,
        "mean_reward": round(mean_reward, 4),
        "mean_frac_of_max": round(mean_frac, 4),
        "drift_sensitive_total": dst,
        "drift_sensitive_correct": dsc,
        "drift_sensitive_acc": round(dsc / dst, 4) if dst else None,
        "per_drift": {
            k: {
                **v,
                "acc": round(v["correct"] / v["total"], 4) if v["total"] else None,
            }
            for k, v in per_drift_agg.items()
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--start-seed", type=int, default=100)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--save", type=str, default="eval_results.json")
    args = ap.parse_args()

    api_base = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    model = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

    if not api_key:
        print("ERROR: set HF_TOKEN (or API_KEY) env var.", file=sys.stderr)
        return 1

    print(f"Model: {model}")
    print(f"Base URL: {api_base}")
    print(f"Episodes: {args.episodes}")
    print("-" * 60)

    agent = LLMAgent(api_key=api_key, base_url=api_base, model=model)

    results = []
    t0 = time.time()
    for i in range(args.episodes):
        seed = args.start_seed + i
        if args.verbose:
            print(f"\n=== Episode seed={seed} ===")
        try:
            r = run_one_episode(agent, seed=seed, verbose=args.verbose)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        results.append(r)
        dsa = r["drift_sensitive_acc"]
        dsa_str = f"{dsa:.2%}" if dsa is not None else "n/a"
        print(f"  seed={seed} reward={r['total_reward']:.2f}/30 "
              f"drift_acc={dsa_str} "
              f"({r['drift_sensitive_correct']}/{r['drift_sensitive_total']})")

    dt = time.time() - t0
    summary = summarise(results)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print(f"\nTook {dt:.1f}s for {args.episodes} episodes.")

    # Interpretation
    dsa = summary["drift_sensitive_acc"]
    if dsa is None:
        print("\n[warn] no drift-sensitive steps encountered.")
    elif dsa < 0.10:
        print(f"\n[warn] drift-sensitive acc = {dsa:.0%}. Too hard for RL — redesign.")
    elif dsa > 0.60:
        print(f"\n[warn] drift-sensitive acc = {dsa:.0%}. Too easy — base model already solves it.")
    elif dsa > 0.40:
        print(f"\n[ok-ish] drift-sensitive acc = {dsa:.0%}. In the training zone but on the easy side.")
    else:
        print(f"\n[OK] drift-sensitive acc = {dsa:.0%}. In the sweet spot for training.")

    with open(args.save, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nFull results written to {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
