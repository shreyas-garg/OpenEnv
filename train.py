"""End-to-end training pipeline: SFT warm-up -> GRPO.

Designed to run top-to-bottom in Colab or on HF compute. Matches the organizer
guidance: verifier-based rewards, multiple independent reward components,
SFT warm-start before RL, and per-component logging.

Usage (Colab / HF Job):
    !pip install -q unsloth trl datasets wandb
    !python train.py

Toggle QUICK_MODE=True for a 5-min pipeline validation on Colab free tier.
Set to False on onsite HF compute for the real run.
"""

from __future__ import annotations

import os
import random
import sys
import warnings
from typing import Optional

import numpy as np
import torch

# Silence transformers generation-config spam during offline eval
warnings.filterwarnings("ignore", message="Both `max_new_tokens`.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Module-level globals; real values set in main() after GPU detection.
USE_BF16: bool = False
USE_FP16: bool = False

from drift_env.dataset import build_dataset, dataset_stats
from drift_env.prompts import SYSTEM_PROMPT
from drift_env.training.rewards import (
    reward_compliance,
    reward_appropriateness,
    reward_drift_bonus,
)

# ---------------------------------------------------------------------------
# Config (edit here or via env vars)
# ---------------------------------------------------------------------------
QUICK_MODE = os.getenv("QUICK_MODE", "true").lower() == "true"

# Model / hardware
MODEL_NAME = os.getenv("MODEL_NAME", "unsloth/Qwen2.5-0.5B-Instruct" if QUICK_MODE
                      else "unsloth/Qwen2.5-3B-Instruct")
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "4096"))
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "true").lower() == "true"

# Data
N_EPISODES_TRAIN = 50 if QUICK_MODE else 800
N_EPISODES_EVAL = 5 if QUICK_MODE else 40
SEED = 42

# SFT
SFT_EPOCHS = 1
SFT_BATCH = 2 if QUICK_MODE else 4
SFT_LR = 2e-4

# GRPO
GRPO_NUM_GEN = 4 if QUICK_MODE else 8         # K completions per prompt
GRPO_MAX_STEPS = 50 if QUICK_MODE else 600
GRPO_BATCH = 1 if QUICK_MODE else 2
GRPO_GRAD_ACCUM = 4
GRPO_LR = 5e-6
GRPO_MAX_COMPLETION = 128

# LoRA
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0

# Output
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "drift-env")
USE_WANDB = os.getenv("USE_WANDB", "false").lower() == "true"


def seed_all(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _precision_flags() -> tuple[bool, bool]:
    """Return (use_bf16, use_fp16). T4 (Turing, CC 7.5) doesn't support bf16 —
    only Ampere (CC 8.0) and later do. Fall back to fp16 on T4."""
    if not torch.cuda.is_available():
        return False, False
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        return True, False   # Ampere+ — use bf16
    return False, True       # T4 etc — use fp16


# ---------------------------------------------------------------------------
# 1. Build train/eval datasets from our environment
# ---------------------------------------------------------------------------
def build_hf_datasets():
    from datasets import Dataset
    train_rows = build_dataset(n_episodes=N_EPISODES_TRAIN, start_seed=0)
    eval_rows = build_dataset(n_episodes=N_EPISODES_EVAL, start_seed=10_000)

    print(f"Train: {dataset_stats(train_rows)}")
    print(f"Eval:  {dataset_stats(eval_rows)}")

    # SFT wants `prompt` + `completion` fields (chat template applied later)
    # GRPO wants `prompt` + the extra columns used in the reward functions.
    def to_chat(row, include_completion=False):
        out = {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["prompt"]},
            ],
            # Kept for TRL to forward into the reward funcs
            "correct_action_hint": row["correct_action_hint"],
            "email_kind": row["email_kind"] or "",
            "can_earn_drift_bonus": bool(row["can_earn_drift_bonus"]),
            "drift_sensitive_to": row["drift_sensitive_to"] or "",
            "is_admin_email": bool(row["is_admin_email"]),
        }
        if include_completion:
            out["completion"] = [{"role": "assistant", "content": row["correct_action_json"]}]
        return out

    sft_train = Dataset.from_list([to_chat(r, include_completion=True) for r in train_rows])
    grpo_train = Dataset.from_list([to_chat(r) for r in train_rows])
    grpo_eval = Dataset.from_list([to_chat(r) for r in eval_rows])
    return sft_train, grpo_train, grpo_eval


# ---------------------------------------------------------------------------
# 2. Load Qwen via Unsloth with LoRA adapters
# ---------------------------------------------------------------------------
def load_model_and_tokenizer():
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,                # auto
        load_in_4bit=LOAD_IN_4BIT,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# 3. SFT warm-up (1 epoch)
# ---------------------------------------------------------------------------
def run_sft(model, tokenizer, train_ds):
    from trl import SFTTrainer, SFTConfig
    args = SFTConfig(
        output_dir=f"{OUTPUT_DIR}/sft",
        num_train_epochs=SFT_EPOCHS,
        per_device_train_batch_size=SFT_BATCH,
        gradient_accumulation_steps=4,
        learning_rate=SFT_LR,
        logging_steps=10,
        save_strategy="no",
        bf16=USE_BF16,
        fp16=USE_FP16,
        report_to="wandb" if USE_WANDB else "none",
        seed=SEED,
        max_length=MAX_SEQ_LEN,
        # Cap tokenization parallelism — Unsloth defaults to num_proc=cpu_count()
        # which spawns 64+ workers on cloud GPUs and OOMs the container.
        dataset_num_proc=4,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        args=args,
    )
    print(f"\n=== SFT warm-up: {len(train_ds)} samples, {SFT_EPOCHS} epoch(s) ===")
    trainer.train()
    # Save structured log history for plotting.
    import json as _json
    with open(f"{OUTPUT_DIR}/sft_log.json", "w") as f:
        _json.dump(trainer.state.log_history, f, indent=2)
    print(f"SFT log saved -> {OUTPUT_DIR}/sft_log.json")
    return trainer.model


# ---------------------------------------------------------------------------
# 4. GRPO training
# ---------------------------------------------------------------------------
def run_grpo(model, tokenizer, train_ds, eval_ds):
    from trl import GRPOTrainer, GRPOConfig
    args = GRPOConfig(
        output_dir=f"{OUTPUT_DIR}/grpo",
        num_generations=GRPO_NUM_GEN,
        max_completion_length=GRPO_MAX_COMPLETION,
        per_device_train_batch_size=GRPO_BATCH,
        gradient_accumulation_steps=GRPO_GRAD_ACCUM,
        learning_rate=GRPO_LR,
        max_steps=GRPO_MAX_STEPS,
        logging_steps=5,
        save_strategy="no",
        bf16=USE_BF16,
        fp16=USE_FP16,
        report_to="wandb" if USE_WANDB else "none",
        seed=SEED,
        # Keep rollouts fast for Colab:
        temperature=0.7,
        top_p=0.9,
        # Cap tokenization parallelism (same OOM defense as SFT).
        dataset_num_proc=4,
    )
    print(f"\n=== GRPO: {len(train_ds)} prompts, max_steps={GRPO_MAX_STEPS}, K={GRPO_NUM_GEN} ===")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_compliance,
            reward_appropriateness,
            reward_drift_bonus,
        ],
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )
    trainer.train()
    # Save structured log history for plotting.
    import json as _json
    with open(f"{OUTPUT_DIR}/grpo_log.json", "w") as f:
        _json.dump(trainer.state.log_history, f, indent=2)
    print(f"GRPO log saved -> {OUTPUT_DIR}/grpo_log.json")
    return trainer.model


# ---------------------------------------------------------------------------
# 5. Offline eval (drift-sensitive accuracy before vs after)
# ---------------------------------------------------------------------------
def offline_eval(model, tokenizer, eval_ds, label: str, max_rows: int = 200):
    """Greedy-decode each eval prompt, score with total_reward, print summary.

    Logs drift-sensitive accuracy broken down by drift direction
    (tightening / loosening / neutral). The *tightening* number is the one
    that actually matters for the pitch — it measures the leniency bias
    that our training specifically aims to remove. Loosening accuracy is
    often high even pre-training because the looser rule matches the base
    model's internet prior.
    """
    from drift_env.training.rewards import parse_generated_action, total_reward
    from drift_env.policy import drift_direction
    model.eval()

    comp_total = appr_total = bonus_total = 0.0
    drift_total = drift_correct = 0
    per_dir = {"tightening": [0, 0], "loosening": [0, 0], "neutral": [0, 0]}  # [correct, total]
    n = min(len(eval_ds), max_rows)
    for i in range(n):
        row = eval_ds[i]
        chat = row["prompt"]
        inputs = tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                inputs,
                max_new_tokens=GRPO_MAX_COMPLETION,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        text = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        r = total_reward(
            completion=text,
            correct_action_hint=row["correct_action_hint"],
            email_kind=row["email_kind"] or None,
            can_earn_drift_bonus=row["can_earn_drift_bonus"],
            drift_sensitive_to=row["drift_sensitive_to"] or None,
            is_admin_email=row["is_admin_email"],
        )
        comp_total += r["compliance"]
        appr_total += r["appropriateness"]
        bonus_total += r["drift_bonus"]
        # per-direction tracking: a row counts only if it is drift-sensitive
        # (i.e. the correct answer is different from what it'd be pre-drift)
        drift_to = row["drift_sensitive_to"] or None
        if drift_to:
            direction = drift_direction(drift_to)
            if direction is not None:
                per_dir[direction][1] += 1
                if r["compliance"] >= 1.0:
                    per_dir[direction][0] += 1
        if row["can_earn_drift_bonus"]:
            drift_total += 1
            if r["compliance"] >= 1.0:
                drift_correct += 1

    drift_acc = drift_correct / drift_total if drift_total else None
    def _acc(pair):
        c, t = pair
        return (c / t) if t else None
    per_dir_acc = {k: _acc(v) for k, v in per_dir.items()}

    print(f"\n=== Offline eval [{label}] over {n} samples ===")
    print(f"  compliance avg     : {comp_total / n:.3f} / 1.0")
    print(f"  appropriateness avg: {appr_total / n:.3f} / 0.5")
    print(f"  drift_bonus avg    : {bonus_total / n:.3f} / 0.5")
    print(f"  total avg          : {(comp_total + appr_total + bonus_total) / n:.3f} / 2.0")
    if drift_acc is not None:
        print(f"  drift-sens acc     : {drift_acc:.1%}  ({drift_correct}/{drift_total})")
    for direction in ("tightening", "loosening", "neutral"):
        c, t = per_dir[direction]
        acc_str = f"{c/t:.1%}" if t else "n/a"
        print(f"    {direction:<11}: {acc_str}  ({c}/{t})")

    return {
        "compliance": comp_total / n,
        "appropriateness": appr_total / n,
        "drift_bonus": bonus_total / n,
        "drift_acc": drift_acc,
        "drift_acc_by_direction": per_dir_acc,
        "drift_counts_by_direction": {k: {"correct": v[0], "total": v[1]} for k, v in per_dir.items()},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    seed_all(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    global USE_BF16, USE_FP16
    USE_BF16, USE_FP16 = _precision_flags()

    print(f"QUICK_MODE={QUICK_MODE}  MODEL={MODEL_NAME}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print(f"GPU: {name}  capability={cap}  bf16={USE_BF16}  fp16={USE_FP16}")
    else:
        print("CUDA not available — training will be extremely slow")

    sft_train, grpo_train, grpo_eval = build_hf_datasets()
    model, tokenizer = load_model_and_tokenizer()

    # Pre-training offline eval — the "before" number for the pitch clip.
    pre = offline_eval(model, tokenizer, grpo_eval, label="pre-training")

    # SFT warm-start.
    model = run_sft(model, tokenizer, sft_train)
    post_sft = offline_eval(model, tokenizer, grpo_eval, label="post-SFT")

    # GRPO.
    model = run_grpo(model, tokenizer, grpo_train, grpo_eval)
    post_grpo = offline_eval(model, tokenizer, grpo_eval, label="post-GRPO")

    print("\n=== Improvement summary ===")
    for k in ("compliance", "appropriateness", "drift_bonus"):
        print(f"  {k:<20} {pre[k]:.3f}  ->  {post_sft[k]:.3f}  ->  {post_grpo[k]:.3f}")
    print(f"  drift-sens acc (all) {_fmt_acc(pre['drift_acc'])} -> "
          f"{_fmt_acc(post_sft['drift_acc'])} -> {_fmt_acc(post_grpo['drift_acc'])}")
    for direction in ("tightening", "loosening", "neutral"):
        pre_a = pre["drift_acc_by_direction"].get(direction)
        sft_a = post_sft["drift_acc_by_direction"].get(direction)
        grpo_a = post_grpo["drift_acc_by_direction"].get(direction)
        print(f"    {direction:<12}     {_fmt_acc(pre_a)} -> {_fmt_acc(sft_a)} -> {_fmt_acc(grpo_a)}")

    # Save eval snapshots for the plotting script.
    import json as _json
    with open(f"{OUTPUT_DIR}/evals.json", "w") as f:
        _json.dump({
            "pre": pre, "post_sft": post_sft, "post_grpo": post_grpo,
            "model_name": MODEL_NAME, "quick_mode": QUICK_MODE,
        }, f, indent=2, default=str)
    print(f"Eval snapshots saved -> {OUTPUT_DIR}/evals.json")

    # Save LoRA adapters ONLY (organizer warning: do not naively merge 4-bit).
    adapter_path = f"{OUTPUT_DIR}/lora_adapters"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\nLoRA adapters saved to {adapter_path}")

    # Push the entire outputs/ directory to HF Hub so it survives the
    # ephemeral Jobs container. Set HUB_REPO_ID to enable.
    hub_repo = os.getenv("HUB_REPO_ID")
    if hub_repo:
        try:
            from huggingface_hub import HfApi, create_repo
            create_repo(hub_repo, repo_type="model", exist_ok=True, private=True,
                        token=os.getenv("HF_TOKEN"))
            HfApi().upload_folder(
                folder_path=OUTPUT_DIR,
                repo_id=hub_repo,
                repo_type="model",
                token=os.getenv("HF_TOKEN"),
                commit_message="Onsite training outputs",
            )
            print(f"\nOutputs pushed to https://huggingface.co/{hub_repo}")
        except Exception as e:
            print(f"\n[warn] failed to push outputs to Hub: {e}")
    else:
        print("\n[note] HUB_REPO_ID not set; outputs only exist inside this container.")

    return 0


def _fmt_acc(a: Optional[float]) -> str:
    return f"{a:.1%}" if a is not None else "n/a"


if __name__ == "__main__":
    sys.exit(main())
