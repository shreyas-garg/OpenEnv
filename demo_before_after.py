"""Side-by-side before/after demo on a single fixed episode.

Runs TWO models against the same episode seed and prints their actions at each
step, marking drift-sensitive turns. Produces the pitch's money-shot clip.

Usage (run inside the Colab / HF compute env after training finishes):

    python demo_before_after.py \\
        --base-model unsloth/Qwen2.5-0.5B-Instruct \\
        --trained-adapter ./outputs/lora_adapters \\
        --seed 42

The script can also render a Markdown table (--markdown) suitable for pasting
into a slide, or just print a terminal-colored side-by-side view.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional

import torch

from drift_env.environment import DriftEnv
from drift_env.episodes import generate_episode
from drift_env.models import ActionType
from drift_env.prompts import SYSTEM_PROMPT
from drift_env.training.rewards import parse_generated_action


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------
RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
RED = "\033[31m"; GREEN = "\033[32m"; YELLOW = "\033[33m"; BLUE = "\033[34m"


def _fmt_action(action) -> str:
    """Compact printable representation of an Action."""
    at = action.action_type.value
    parts = [at]
    for key in ("refund_amount", "escalation_tier", "followup_hours",
                "resolution_code", "info_field"):
        v = getattr(action, key, None)
        if v is not None:
            parts.append(f"{key}={v}")
    return "(" + ", ".join(parts) + ")"


def _fmt_email(email) -> str:
    tag = "ADMIN" if email.kind.value == "admin" else "cust"
    return f"[{tag}] {email.subject}"


@dataclass
class StepDecision:
    email_subject: str
    email_kind: str
    drift_sensitive_to: Optional[str]
    correct_action: dict
    before_action: object
    after_action: object
    before_correct: bool
    after_correct: bool


# ---------------------------------------------------------------------------
# Agent wrapper (loads a model + adapters and can generate one action per obs)
# ---------------------------------------------------------------------------
class LocalAgent:
    def __init__(self, base_model: str, adapter_path: Optional[str] = None,
                 max_new_tokens: int = 128):
        from unsloth import FastLanguageModel
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=4096,
            load_in_4bit=True,
        )
        if adapter_path and os.path.isdir(adapter_path):
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            print(f"[agent] loaded adapter from {adapter_path}")
        FastLanguageModel.for_inference(self.model)
        self.max_new_tokens = max_new_tokens

    def act(self, obs) -> object:
        from drift_env.prompts import render_user_prompt
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_user_prompt(obs)},
        ]
        inputs = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                inputs, max_new_tokens=self.max_new_tokens,
                do_sample=False, pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        text = self.tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        return parse_generated_action(text)


# ---------------------------------------------------------------------------
def run_one_episode(agent: LocalAgent, seed: int) -> list[object]:
    """Return the list of actions the agent took across an episode."""
    env = DriftEnv()
    obs = env.reset(seed=seed, episode_id=f"demo_{seed}")
    actions = []
    ep = generate_episode(seed=seed, episode_id=f"demo_{seed}")
    for _ in ep.steps:
        a = agent.act(obs)
        actions.append(a)
        res = env.step(a)
        if res.done:
            break
        if res.observation is not None:
            obs = res.observation
    return actions


def _check_compliance(action, hint: dict) -> bool:
    """Same as grader._compliance >= 1.0."""
    from drift_env.grader import _compliance
    return _compliance(action, hint) >= 1.0


def collect(before_actions, after_actions, seed: int) -> list[StepDecision]:
    ep = generate_episode(seed=seed, episode_id=f"demo_{seed}")
    rows = []
    for step, ba, aa in zip(ep.steps, before_actions, after_actions):
        rows.append(StepDecision(
            email_subject=step.email.subject,
            email_kind=step.email.kind.value,
            drift_sensitive_to=step.drift_sensitive_to,
            correct_action=step.correct_action_hint,
            before_action=ba,
            after_action=aa,
            before_correct=_check_compliance(ba, step.correct_action_hint),
            after_correct=_check_compliance(aa, step.correct_action_hint),
        ))
    return rows


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------
def render_terminal(rows: list[StepDecision]) -> None:
    print(f"\n{BOLD}Step  Email                                        Before               After{RESET}")
    print("-" * 120)
    for i, r in enumerate(rows):
        tag = f"{YELLOW}DRIFT-SENSITIVE{RESET}" if r.drift_sensitive_to else ""
        if r.email_kind == "admin":
            tag = f"{BLUE}ADMIN EMAIL{RESET}"
        subj = r.email_subject[:40].ljust(40)
        b_sym = f"{GREEN}✓{RESET}" if r.before_correct else f"{RED}✗{RESET}"
        a_sym = f"{GREEN}✓{RESET}" if r.after_correct else f"{RED}✗{RESET}"
        b_txt = _fmt_action(r.before_action)[:45]
        a_txt = _fmt_action(r.after_action)[:45]
        print(f"{i:>3}   {subj} {b_sym} {b_txt:<46} {a_sym} {a_txt}  {tag}")

    b_count = sum(r.before_correct for r in rows)
    a_count = sum(r.after_correct for r in rows)
    print()
    print(f"{BOLD}Before: {b_count}/{len(rows)} correct   After: {a_count}/{len(rows)} correct{RESET}")
    drift_rows = [r for r in rows if r.drift_sensitive_to]
    if drift_rows:
        db = sum(r.before_correct for r in drift_rows)
        da = sum(r.after_correct for r in drift_rows)
        print(f"{BOLD}Drift-sensitive: before {db}/{len(drift_rows)}   after {da}/{len(drift_rows)}{RESET}")


def render_markdown(rows: list[StepDecision], out_path: str) -> None:
    b_count = sum(r.before_correct for r in rows)
    a_count = sum(r.after_correct for r in rows)
    drift_rows = [r for r in rows if r.drift_sensitive_to]
    db = sum(r.before_correct for r in drift_rows) if drift_rows else 0
    da = sum(r.after_correct for r in drift_rows) if drift_rows else 0

    lines = ["# Before vs After — single episode\n",
             f"- Overall: **{b_count}/{len(rows)}** → **{a_count}/{len(rows)}**",
             f"- Drift-sensitive: **{db}/{len(drift_rows)}** → **{da}/{len(drift_rows)}**\n",
             "| # | Email | Drift? | Before | After |",
             "|---|-------|--------|--------|-------|"]
    for i, r in enumerate(rows):
        drift = f"**{r.drift_sensitive_to}**" if r.drift_sensitive_to else ("_admin_" if r.email_kind == "admin" else "-")
        b = ("✅ " if r.before_correct else "❌ ") + _fmt_action(r.before_action)
        a = ("✅ " if r.after_correct else "❌ ") + _fmt_action(r.after_action)
        subj = r.email_subject[:40]
        lines.append(f"| {i} | {subj} | {drift} | {b} | {a} |")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[ok] wrote {out_path}")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="unsloth/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--trained-adapter", default="./outputs/lora_adapters")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--markdown", type=str, default=None,
                    help="If given, also write a markdown table to this path")
    args = ap.parse_args()

    print(f"=== Episode seed={args.seed} ===")

    print("\n[1/2] Rolling out BEFORE (base model, no adapter)...")
    before_agent = LocalAgent(args.base_model, adapter_path=None)
    before_actions = run_one_episode(before_agent, args.seed)
    del before_agent
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n[2/2] Rolling out AFTER (base + trained adapter)...")
    after_agent = LocalAgent(args.base_model, adapter_path=args.trained_adapter)
    after_actions = run_one_episode(after_agent, args.seed)

    rows = collect(before_actions, after_actions, args.seed)
    render_terminal(rows)
    if args.markdown:
        render_markdown(rows, args.markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
