"""Dataset generator: episodes -> per-step training rows.

Each row contains:
  - `prompt`         : the user prompt the model would see at that step
  - `correct_action` : JSON string of the ground-truth correct action (SFT target)
  - `email_kind`     : for appropriateness scoring in the reward function
  - `drift_sensitive_to` : drift_event_name this step tests, or None
  - `can_earn_drift_bonus`: True only for the FIRST drift-sensitive step after
                            each drift event (matches env's armed-drift state)
  - `episode_id` / `step_index` : traceability

History is built using the GROUND-TRUTH actions (teacher-forced). This matches
the inference distribution after the agent has been trained — we want the model
to see clean histories, not noise.

Usage:
  from drift_env.dataset import build_dataset
  rows = build_dataset(n_episodes=500, start_seed=0)
  # -> list[dict]  OR  datasets.Dataset if `as_hf=True`
"""

from __future__ import annotations

import json
from typing import List

from drift_env.emails import CUSTOMER_TEMPLATES
from drift_env.episodes import Episode, EpisodeStep, generate_episode
from drift_env.models import Email, EmailKind, Observation
from drift_env.prompts import render_user_prompt


def _ground_truth_action_json(hint: dict) -> str:
    """Serialize the correct action hint as the canonical JSON the model should emit."""
    out = {"action_type": hint["action_type"]}
    for key in ("refund_amount", "escalation_tier", "followup_hours",
                "resolution_code", "info_field"):
        v = hint.get(key)
        if v is not None:
            out[key] = v
    return json.dumps(out, separators=(",", ": "))


def _summary_for_history(step: EpisodeStep) -> dict:
    """Build the inbox-history entry as if the correct action had been taken."""
    email = step.email
    return {
        "email_id": email.id,
        "kind": email.kind.value,
        "subject": email.subject,
        "body": email.body,
        "sender": email.sender,
        "action_taken": step.correct_action_hint["action_type"],
    }


def _mark_first_bonus_steps(steps: List[EpisodeStep]) -> List[bool]:
    """For each step, return True iff it is the FIRST drift-sensitive step
    (post-drift) that can earn the drift-attention bonus for its drift event.
    """
    seen_drifts: set[str] = set()
    flags = []
    for s in steps:
        earn = False
        if s.drift_sensitive_to is not None and s.drift_sensitive_to not in seen_drifts:
            earn = True
            seen_drifts.add(s.drift_sensitive_to)
        flags.append(earn)
    return flags


def _observation_from_step(
    step: EpisodeStep, history: list[dict], index: int, total: int,
) -> Observation:
    """Build an Observation as the agent would see it (no grader metadata)."""
    clean_email = Email(
        id=step.email.id, kind=step.email.kind, subject=step.email.subject,
        body=step.email.body, sender=step.email.sender, meta={},
    )
    return Observation(
        current_email=clean_email,
        email_index=index,
        total_emails=total,
        inbox_history=list(history),
    )


def episode_to_rows(ep: Episode) -> List[dict]:
    """Convert one episode to a list of per-step training rows."""
    bonus_flags = _mark_first_bonus_steps(ep.steps)
    rows: List[dict] = []
    history: list[dict] = []
    total = len(ep.steps)

    for i, step in enumerate(ep.steps):
        obs = _observation_from_step(step, history, i, total)
        prompt = render_user_prompt(obs)
        row = {
            "episode_id": ep.id,
            "step_index": i,
            "prompt": prompt,
            "correct_action_json": _ground_truth_action_json(step.correct_action_hint),
            "correct_action_hint": step.correct_action_hint,
            "email_kind": step.email.meta.get("kind"),
            "is_admin_email": step.email.kind == EmailKind.ADMIN,
            "drift_sensitive_to": step.drift_sensitive_to,
            "can_earn_drift_bonus": bonus_flags[i],
            "policy_refund_cap": step.policy_at_step.refund_cap,
            "policy_escalation_tier": step.policy_at_step.critical_escalation_tier,
            "policy_sla_hours": step.policy_at_step.sla_hours_critical,
        }
        rows.append(row)

        # Teacher-force: append what the CORRECT action would have been to history
        history.append(_summary_for_history(step))

    return rows


def build_dataset(n_episodes: int, start_seed: int = 0) -> List[dict]:
    """Generate a list of training rows from `n_episodes` episodes."""
    all_rows: List[dict] = []
    for i in range(n_episodes):
        seed = start_seed + i
        ep = generate_episode(seed=seed, episode_id=f"train_{seed}")
        all_rows.extend(episode_to_rows(ep))
    return all_rows


def dataset_stats(rows: List[dict]) -> dict:
    """Quick sanity-check numbers."""
    n = len(rows)
    admin = sum(1 for r in rows if r["is_admin_email"])
    drift_sens = sum(1 for r in rows if r["drift_sensitive_to"])
    bonus_eligible = sum(1 for r in rows if r["can_earn_drift_bonus"])
    kinds: dict[str, int] = {}
    for r in rows:
        k = r["email_kind"] or "admin"
        kinds[k] = kinds.get(k, 0) + 1
    return {
        "n_rows": n,
        "admin_rows": admin,
        "customer_rows": n - admin,
        "drift_sensitive_rows": drift_sens,
        "bonus_eligible_rows": bonus_eligible,
        "kinds_distribution": kinds,
    }
