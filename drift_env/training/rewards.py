"""TRL-compatible reward functions.

Exposes each component of the 3-component grader as an independent
reward_func so TRL/wandb can plot them separately during training.
Per the organizers' guidance: "use multiple independent reward functions"
and "monitor individual reward components, not just the total."

TRL calls each reward_func with signature:
    func(completions: list[str], **kwargs) -> list[float]

where kwargs contains the dataset columns (prompt, correct_action_hint,
email_kind, can_earn_drift_bonus, is_admin_email).
"""

from __future__ import annotations

import json
from typing import Any

from drift_env.grader import _compliance, _appropriateness, _drift_bonus, _grade_admin
from drift_env.models import Action, ActionType


FALLBACK = Action(action_type=ActionType.CLOSE, resolution_code="error_fallback")


def parse_generated_action(raw: str) -> Action:
    text = raw.strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return FALLBACK
        text = text[start:end + 1]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return FALLBACK
    a_type = obj.get("action_type")
    if a_type not in {e.value for e in ActionType}:
        return FALLBACK
    try:
        return Action(
            action_type=ActionType(a_type),
            refund_amount=obj.get("refund_amount"),
            escalation_tier=obj.get("escalation_tier"),
            followup_hours=obj.get("followup_hours"),
            resolution_code=obj.get("resolution_code"),
            info_field=obj.get("info_field"),
            reply_text=obj.get("reply_text"),
        )
    except Exception:
        return FALLBACK


def _as_list(v, n):
    """Utility to lift scalar dataset columns into per-completion lists."""
    if isinstance(v, list):
        return v
    return [v] * n


def _per_sample(
    completions: list[str],
    correct_action_hint: Any,
    email_kind: Any,
    can_earn_drift_bonus: Any,
    drift_sensitive_to: Any,
    is_admin_email: Any,
):
    """Generator yielding (action, hint, email_kind, can_earn, sensitive_to, is_admin)
    for each completion. Handles TRL's list-or-scalar column conventions."""
    n = len(completions)
    hints = _as_list(correct_action_hint, n)
    kinds = _as_list(email_kind, n)
    earns = _as_list(can_earn_drift_bonus, n)
    sens = _as_list(drift_sensitive_to, n)
    admins = _as_list(is_admin_email, n)
    for comp, hint, kind, earn, s, admin in zip(completions, hints, kinds, earns, sens, admins):
        action = parse_generated_action(
            comp[0]["content"] if isinstance(comp, list) else comp
        )
        yield action, hint, kind, earn, s, admin


# ---------------------------------------------------------------------------
# Reward components — each exposed to TRL as a separate reward_func
# ---------------------------------------------------------------------------


def reward_compliance(
    completions: list[str],
    correct_action_hint=None,
    email_kind=None,
    can_earn_drift_bonus=None,
    drift_sensitive_to=None,
    is_admin_email=None,
    **_,
) -> list[float]:
    out = []
    for action, hint, kind, earn, s, admin in _per_sample(
        completions, correct_action_hint, email_kind,
        can_earn_drift_bonus, drift_sensitive_to, is_admin_email,
    ):
        if admin:
            out.append(_grade_admin(action))
        else:
            out.append(_compliance(action, hint))
    return out


def reward_appropriateness(
    completions: list[str],
    correct_action_hint=None,
    email_kind=None,
    can_earn_drift_bonus=None,
    drift_sensitive_to=None,
    is_admin_email=None,
    **_,
) -> list[float]:
    out = []
    for action, hint, kind, earn, s, admin in _per_sample(
        completions, correct_action_hint, email_kind,
        can_earn_drift_bonus, drift_sensitive_to, is_admin_email,
    ):
        if admin:
            out.append(0.0)
        else:
            out.append(_appropriateness(action, {"kind": kind}))
    return out


def reward_drift_bonus(
    completions: list[str],
    correct_action_hint=None,
    email_kind=None,
    can_earn_drift_bonus=None,
    drift_sensitive_to=None,
    is_admin_email=None,
    **_,
) -> list[float]:
    """Only fires on rows flagged `can_earn_drift_bonus=True` AND where the
    model's action is policy-compliant (compliance >= 1.0). This matches the
    env's armed-drift semantics: first correct drift-aware action earns +0.5."""
    out = []
    for action, hint, kind, earn, s, admin in _per_sample(
        completions, correct_action_hint, email_kind,
        can_earn_drift_bonus, drift_sensitive_to, is_admin_email,
    ):
        if admin or not earn:
            out.append(0.0)
            continue
        comp = _compliance(action, hint)
        bonus, _ = _drift_bonus(s, {s} if s else set(), comp)
        out.append(bonus)
    return out


# Handy wrapper for quick total-reward sanity checks outside TRL
def total_reward(
    completion: str,
    correct_action_hint: dict,
    email_kind: str | None,
    can_earn_drift_bonus: bool,
    drift_sensitive_to: str | None,
    is_admin_email: bool,
) -> dict:
    action = parse_generated_action(completion)
    if is_admin_email:
        comp = _grade_admin(action)
        appr = 0.0
    else:
        comp = _compliance(action, correct_action_hint)
        appr = _appropriateness(action, {"kind": email_kind})
    bonus = 0.0
    if not is_admin_email and can_earn_drift_bonus:
        b, _ = _drift_bonus(drift_sensitive_to, {drift_sensitive_to} if drift_sensitive_to else set(), comp)
        bonus = b
    return {
        "compliance": round(comp, 4),
        "appropriateness": round(appr, 4),
        "drift_bonus": round(bonus, 4),
        "total": round(comp + appr + bonus, 4),
    }
