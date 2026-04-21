"""Deterministic 3-component grader.

Per-step reward in [0.0, 2.0] composed of:
  1. compliance      0.0 – 1.0  — did the action respect current policy?
  2. appropriateness 0.0 – 0.5  — was the action TYPE sensible for the email?
  3. drift_bonus     0.0 – 0.5  — first correct drift-aware action after a
                                 drift fires once per drift event.

No LLM-as-judge, no randomness. Ground truth is pre-computed in
`episodes.EpisodeStep.correct_action_hint`, so the grader is a pure function
of (action, correct_hint, email_meta, drift_sensitive_to, armed_drifts).
"""

from __future__ import annotations

from typing import Tuple

from drift_env.models import Action


# --------------------------------------------------------------------------
# Compliance — exact structural match on the fields that define "correct"
# --------------------------------------------------------------------------
def _compliance(action: Action, hint: dict) -> float:
    want = hint.get("action_type")
    got = action.action_type.value
    if got != want:
        return 0.0

    # Field-level exact match depending on action_type
    if want == "approve_refund":
        return 1.0 if action.refund_amount == hint.get("refund_amount") else 0.5
    if want == "escalate":
        tier_ok = action.escalation_tier == hint.get("escalation_tier")
        hours_want = hint.get("followup_hours")
        if hours_want is None:
            return 1.0 if tier_ok else 0.5
        hours_ok = action.followup_hours == hours_want
        if tier_ok and hours_ok:
            return 1.0
        if tier_ok or hours_ok:
            return 0.7
        return 0.4
    if want == "request_info":
        return 1.0 if action.info_field == hint.get("info_field") else 0.5
    if want == "close":
        return 1.0 if action.resolution_code == hint.get("resolution_code") else 0.7

    # Plain reply / anything else: action_type match is enough.
    return 1.0


# --------------------------------------------------------------------------
# Appropriateness — is the chosen action TYPE reasonable for this email kind?
#   Gives partial credit when the agent picks a sensible action type even if
#   specific fields are wrong. Caps at 0.5.
# --------------------------------------------------------------------------
_APPROPRIATE_BY_KIND = {
    "refund":             {"approve_refund", "escalate", "request_info"},
    "critical_incident":  {"escalate"},
    "info_request":       {"request_info"},
    "billing_q":          {"reply"},
    "chitchat":           {"close"},
}


def _appropriateness(action: Action, email_meta: dict) -> float:
    kind = email_meta.get("kind")
    if kind is None:
        return 0.0
    ok_set = _APPROPRIATE_BY_KIND.get(kind, set())
    if action.action_type.value in ok_set:
        return 0.5
    # Admin emails are handled by the env; this path is customer-only.
    return 0.0


# --------------------------------------------------------------------------
# Drift bonus — +0.5 the FIRST time the agent nails a drift-sensitive step
# after the corresponding drift has fired. Stateful across the episode,
# tracked by the environment in `armed_drifts`.
# --------------------------------------------------------------------------
def _drift_bonus(
    drift_sensitive_to: str | None,
    armed_drifts: set[str],
    compliance_score: float,
) -> Tuple[float, str | None]:
    """Returns (bonus, drift_to_clear_if_awarded)."""
    if drift_sensitive_to is None:
        return 0.0, None
    if drift_sensitive_to not in armed_drifts:
        return 0.0, None
    if compliance_score < 1.0:
        return 0.0, None
    return 0.5, drift_sensitive_to


# --------------------------------------------------------------------------
# Admin emails — handled as a separate scoring path (not drift-sensitive,
# grader doesn't compute appropriateness for them).
# --------------------------------------------------------------------------
def _grade_admin(action: Action) -> float:
    if action.action_type.value == "close":
        # Full credit regardless of resolution_code — we don't want to
        # punish harmless reword-ing of the ack.
        return 1.0
    # Any non-close action on an admin email is a minor error.
    return 0.2


# --------------------------------------------------------------------------
# Top-level: returns (total_reward, breakdown_dict, drift_to_clear)
# --------------------------------------------------------------------------
def grade_step(
    action: Action,
    hint: dict,
    email_meta: dict,
    drift_sensitive_to: str | None,
    armed_drifts: set[str],
    is_admin_email: bool,
) -> Tuple[float, dict, str | None]:
    if is_admin_email:
        reward = _grade_admin(action)
        return reward, {"compliance": reward, "appropriateness": 0.0,
                         "drift_bonus": 0.0}, None

    compliance = _compliance(action, hint)
    appropriateness = _appropriateness(action, email_meta)
    drift_bonus, drift_to_clear = _drift_bonus(
        drift_sensitive_to, armed_drifts, compliance,
    )
    total = compliance + appropriateness + drift_bonus
    breakdown = {
        "compliance": round(compliance, 4),
        "appropriateness": round(appropriateness, 4),
        "drift_bonus": round(drift_bonus, 4),
    }
    return round(total, 4), breakdown, drift_to_clear
