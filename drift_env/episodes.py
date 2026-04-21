"""Episode generator — deterministic given a seed.

Produces a 20-email sequence with 2 admin (drift) emails at positions (3, 11).
The policy timeline and per-step correct_action_hint are pre-computed so the
grader can look up ground truth in O(1).

Each customer step also records whether it is "drift-sensitive" — i.e. the
correct action under the current policy is different from what it would have
been BEFORE the most recent drift event. The grader uses this to award the
drift-attention bonus.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

from drift_env.emails import CUSTOMER_TEMPLATES, CustomerTemplate
from drift_env.models import Email, EmailKind
from drift_env.policy import DEFAULT_POLICY, DRIFT_EVENTS, DriftEvent, Policy


EPISODE_LENGTH = 20
DRIFT_POSITIONS = (3, 11)


@dataclass(frozen=True)
class EpisodeStep:
    email: Email
    policy_at_step: Policy
    correct_action_hint: dict
    # Drift sensitivity: which drift_event_name (if any) this step proves
    # awareness of. None = not drift-sensitive.
    drift_sensitive_to: Optional[str] = None


@dataclass(frozen=True)
class Episode:
    id: str
    steps: List[EpisodeStep]
    drift_timeline: List[tuple[int, str]] = field(default_factory=list)
    # e.g. [(3, "sla_2hr"), (11, "refund_cap_200")]


def _make_admin_email(event: DriftEvent, idx: int) -> Email:
    return Email(
        id=f"admin_{idx}_{event.name}",
        kind=EmailKind.ADMIN,
        subject=event.admin_subject,
        body=event.admin_body,
        sender="ops@company.com",
        meta={"drift_event": event.name},
    )


def _make_customer_email(t: CustomerTemplate, idx: int) -> Email:
    return Email(
        id=f"c_{idx}_{t.id}",
        kind=EmailKind.CUSTOMER,
        subject=t.subject,
        body=t.body,
        sender="customer@example.com",
        meta={
            "template_id": t.id,
            "kind": t.kind,
            "refund_amount": t.refund_amount,
            "severity": t.severity,
            "needs_info": t.needs_info,
        },
    )


def _correct_action_hint(template: CustomerTemplate, p: Policy) -> dict:
    """Pre-computed ground-truth correct action for (template, policy)."""
    if template.kind == "refund":
        if template.refund_amount is None:
            return {"action_type": "request_info", "info_field": "refund_amount"}
        if template.refund_amount <= p.refund_cap:
            return {"action_type": "approve_refund", "refund_amount": template.refund_amount}
        return {"action_type": "escalate", "escalation_tier": "manager"}

    if template.kind == "critical_incident":
        return {
            "action_type": "escalate",
            "escalation_tier": p.critical_escalation_tier,
            "followup_hours": p.sla_hours_critical,
        }

    if template.kind == "info_request":
        return {"action_type": "request_info", "info_field": template.needs_info}

    if template.kind == "billing_q":
        return {"action_type": "reply"}

    if template.kind == "chitchat":
        return {"action_type": "close", "resolution_code": "no_action_needed"}

    return {"action_type": "reply"}


def _hints_differ(a: dict, b: dict) -> bool:
    """Do two action hints prescribe a meaningfully different action?
    We ignore free-text fields (reply_text) and compare the structural fields.
    """
    keys = ("action_type", "refund_amount", "escalation_tier",
            "followup_hours", "resolution_code", "info_field")
    return any(a.get(k) != b.get(k) for k in keys)


def generate_episode(seed: int = 0, episode_id: str = "ep_0") -> Episode:
    rng = random.Random(seed)

    # Sample 2 drift events from different TYPES so they stack.
    drift_by_type = {
        "refund": ["refund_cap_50", "refund_cap_25", "refund_cap_200"],
        "escalate": ["escalate_manager", "escalate_tier_1", "escalate_keep_tier_2"],
        "sla": ["sla_2hr", "sla_4hr", "sla_48hr"],
    }
    types = rng.sample(list(drift_by_type.keys()), k=2)
    drift_names = [rng.choice(drift_by_type[t]) for t in types]
    drifts = [DRIFT_EVENTS[n] for n in drift_names]

    # Sample customer templates with a bias toward drift-sensitive kinds.
    drift_sensitive_pool = [t for t in CUSTOMER_TEMPLATES
                            if t.kind in ("refund", "critical_incident")]
    other_pool = [t for t in CUSTOMER_TEMPLATES
                  if t.kind not in ("refund", "critical_incident")]

    customer_count = EPISODE_LENGTH - len(drifts)
    picks: List[CustomerTemplate] = []
    for _ in range(customer_count):
        pool = drift_sensitive_pool if rng.random() < 0.7 else other_pool
        picks.append(rng.choice(pool))

    # Walk through episode positions, interleaving admin emails.
    steps: List[EpisodeStep] = []
    current_policy = DEFAULT_POLICY
    policy_before_last_drift: Optional[Policy] = None
    last_drift_name: Optional[str] = None

    drift_queue = list(zip(DRIFT_POSITIONS, drifts))
    cust_iter = iter(picks)
    timeline: List[tuple[int, str]] = []

    for idx in range(EPISODE_LENGTH):
        if drift_queue and drift_queue[0][0] == idx:
            _, event = drift_queue.pop(0)
            admin_email = _make_admin_email(event, idx)
            steps.append(EpisodeStep(
                email=admin_email,
                policy_at_step=current_policy,
                correct_action_hint={"action_type": "close",
                                     "resolution_code": "policy_acknowledged"},
                drift_sensitive_to=None,
            ))
            # AFTER processing this admin email, record prior policy + update.
            policy_before_last_drift = current_policy
            current_policy = event.apply(current_policy)
            last_drift_name = event.name
            timeline.append((idx, event.name))
        else:
            template = next(cust_iter)
            cust_email = _make_customer_email(template, idx)
            correct_now = _correct_action_hint(template, current_policy)

            # Drift sensitivity: does the answer change vs pre-drift policy?
            sensitive_to: Optional[str] = None
            if last_drift_name and policy_before_last_drift is not None:
                correct_pre = _correct_action_hint(template, policy_before_last_drift)
                if _hints_differ(correct_now, correct_pre):
                    sensitive_to = last_drift_name

            steps.append(EpisodeStep(
                email=cust_email,
                policy_at_step=current_policy,
                correct_action_hint=correct_now,
                drift_sensitive_to=sensitive_to,
            ))

    return Episode(id=episode_id, steps=steps, drift_timeline=timeline)
