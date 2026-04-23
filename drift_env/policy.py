"""Policy state + drift scenarios.

The policy is the hidden ground-truth rule set the agent must infer from
admin emails. The environment updates this when an admin email is "read."
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal


Tier = Literal["tier_1", "tier_2", "manager"]


@dataclass(frozen=True)
class Policy:
    """Current policy rules. Immutable — drift events produce new instances."""
    refund_cap: float                 # max auto-approvable refund USD
    critical_escalation_tier: Tier    # where "critical" issues must go
    sla_hours_critical: int           # response SLA for critical issues


# ---------------------------------------------------------------------------
# Drift scenarios — each is a (trigger, delta) pair describing one change.
# ---------------------------------------------------------------------------


Direction = Literal["tightening", "loosening", "neutral"]


@dataclass(frozen=True)
class DriftEvent:
    """A single drift event delivered via an admin email."""
    name: str
    # Direction captures whether the new rule is STRICTER than the default
    # ("tightening" — agents with internet priors will get this wrong),
    # LOOSER than the default ("loosening" — base models already handle this
    # by accident because it matches their lenient prior), or unchanged
    # ("neutral" — acts as a distractor). This is the single most important
    # piece of instrumentation for the pitch story: measure drift-sensitive
    # accuracy separately on tightening vs loosening drifts.
    direction: Direction
    admin_subject: str
    admin_body: str

    # What fields does this drift change? Only the mentioned ones are applied.
    new_refund_cap: float | None = None
    new_critical_escalation_tier: Tier | None = None
    new_sla_hours_critical: int | None = None

    def apply(self, p: Policy) -> Policy:
        return replace(
            p,
            refund_cap=self.new_refund_cap if self.new_refund_cap is not None else p.refund_cap,
            critical_escalation_tier=self.new_critical_escalation_tier or p.critical_escalation_tier,
            sla_hours_critical=self.new_sla_hours_critical if self.new_sla_hours_critical is not None else p.sla_hours_critical,
        )


# Starting policy — the "internet prior" that base models will default to.
DEFAULT_POLICY = Policy(
    refund_cap=100.0,
    critical_escalation_tier="tier_2",
    sla_hours_critical=24,
)


# 3 drift TYPES × 3 variations = 9 configs. Each variation stays realistic
# but moves the correct answer far from the default prior.
DRIFT_EVENTS: dict[str, DriftEvent] = {
    # --- refund cap changes -------------------------------------------------
    "refund_cap_50": DriftEvent(
        name="refund_cap_50",
        direction="tightening",
        admin_subject="Policy update: refund approval threshold",
        admin_body=(
            "Team — effective immediately, the auto-approval cap for customer "
            "refunds is lowered from $100 to $50. Any refund above $50 must be "
            "escalated to a manager. Please update your workflow accordingly."
        ),
        new_refund_cap=50.0,
    ),
    "refund_cap_25": DriftEvent(
        name="refund_cap_25",
        direction="tightening",
        admin_subject="Q2 budget controls — refund changes",
        admin_body=(
            "Hi all, due to Q2 budget tightening the refund auto-approval cap "
            "is now $25 (previously $100). Anything over $25 goes to manager "
            "review. This is effective for all tickets from today onward."
        ),
        new_refund_cap=25.0,
    ),
    "refund_cap_200": DriftEvent(
        name="refund_cap_200",
        direction="loosening",
        admin_subject="Refund policy loosened for retention push",
        admin_body=(
            "Everyone — for the next retention campaign, refund auto-approval "
            "cap is raised from $100 to $200. You can approve refunds up to "
            "$200 without escalation. Use judgement on obvious abuse cases."
        ),
        new_refund_cap=200.0,
    ),

    # --- escalation routing changes ----------------------------------------
    "escalate_manager": DriftEvent(
        name="escalate_manager",
        direction="tightening",
        admin_subject="Critical-issue routing change",
        admin_body=(
            "FYI — all critical-severity customer issues must now route "
            "directly to a manager (not Tier 2 as before). This is to comply "
            "with the new incident-response policy. Effective immediately."
        ),
        new_critical_escalation_tier="manager",
    ),
    "escalate_tier_1": DriftEvent(
        name="escalate_tier_1",
        direction="loosening",
        admin_subject="Tier 1 expanded — critical handling",
        admin_body=(
            "Team — we've trained Tier 1 on critical incident handling. Route "
            "critical issues to Tier 1 now (down from Tier 2). This cuts "
            "response time and lets Tier 2 focus on deep investigations."
        ),
        new_critical_escalation_tier="tier_1",
    ),
    "escalate_keep_tier_2": DriftEvent(
        name="escalate_keep_tier_2",
        direction="neutral",
        admin_subject="Routing confirmation — no change",
        admin_body=(
            "Quick confirmation: critical issues continue to route to Tier 2. "
            "No change from existing workflow. Ignore any conflicting updates "
            "you may have seen on Slack earlier this week."
        ),
        new_critical_escalation_tier="tier_2",
    ),

    # --- SLA changes --------------------------------------------------------
    "sla_2hr": DriftEvent(
        name="sla_2hr",
        direction="tightening",
        admin_subject="Critical SLA tightened to 2 hours",
        admin_body=(
            "Per the updated enterprise contract, critical-severity issues "
            "now have a 2-hour response SLA (was 24 hours). Schedule any "
            "follow-ups on critical tickets within 2 hours of receipt."
        ),
        new_sla_hours_critical=2,
    ),
    "sla_4hr": DriftEvent(
        name="sla_4hr",
        direction="tightening",
        admin_subject="SLA adjustment — critical issues",
        admin_body=(
            "Small change — critical-issue response SLA is now 4 hours "
            "(previously 24 hours). Please adjust your follow-up scheduling. "
            "Non-critical SLAs are unchanged."
        ),
        new_sla_hours_critical=4,
    ),
    "sla_48hr": DriftEvent(
        name="sla_48hr",
        direction="loosening",
        admin_subject="SLA relaxed during platform migration",
        admin_body=(
            "During the platform migration this week, critical-issue SLA is "
            "temporarily extended to 48 hours to give the infra team room. "
            "Please batch follow-ups accordingly."
        ),
        new_sla_hours_critical=48,
    ),
}


def list_drift_events() -> list[str]:
    return list(DRIFT_EVENTS.keys())


def drift_direction(name: str | None) -> Direction | None:
    """Look up the direction label of a drift event by name. Returns None if
    the name is unknown or None."""
    if name is None:
        return None
    ev = DRIFT_EVENTS.get(name)
    return ev.direction if ev else None
