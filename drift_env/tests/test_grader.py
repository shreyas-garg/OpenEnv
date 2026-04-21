"""Unit tests for the 3-component drift grader."""

from drift_env.grader import grade_step
from drift_env.models import Action, ActionType


# ---------- compliance ----------

def test_perfect_refund_approval_under_cap():
    action = Action(action_type=ActionType.APPROVE_REFUND, refund_amount=40.0)
    hint = {"action_type": "approve_refund", "refund_amount": 40.0}
    r, b, _ = grade_step(action, hint, {"kind": "refund"}, None, set(), False)
    assert b["compliance"] == 1.0
    assert b["appropriateness"] == 0.5
    assert r == 1.5


def test_refund_wrong_amount_partial_compliance():
    action = Action(action_type=ActionType.APPROVE_REFUND, refund_amount=50.0)
    hint = {"action_type": "approve_refund", "refund_amount": 40.0}
    r, b, _ = grade_step(action, hint, {"kind": "refund"}, None, set(), False)
    assert b["compliance"] == 0.5  # right action_type, wrong amount
    assert b["appropriateness"] == 0.5
    assert r == 1.0


def test_refund_above_cap_should_escalate():
    # Under cap=$50, $75 refund must escalate to manager
    action = Action(action_type=ActionType.ESCALATE, escalation_tier="manager")
    hint = {"action_type": "escalate", "escalation_tier": "manager"}
    r, b, _ = grade_step(action, hint, {"kind": "refund"}, None, set(), False)
    assert b["compliance"] == 1.0
    assert b["appropriateness"] == 0.5


def test_approve_when_should_escalate():
    # Agent approves $75 when cap is $50 — compliance 0
    action = Action(action_type=ActionType.APPROVE_REFUND, refund_amount=75.0)
    hint = {"action_type": "escalate", "escalation_tier": "manager"}
    r, b, _ = grade_step(action, hint, {"kind": "refund"}, None, set(), False)
    assert b["compliance"] == 0.0
    # Appropriate TYPE for refund email (approve_refund is in the valid set)
    assert b["appropriateness"] == 0.5


def test_critical_escalation_tier_match():
    action = Action(action_type=ActionType.ESCALATE, escalation_tier="manager", followup_hours=2)
    hint = {"action_type": "escalate", "escalation_tier": "manager", "followup_hours": 2}
    r, b, _ = grade_step(action, hint, {"kind": "critical_incident"}, None, set(), False)
    assert b["compliance"] == 1.0


def test_critical_wrong_tier_partial():
    action = Action(action_type=ActionType.ESCALATE, escalation_tier="tier_2", followup_hours=2)
    hint = {"action_type": "escalate", "escalation_tier": "manager", "followup_hours": 2}
    r, b, _ = grade_step(action, hint, {"kind": "critical_incident"}, None, set(), False)
    assert 0.5 < b["compliance"] < 1.0  # hours ok, tier wrong


# ---------- appropriateness ----------

def test_refund_close_is_inappropriate():
    # Agent closes a refund email -> wrong compliance AND wrong appropriateness
    action = Action(action_type=ActionType.CLOSE, resolution_code="ack")
    hint = {"action_type": "approve_refund", "refund_amount": 40.0}
    r, b, _ = grade_step(action, hint, {"kind": "refund"}, None, set(), False)
    assert b["compliance"] == 0.0
    assert b["appropriateness"] == 0.0


def test_chitchat_close_is_perfect():
    action = Action(action_type=ActionType.CLOSE, resolution_code="no_action_needed")
    hint = {"action_type": "close", "resolution_code": "no_action_needed"}
    r, b, _ = grade_step(action, hint, {"kind": "chitchat"}, None, set(), False)
    assert b["compliance"] == 1.0
    assert b["appropriateness"] == 0.5


# ---------- drift bonus ----------

def test_drift_bonus_awarded_on_first_correct_drift_sensitive():
    action = Action(action_type=ActionType.APPROVE_REFUND, refund_amount=75.0)
    hint = {"action_type": "approve_refund", "refund_amount": 75.0}
    armed = {"refund_cap_200"}
    r, b, clear = grade_step(
        action, hint, {"kind": "refund"},
        drift_sensitive_to="refund_cap_200", armed_drifts=armed, is_admin_email=False,
    )
    assert b["drift_bonus"] == 0.5
    assert clear == "refund_cap_200"
    assert r == 1.0 + 0.5 + 0.5  # compliance + appropriateness + bonus


def test_drift_bonus_not_awarded_when_already_cleared():
    action = Action(action_type=ActionType.APPROVE_REFUND, refund_amount=75.0)
    hint = {"action_type": "approve_refund", "refund_amount": 75.0}
    armed: set[str] = set()  # already awarded earlier
    r, b, clear = grade_step(
        action, hint, {"kind": "refund"},
        drift_sensitive_to="refund_cap_200", armed_drifts=armed, is_admin_email=False,
    )
    assert b["drift_bonus"] == 0.0
    assert clear is None


def test_drift_bonus_not_awarded_when_compliance_fails():
    # Agent got drift-sensitive step WRONG — no bonus even if armed
    action = Action(action_type=ActionType.ESCALATE, escalation_tier="manager")
    hint = {"action_type": "approve_refund", "refund_amount": 75.0}
    armed = {"refund_cap_200"}
    r, b, clear = grade_step(
        action, hint, {"kind": "refund"},
        drift_sensitive_to="refund_cap_200", armed_drifts=armed, is_admin_email=False,
    )
    assert b["drift_bonus"] == 0.0
    assert clear is None


def test_drift_bonus_not_awarded_when_step_not_drift_sensitive():
    action = Action(action_type=ActionType.APPROVE_REFUND, refund_amount=40.0)
    hint = {"action_type": "approve_refund", "refund_amount": 40.0}
    armed = {"refund_cap_200"}
    r, b, clear = grade_step(
        action, hint, {"kind": "refund"},
        drift_sensitive_to=None, armed_drifts=armed, is_admin_email=False,
    )
    assert b["drift_bonus"] == 0.0
    assert clear is None


# ---------- admin emails ----------

def test_admin_email_close_scores_full():
    action = Action(action_type=ActionType.CLOSE, resolution_code="policy_acknowledged")
    r, b, clear = grade_step(
        action, {}, {}, drift_sensitive_to=None, armed_drifts=set(), is_admin_email=True,
    )
    assert r == 1.0
    assert b["drift_bonus"] == 0.0


def test_admin_email_wrong_action_scores_low():
    action = Action(action_type=ActionType.APPROVE_REFUND, refund_amount=100.0)
    r, b, _ = grade_step(
        action, {}, {}, drift_sensitive_to=None, armed_drifts=set(), is_admin_email=True,
    )
    assert r <= 0.3


# ---------- determinism ----------

def test_grader_is_deterministic():
    action = Action(action_type=ActionType.APPROVE_REFUND, refund_amount=40.0)
    hint = {"action_type": "approve_refund", "refund_amount": 40.0}
    r1, _, _ = grade_step(action, hint, {"kind": "refund"}, None, set(), False)
    r2, _, _ = grade_step(action, hint, {"kind": "refund"}, None, set(), False)
    assert r1 == r2


def test_reward_bounds_per_step():
    """Sweep many (action, hint) combos — reward always in [0, 2.0]."""
    hints = [
        {"action_type": "approve_refund", "refund_amount": 40.0},
        {"action_type": "escalate", "escalation_tier": "manager"},
        {"action_type": "reply"},
        {"action_type": "close", "resolution_code": "no_action_needed"},
    ]
    for h in hints:
        for at in ActionType:
            action = Action(action_type=at, refund_amount=50.0,
                            escalation_tier="tier_2", followup_hours=24,
                            resolution_code="x", info_field="y")
            r, _, _ = grade_step(
                action, h, {"kind": "refund"},
                drift_sensitive_to="refund_cap_200",
                armed_drifts={"refund_cap_200"},
                is_admin_email=False,
            )
            assert 0.0 <= r <= 2.0, f"out of bounds: {r}"
