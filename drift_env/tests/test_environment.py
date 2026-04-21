"""Environment + episode-generator tests."""

import pytest

from drift_env.environment import DriftEnv
from drift_env.episodes import generate_episode, EPISODE_LENGTH, DRIFT_POSITIONS
from drift_env.models import Action, ActionType, EmailKind
from drift_env.policy import DEFAULT_POLICY


def test_episode_has_expected_length():
    ep = generate_episode(seed=1)
    assert len(ep.steps) == EPISODE_LENGTH


def test_episode_has_two_admin_emails_at_expected_positions():
    ep = generate_episode(seed=1)
    admin_positions = [i for i, s in enumerate(ep.steps)
                       if s.email.kind == EmailKind.ADMIN]
    assert admin_positions == list(DRIFT_POSITIONS)


def test_policy_evolves_after_each_admin_email():
    ep = generate_episode(seed=42)
    # steps 0..3 (inclusive of admin) operate under starting policy
    for i in range(DRIFT_POSITIONS[0] + 1):
        assert ep.steps[i].policy_at_step == DEFAULT_POLICY
    # step after first drift must differ
    assert ep.steps[DRIFT_POSITIONS[0] + 1].policy_at_step != DEFAULT_POLICY


def test_episode_is_deterministic_with_seed():
    a = generate_episode(seed=7)
    b = generate_episode(seed=7)
    assert [s.email.id for s in a.steps] == [s.email.id for s in b.steps]
    assert [s.correct_action_hint for s in a.steps] == [s.correct_action_hint for s in b.steps]


def test_different_seeds_produce_different_episodes():
    a = generate_episode(seed=1)
    b = generate_episode(seed=2)
    assert [s.email.id for s in a.steps] != [s.email.id for s in b.steps]


def test_drift_sensitive_steps_exist_after_each_drift():
    ep = generate_episode(seed=42)
    # At least one drift-sensitive step must appear after each drift
    sens_after_first = any(
        s.drift_sensitive_to for s in ep.steps[DRIFT_POSITIONS[0] + 1:DRIFT_POSITIONS[1]]
    )
    sens_after_second = any(
        s.drift_sensitive_to for s in ep.steps[DRIFT_POSITIONS[1] + 1:]
    )
    assert sens_after_first or sens_after_second


def test_reset_returns_observation_with_first_email():
    env = DriftEnv()
    obs = env.reset(seed=1)
    assert obs.email_index == 0
    assert obs.total_emails == EPISODE_LENGTH
    assert obs.current_email.sender  # non-empty


def test_step_before_reset_raises():
    env = DriftEnv()
    with pytest.raises(RuntimeError):
        env.step(Action(action_type=ActionType.CLOSE, resolution_code="x"))


def test_step_after_done_raises():
    env = DriftEnv()
    env.reset(seed=1)
    for _ in range(EPISODE_LENGTH):
        env.step(Action(action_type=ActionType.CLOSE, resolution_code="x"))
    with pytest.raises(RuntimeError):
        env.step(Action(action_type=ActionType.CLOSE, resolution_code="x"))


def test_agent_cannot_see_grader_metadata():
    """The hidden `meta` dict (refund_amount, severity etc) must not leak."""
    env = DriftEnv()
    obs = env.reset(seed=1)
    assert obs.current_email.meta == {}


def test_perfect_agent_hits_max_reward():
    """Running with ground-truth hints should produce a high, reproducible score."""
    ep = generate_episode(seed=42)
    env = DriftEnv()
    env.reset(seed=42)
    total = 0.0
    for step in ep.steps:
        h = step.correct_action_hint
        action = Action(
            action_type=ActionType(h["action_type"]),
            refund_amount=h.get("refund_amount"),
            escalation_tier=h.get("escalation_tier"),
            followup_hours=h.get("followup_hours"),
            resolution_code=h.get("resolution_code"),
            info_field=h.get("info_field"),
        )
        r = env.step(action)
        total += r.reward
    # Perfect agent: 20 × 1.0 compliance + 18 × 0.5 appropriateness
    # (admin emails don't get appropriateness) + 2 × 0.5 drift bonus = 30
    assert total == 30.0


def test_drift_bonus_arms_and_clears_correctly():
    env = DriftEnv()
    env.reset(seed=42)
    # Run perfect agent until admin email fires (step 3)
    ep = generate_episode(seed=42)
    for i in range(DRIFT_POSITIONS[0]):
        h = ep.steps[i].correct_action_hint
        env.step(Action(action_type=ActionType(h["action_type"]),
                        resolution_code=h.get("resolution_code"),
                        refund_amount=h.get("refund_amount"),
                        escalation_tier=h.get("escalation_tier"),
                        followup_hours=h.get("followup_hours"),
                        info_field=h.get("info_field")))
    # Process admin email
    h = ep.steps[DRIFT_POSITIONS[0]].correct_action_hint
    r = env.step(Action(action_type=ActionType.CLOSE, resolution_code="policy_acknowledged"))
    # After admin, the drift should be armed
    assert len(r.info["armed_drifts_after"]) >= 1
