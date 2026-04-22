"""Adversarial sanity test.

Runs a handful of dumb policies (constant-action agents, mode-based agents,
stale-policy agents) against the environment. None of them should score well —
if any of them beats ~60% of max, our reward is gameable and we need to fix it.

Per organizer guidance: "Do not optimize a reward you have not tried to break
yourself first."
"""

from drift_env.environment import DriftEnv
from drift_env.episodes import generate_episode
from drift_env.models import Action, ActionType


EP_MAX = 30.0   # per-episode maximum reward (20 compliance + 9 appropriateness + 1 drift bonus)
N_SEEDS = 20    # episodes to average over for stability
GAMEABILITY_CEILING = 0.60  # no dumb policy should exceed this fraction of max


def _run_policy(policy_fn, n_seeds: int = N_SEEDS) -> float:
    """Run a callable(step_plan) -> Action across n episodes and return mean
    fraction of max reward."""
    total = 0.0
    for i in range(n_seeds):
        env = DriftEnv()
        env.reset(seed=i, episode_id=f"adv_{i}")
        ep = generate_episode(seed=i, episode_id=f"adv_{i}")
        ep_reward = 0.0
        for step in ep.steps:
            result = env.step(policy_fn(step))
            ep_reward += result.reward
        total += ep_reward / EP_MAX
    return total / n_seeds


def test_always_close_fails():
    mean = _run_policy(lambda s: Action(action_type=ActionType.CLOSE, resolution_code="x"))
    assert mean < GAMEABILITY_CEILING, f"always-close scored {mean:.2%}"


def test_always_approve_40_fails():
    mean = _run_policy(lambda s: Action(
        action_type=ActionType.APPROVE_REFUND, refund_amount=40.0,
    ))
    assert mean < GAMEABILITY_CEILING, f"always-approve-40 scored {mean:.2%}"


def test_always_escalate_manager_fails():
    mean = _run_policy(lambda s: Action(
        action_type=ActionType.ESCALATE, escalation_tier="manager", followup_hours=24,
    ))
    assert mean < GAMEABILITY_CEILING, f"always-escalate-manager scored {mean:.2%}"


def test_always_reply_fails():
    mean = _run_policy(lambda s: Action(action_type=ActionType.REPLY))
    assert mean < GAMEABILITY_CEILING, f"always-reply scored {mean:.2%}"


def test_random_action_mode_fails():
    """Constant mode: pick whichever ACTION_TYPE dominates correct answers."""
    # empirically close, approve_refund, and escalate will dominate
    for at in ActionType:
        mean = _run_policy(lambda s, at=at: Action(
            action_type=at,
            refund_amount=50.0, escalation_tier="manager", followup_hours=24,
            resolution_code="x", info_field="x",
        ))
        assert mean < GAMEABILITY_CEILING, f"always-{at.value} scored {mean:.2%}"


def test_stale_policy_agent_approximates_baseline():
    """An agent that behaves correctly under the STARTING policy (refund_cap=$100,
    tier_2 escalations, 24h SLA) but ignores all drift should score ~baseline:
    good on non-drift-sensitive steps, 0 on drift-sensitive ones. This is the
    canonical base-model behaviour we saw in the eval (~77% of max). The point
    of this test: it should NOT accidentally earn the drift bonus."""
    from drift_env.policy import DEFAULT_POLICY
    from drift_env.episodes import _correct_action_hint

    def stale(step):
        if step.email.kind.value == "admin":
            return Action(action_type=ActionType.CLOSE, resolution_code="policy_acknowledged")
        # Pick the action correct under the DEFAULT policy (ignoring drifts)
        template_kind = step.email.meta.get("kind", "")
        refund_amt = step.email.meta.get("refund_amount")
        # Rebuild a minimal template-like object:
        class _T:
            kind = template_kind
            refund_amount = refund_amt
            severity = step.email.meta.get("severity")
            needs_info = step.email.meta.get("needs_info")
        hint = _correct_action_hint(_T(), DEFAULT_POLICY)
        return Action(
            action_type=ActionType(hint["action_type"]),
            refund_amount=hint.get("refund_amount"),
            escalation_tier=hint.get("escalation_tier"),
            followup_hours=hint.get("followup_hours"),
            resolution_code=hint.get("resolution_code"),
            info_field=hint.get("info_field"),
        )

    mean = _run_policy(stale)
    # Stale agent should beat dumb constants but shouldn't reach perfect
    assert 0.5 < mean < 0.95, f"stale-policy agent scored {mean:.2%} (expected 0.5-0.95)"


def test_perfect_agent_still_wins_by_margin():
    """Reference: a perfect agent (uses policy-aware ground truth) must clearly
    beat all the above. This protects against the worst failure mode: a dumb
    agent scoring as high as a perfect one."""
    from drift_env.episodes import _correct_action_hint

    def perfect(step):
        h = step.correct_action_hint
        return Action(
            action_type=ActionType(h["action_type"]),
            refund_amount=h.get("refund_amount"),
            escalation_tier=h.get("escalation_tier"),
            followup_hours=h.get("followup_hours"),
            resolution_code=h.get("resolution_code"),
            info_field=h.get("info_field"),
        )

    mean = _run_policy(perfect)
    assert mean > 0.95, f"perfect agent only scored {mean:.2%}"
