"""Unit tests for the EmailTriageEnv environment."""

import pytest

from email_env.server.environment import EmailTriageEnv
from email_env.models import Action, Observation, State, StepResult
from email_env.tasks import TASKS


def test_reset_returns_observation():
    env = EmailTriageEnv()
    obs = env.reset("task_1")
    assert isinstance(obs, Observation)
    assert obs.email_text
    assert obs.sender_type in ("customer", "internal", "system")


def test_reset_unknown_task_raises():
    env = EmailTriageEnv()
    with pytest.raises(ValueError):
        env.reset("nope")


def test_step_returns_step_result():
    env = EmailTriageEnv()
    env.reset("task_1")
    result = env.step(Action(category="general", priority="low", response="ok."))
    assert isinstance(result, StepResult)
    assert result.done is True
    assert 0.0 <= result.reward <= 1.0
    assert "task_id" in result.info


def test_step_before_reset_raises():
    env = EmailTriageEnv()
    with pytest.raises(RuntimeError):
        env.step(Action(category="billing", priority="high", response="ok."))


def test_double_step_raises():
    env = EmailTriageEnv()
    env.reset("task_1")
    env.step(Action(category="billing", priority="high", response="ok."))
    with pytest.raises(RuntimeError):
        env.step(Action(category="billing", priority="high", response="ok."))


def test_state_tracks_progress():
    env = EmailTriageEnv()
    env.reset("task_2")
    s1 = env.state()
    assert isinstance(s1, State)
    assert s1.task_id == "task_2"
    assert s1.step_count == 0
    assert s1.done is False
    env.step(Action(category="billing", priority="high", response="ok."))
    s2 = env.state()
    assert s2.step_count == 1
    assert s2.done is True


def test_all_tasks_loadable():
    env = EmailTriageEnv()
    for task_id in TASKS:
        obs = env.reset(task_id)
        assert obs.email_text


def test_each_task_has_required_fields():
    required = {
        "id", "difficulty", "description", "email_text", "sender_type",
        "expected_category", "expected_priority", "response_keywords",
    }
    for task in TASKS.values():
        assert required.issubset(task.keys())
        assert task["expected_category"] in ("billing", "technical", "general")
        assert task["expected_priority"] in ("low", "medium", "high")
        assert task["difficulty"] in ("easy", "medium", "hard")
        assert len(task["response_keywords"]) > 0


def test_tasks_cover_difficulty_range():
    difficulties = {t["difficulty"] for t in TASKS.values()}
    assert {"easy", "medium", "hard"}.issubset(difficulties)


def test_at_least_three_tasks():
    assert len(TASKS) >= 3


def test_perfect_action_per_task_scores_above_threshold():
    """Sanity: a textbook response should clear 0.6 on every task."""
    env = EmailTriageEnv()
    perfect_response = (
        "Thank you for reaching out. We sincerely apologize for the trouble. "
        "Our team will investigate this immediately and contact you with a "
        "resolution. We will refund, escalate, and fix any issues with your "
        "account, billing, hours, support, chat, upgrade, database or "
        "production systems as needed."
    )
    for task_id, task in TASKS.items():
        env.reset(task_id)
        result = env.step(Action(
            category=task["expected_category"],
            priority=task["expected_priority"],
            response=perfect_response,
        ))
        assert result.reward >= 0.6, f"{task_id} scored {result.reward}"
