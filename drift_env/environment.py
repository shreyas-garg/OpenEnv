"""DriftEnv — OpenEnv-compliant multi-step environment for the
Policy-Drift support-triage task.
"""

from __future__ import annotations

from typing import Optional, List, Set

from drift_env.models import Action, Email, EmailKind, Observation, State, StepResult
from drift_env.episodes import Episode, generate_episode
from drift_env.grader import grade_step


def _email_to_summary(email: Email, action_taken: Optional[Action] = None) -> dict:
    entry = {
        "email_id": email.id,
        "kind": email.kind.value,
        "subject": email.subject,
        "body": email.body,
        "sender": email.sender,
    }
    if action_taken is not None:
        entry["action_taken"] = action_taken.action_type.value
    return entry


def _email_for_observation(email: Email) -> Email:
    """Strip grader-only metadata before exposing to agent."""
    return Email(
        id=email.id,
        kind=email.kind,
        subject=email.subject,
        body=email.body,
        sender=email.sender,
        meta={},
    )


class DriftEnv:
    def __init__(self) -> None:
        self._episode: Optional[Episode] = None
        self._index: int = 0
        self._cumulative: float = 0.0
        self._done: bool = True
        self._history: List[dict] = []
        self._armed_drifts: Set[str] = set()

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------
    def reset(self, seed: int = 0, episode_id: str = "ep_0") -> Observation:
        self._episode = generate_episode(seed=seed, episode_id=episode_id)
        self._index = 0
        self._cumulative = 0.0
        self._done = False
        self._history = []
        self._armed_drifts = set()
        return self._current_observation()

    def step(self, action: Action) -> StepResult:
        if self._episode is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        step_record = self._episode.steps[self._index]
        is_admin = step_record.email.kind == EmailKind.ADMIN

        reward, breakdown, drift_to_clear = grade_step(
            action=action,
            hint=step_record.correct_action_hint,
            email_meta=step_record.email.meta,
            drift_sensitive_to=step_record.drift_sensitive_to,
            armed_drifts=self._armed_drifts,
            is_admin_email=is_admin,
        )

        # If agent proved awareness of a drift, clear the arm.
        if drift_to_clear:
            self._armed_drifts.discard(drift_to_clear)

        # If THIS step is an admin email, arm its drift for future steps.
        if is_admin:
            drift_name = step_record.email.meta.get("drift_event")
            if drift_name:
                self._armed_drifts.add(drift_name)

        self._cumulative += reward

        # Record this email + the action for history.
        self._history.append(_email_to_summary(step_record.email, action_taken=action))

        self._index += 1
        self._done = self._index >= len(self._episode.steps)

        next_obs = None if self._done else self._current_observation()

        info = {
            "episode_id": self._episode.id,
            "step": self._index,
            "total_steps": len(self._episode.steps),
            "correct_action_hint": step_record.correct_action_hint,
            "drift_sensitive_to": step_record.drift_sensitive_to,
            "armed_drifts_after": sorted(self._armed_drifts),
            "breakdown": breakdown,
            "cumulative_reward": round(self._cumulative, 4),
        }

        return StepResult(
            observation=next_obs,
            reward=round(reward, 4),
            done=self._done,
            info=info,
        )

    def state(self) -> State:
        return State(
            episode_id=self._episode.id if self._episode else None,
            email_index=self._index,
            total_emails=len(self._episode.steps) if self._episode else 0,
            done=self._done,
            cumulative_reward=round(self._cumulative, 4),
        )

    # ------------------------------------------------------------------
    def _current_observation(self) -> Observation:
        assert self._episode is not None
        step = self._episode.steps[self._index]
        return Observation(
            current_email=_email_for_observation(step.email),
            email_index=self._index,
            total_emails=len(self._episode.steps),
            inbox_history=list(self._history),
        )
