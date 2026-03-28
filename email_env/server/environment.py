"""
Core OpenEnv environment: reset(), step(action), state()
"""

from email_env.models import Observation, Action, StepResult, State
from email_env.tasks import get_task
from email_env.grader import grade


class EmailTriageEnv:
    def __init__(self):
        self._task: dict | None = None
        self._step_count: int = 0
        self._done: bool = False

    def reset(self, task_id: str = "task_1") -> Observation:
        task = get_task(task_id)
        self._task = task
        self._step_count = 0
        self._done = False
        return Observation(
            email_text=task["email_text"],
            sender_type=task["sender_type"],
        )

    def step(self, action: Action) -> StepResult:
        if self._task is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        self._step_count += 1
        self._done = True  # single-step episode

        reward = grade(action.model_dump(), self._task)

        info = {
            "task_id": self._task["id"],
            "step_count": self._step_count,
            "expected_category": self._task["expected_category"],
            "expected_priority": self._task["expected_priority"],
            "got_category": action.category,
            "got_priority": action.priority,
        }

        return StepResult(
            observation=None,
            reward=reward,
            done=self._done,
            info=info,
        )

    def state(self) -> State:
        return State(
            current_email=self._task["email_text"] if self._task else None,
            step_count=self._step_count,
            done=self._done,
            task_id=self._task["id"] if self._task else None,
        )
