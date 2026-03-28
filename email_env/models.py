from pydantic import BaseModel, Field
from typing import Optional


class Observation(BaseModel):
    email_text: str
    sender_type: str  # customer / internal / system


class Action(BaseModel):
    category: str   # billing / technical / general
    priority: str   # low / medium / high
    response: str


class StepResult(BaseModel):
    observation: Optional[Observation]
    reward: float
    done: bool
    info: dict


class State(BaseModel):
    current_email: Optional[str]
    step_count: int
    done: bool
    task_id: Optional[str]
