"""Typed Pydantic models for the Policy Drift environment."""

from __future__ import annotations

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class EmailKind(str, Enum):
    CUSTOMER = "customer"
    ADMIN = "admin"


class Email(BaseModel):
    """A single email visible to the agent."""
    id: str
    kind: EmailKind
    subject: str
    body: str
    sender: str
    # For customer emails only — drift-relevant metadata used by the grader
    # (agent does NOT see these fields; they're stripped before observation).
    meta: dict = Field(default_factory=dict)


class ActionType(str, Enum):
    REPLY = "reply"
    APPROVE_REFUND = "approve_refund"
    ESCALATE = "escalate"
    SCHEDULE_FOLLOWUP = "schedule_followup"
    CLOSE = "close"
    REQUEST_INFO = "request_info"


class Action(BaseModel):
    """A single discrete action with optional scalar/string args."""
    action_type: ActionType
    refund_amount: Optional[float] = None          # approve_refund
    escalation_tier: Optional[str] = None          # escalate: tier_1 | tier_2 | manager
    followup_hours: Optional[int] = None           # schedule_followup
    resolution_code: Optional[str] = None          # close
    info_field: Optional[str] = None               # request_info
    reply_text: Optional[str] = None               # reply


class Observation(BaseModel):
    """What the agent sees. Current policy is NOT exposed — agent must infer
    it from admin emails in the inbox history."""
    current_email: Email
    email_index: int                # 0-based position in episode
    total_emails: int
    inbox_history: List[dict] = Field(default_factory=list)
    # Each history entry: {email_id, kind, subject, body_summary, action_taken?}


class StepResult(BaseModel):
    observation: Optional[Observation]
    reward: float
    done: bool
    info: dict


class State(BaseModel):
    """Env state snapshot for debugging / validator."""
    episode_id: Optional[str]
    email_index: int
    total_emails: int
    done: bool
    cumulative_reward: float
