"""A thin LLM agent that reads the current email + inbox history and emits
an Action. Used by both the eval harness and the onsite inference pipeline.

Uses the OpenAI Python client against any OpenAI-compatible endpoint
(HF router, Groq, etc). No Anthropic / Google SDKs — hackathon requires
OpenAI client only.
"""

from __future__ import annotations

import json
from typing import Optional

from openai import OpenAI

from drift_env.models import Action, ActionType, Observation


SYSTEM_PROMPT = """You are a customer-support triage agent.

Your inbox contains a mix of:
  - CUSTOMER emails: regular tickets (refunds, complaints, questions, chit-chat).
  - ADMIN emails from ops@company.com: internal policy announcements that
    change the rules you must follow from that point onwards.

You must read the CURRENT email and reply with exactly one action, formatted
as a single JSON object. Allowed actions and fields:

  {"action_type": "reply"}
  {"action_type": "approve_refund", "refund_amount": <number>}
  {"action_type": "escalate", "escalation_tier": "tier_1" | "tier_2" | "manager", "followup_hours": <int>}
  {"action_type": "schedule_followup", "followup_hours": <int>}
  {"action_type": "close", "resolution_code": <short string>}
  {"action_type": "request_info", "info_field": <short string>}

Rules of thumb (apply the MOST RECENT admin policy you have seen):
  - Refunds at or below the current auto-approval cap → approve_refund with
    the requested amount. Above the cap → escalate to manager.
  - Critical incidents → escalate to the currently-mandated tier with a
    followup_hours matching the current critical-SLA.
  - Admin email → close with resolution_code "policy_acknowledged".
  - Questions you cannot answer without a detail the customer omitted
    → request_info with the missing field name (e.g. order_id, account_email).
  - Pure thank-you / FYI / chit-chat → close with resolution_code
    "no_action_needed".
  - Product / how-does-this-work questions → reply.

Reply with ONLY the JSON object. No prose, no markdown."""


FALLBACK = Action(action_type=ActionType.CLOSE, resolution_code="error_fallback")


def _render_history(history: list[dict], last_n: int = 8) -> str:
    """Compact text rendering of prior inbox entries for context."""
    if not history:
        return "(no prior emails in this session)"
    shown = history[-last_n:]
    lines = []
    for h in shown:
        prefix = "[ADMIN]" if h["kind"] == "admin" else "[CUSTOMER]"
        action = h.get("action_taken", "?")
        # Truncate body for token budget
        body = h["body"]
        if len(body) > 200:
            body = body[:200] + "..."
        lines.append(
            f"{prefix} subject={h['subject']!r}\n  body: {body}\n  action_taken: {action}"
        )
    return "\n".join(lines)


def _render_observation(obs: Observation) -> str:
    hist = _render_history(obs.inbox_history)
    cur = obs.current_email
    return (
        f"INBOX SO FAR (most recent last):\n{hist}\n\n"
        f"------\nCURRENT EMAIL (#{obs.email_index + 1} of {obs.total_emails}):\n"
        f"  from: {cur.sender}\n  subject: {cur.subject}\n  body: {cur.body}\n\n"
        f"Reply with exactly one JSON action."
    )


def _parse_action(raw: str) -> Action:
    """Parse the LLM output into an Action. Returns FALLBACK on failure."""
    text = raw.strip()
    # strip markdown fences if present
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    # Try to locate a JSON object if surrounded by prose
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return FALLBACK
        text = text[start:end + 1]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return FALLBACK
    a_type = obj.get("action_type")
    if a_type not in {e.value for e in ActionType}:
        return FALLBACK
    try:
        return Action(
            action_type=ActionType(a_type),
            refund_amount=obj.get("refund_amount"),
            escalation_tier=obj.get("escalation_tier"),
            followup_hours=obj.get("followup_hours"),
            resolution_code=obj.get("resolution_code"),
            info_field=obj.get("info_field"),
            reply_text=obj.get("reply_text"),
        )
    except Exception:
        return FALLBACK


class LLMAgent:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 200,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

    def act(self, obs: Observation) -> tuple[Action, str]:
        """Returns (parsed_action, raw_text) for inspection."""
        user_msg = _render_observation(obs)
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            raw = completion.choices[0].message.content or ""
        except Exception as e:
            return FALLBACK, f"ERROR: {e}"
        return _parse_action(raw), raw
