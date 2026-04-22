"""Shared prompt rendering used by both the inference agent and the
training-dataset generator. Keeping it in one place guarantees the training
distribution matches the inference distribution exactly.
"""

from __future__ import annotations

from drift_env.models import Observation


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
  - Refunds at or below the current auto-approval cap -> approve_refund with
    the requested amount. Above the cap -> escalate to manager.
  - Critical incidents -> escalate to the currently-mandated tier with a
    followup_hours matching the current critical-SLA.
  - Admin email -> close with resolution_code "policy_acknowledged".
  - Questions you cannot answer without a detail the customer omitted
    -> request_info with the missing field name (e.g. order_id, account_email).
  - Pure thank-you / FYI / chit-chat -> close with resolution_code
    "no_action_needed".
  - Product / how-does-this-work questions -> reply.

Reply with ONLY the JSON object. No prose, no markdown."""


def render_history(history: list[dict], last_n: int = 8, max_body_chars: int = 200) -> str:
    """Compact text rendering of prior inbox entries."""
    if not history:
        return "(no prior emails in this session)"
    shown = history[-last_n:]
    lines = []
    for h in shown:
        prefix = "[ADMIN]" if h["kind"] == "admin" else "[CUSTOMER]"
        action = h.get("action_taken", "?")
        body = h["body"]
        if len(body) > max_body_chars:
            body = body[:max_body_chars] + "..."
        lines.append(
            f"{prefix} subject={h['subject']!r}\n  body: {body}\n  action_taken: {action}"
        )
    return "\n".join(lines)


def render_user_prompt(obs: Observation) -> str:
    hist = render_history(obs.inbox_history)
    cur = obs.current_email
    return (
        f"INBOX SO FAR (most recent last):\n{hist}\n\n"
        f"------\nCURRENT EMAIL (#{obs.email_index + 1} of {obs.total_emails}):\n"
        f"  from: {cur.sender}\n  subject: {cur.subject}\n  body: {cur.body}\n\n"
        f"Reply with exactly one JSON action."
    )
