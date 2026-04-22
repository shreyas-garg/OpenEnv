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
from drift_env.prompts import SYSTEM_PROMPT, render_user_prompt


FALLBACK = Action(action_type=ActionType.CLOSE, resolution_code="error_fallback")


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
        user_msg = render_user_prompt(obs)
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
