"""
Baseline agent using OpenAI gpt-4o-mini.
Runs on all 3 tasks and prints average score.

Usage:
    OPENAI_API_KEY=sk-... python -m email_env.baseline
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI  # groq uses the same openai SDK

load_dotenv(Path(__file__).parent / ".env")

from email_env.server.environment import EmailTriageEnv
from email_env.models import Action
from email_env.tasks import TASKS

SYSTEM_PROMPT = """You are an email triage assistant. Given an email, you must:
1. Classify the email into exactly one category: billing, technical, or general
2. Assign a priority: low, medium, or high
3. Write a professional response

Reply ONLY with valid JSON in this exact format:
{
  "category": "<billing|technical|general>",
  "priority": "<low|medium|high>",
  "response": "<your response text>"
}"""


def run_baseline():
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set GROQ_API_KEY (or OPENAI_API_KEY) environment variable.")

    base_url = "https://api.groq.com/openai/v1" if os.environ.get("GROQ_API_KEY") else None
    client = OpenAI(api_key=api_key, base_url=base_url)
    env = EmailTriageEnv()
    scores = []

    for task_id, task in TASKS.items():
        print(f"\n--- {task_id} ({task['difficulty']}) ---")
        obs = env.reset(task_id=task_id)
        print(f"Email: {obs.email_text[:80]}...")

        user_msg = (
            f"Sender type: {obs.sender_type}\n\n"
            f"Email:\n{obs.email_text}"
        )

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant" if os.environ.get("GROQ_API_KEY") else "gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
        )

        raw = completion.choices[0].message.content.strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            print(f"Failed to parse response: {raw}")
            scores.append(0.0)
            continue

        action = Action(
            category=parsed.get("category", "general"),
            priority=parsed.get("priority", "low"),
            response=parsed.get("response", ""),
        )

        result = env.step(action)
        print(f"Category: {action.category} (expected: {task['expected_category']})")
        print(f"Priority: {action.priority} (expected: {task['expected_priority']})")
        print(f"Score: {result.reward}")
        scores.append(result.reward)

    avg = round(sum(scores) / len(scores), 4)
    print(f"\n=== Average Score: {avg} ===")
    return avg


if __name__ == "__main__":
    run_baseline()
