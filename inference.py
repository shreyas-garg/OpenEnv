"""
Inference Script — Email Triage OpenEnv
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables
"""

import os
import json

from openai import OpenAI

from email_env.server.environment import EmailTriageEnv
from email_env.models import Action
from email_env.tasks import TASKS

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

SYSTEM_PROMPT = """You are an email triage assistant. Given an email, you must:
1. Classify the email into exactly one category: billing, technical, or general
2. Assign a priority: low, medium, or high
3. Write a professional response to the sender

Reply ONLY with valid JSON in this exact format (no markdown, no extra text):
{
  "category": "<billing|technical|general>",
  "priority": "<low|medium|high>",
  "response": "<your response text>"
}"""


def run_inference():
    if not API_KEY:
        raise EnvironmentError(
            "Set HF_TOKEN or API_KEY environment variable."
        )

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
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
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=300,
        )

        raw = completion.choices[0].message.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            raw = "\n".join(lines).strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            print(f"Failed to parse LLM response: {raw}")
            # Fallback action
            parsed = {"category": "general", "priority": "low", "response": ""}

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

    avg = round(sum(scores) / len(scores), 4) if scores else 0.0
    print(f"\n=== Average Score: {avg} ===")
    return avg


if __name__ == "__main__":
    run_inference()
