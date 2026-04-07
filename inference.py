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
import sys

from openai import OpenAI

from email_env.server.environment import EmailTriageEnv
from email_env.models import Action
from email_env.tasks import TASKS

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN environment variable is required.")

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
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    env = EmailTriageEnv()
    scores = []

    for task_id, task in TASKS.items():
        reward = 0.0
        success = False
        try:
            print(f"[START] task={task_id}", flush=True)

            obs = env.reset(task_id=task_id)

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
                parsed = {"category": "general", "priority": "low", "response": ""}

            action = Action(
                category=parsed.get("category", "general"),
                priority=parsed.get("priority", "low"),
                response=parsed.get("response", ""),
            )

            result = env.step(action)
            reward = float(result.reward)
            done = bool(result.done)
            success = reward >= 0.7

            print(
                f"[STEP] task={task_id} step=1 reward={reward:.2f} "
                f"done={'true' if done else 'false'} "
                f"success={'true' if success else 'false'}",
                flush=True,
            )
            scores.append(reward)

        except Exception as exc:
            print(f"ERROR in task {task_id}: {exc}", file=sys.stderr, flush=True)
        finally:
            print(
                f"[END] task={task_id} score={reward:.2f} steps=1 "
                f"success={'true' if success else 'false'}",
                flush=True,
            )

    avg = round(sum(scores) / len(scores), 2) if scores else 0.0
    print(f"\n=== Average Score: {avg:.2f} ===", flush=True)
    return avg


if __name__ == "__main__":
    run_inference()
