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

STDOUT FORMAT
- The script emits exactly three line types to stdout:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
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
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
BENCHMARK = "email-triage"
SUCCESS_THRESHOLD = 0.5

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
    all_scores = []

    for task_id in TASKS.keys():
        rewards = []
        success = False
        score = 0.0
        steps = 0
        error_msg = "null"
        action_str = "noop"
        done = False

        print(
            f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}",
            flush=True,
        )

        try:
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
            if raw.startswith("```"):
                lines = [l for l in raw.split("\n") if not l.startswith("```")]
                raw = "\n".join(lines).strip()

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {"category": "general", "priority": "low", "response": ""}
                error_msg = "json_parse_error"

            action = Action(
                category=parsed.get("category", "general"),
                priority=parsed.get("priority", "low"),
                response=parsed.get("response", ""),
            )
            action_str = (
                f"triage(category='{action.category}',"
                f"priority='{action.priority}')"
            )

            result = env.step(action)
            reward = float(result.reward)
            done = bool(result.done)
            steps = 1
            rewards.append(reward)
            score = reward
            success = score >= SUCCESS_THRESHOLD

            print(
                f"[STEP] step=1 action={action_str} reward={reward:.2f} "
                f"done={'true' if done else 'false'} error={error_msg}",
                flush=True,
            )
            all_scores.append(score)

        except Exception as exc:
            error_msg = str(exc).replace("\n", " ")
            print(
                f"[STEP] step=1 action={action_str} reward=0.00 done=true "
                f"error={error_msg}",
                file=sys.stderr,
                flush=True,
            )
        finally:
            rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
            print(
                f"[END] success={'true' if success else 'false'} steps={steps} "
                f"score={score:.2f} rewards={rewards_str}",
                flush=True,
            )

    avg = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0.0
    print(f"\n=== Average Score: {avg:.2f} ===", flush=True)
    return avg


if __name__ == "__main__":
    run_inference()
