---
title: OpenEnv
emoji: 📈
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Email Triage and Response System — OpenEnv

An OpenEnv-compliant environment where an AI agent reads incoming emails, classifies them, assigns a priority, and writes an appropriate response.

## Problem Description

Customer support teams receive hundreds of emails daily. This environment simulates the core triage challenge: given an email, an agent must:

1. **Classify** it — `billing`, `technical`, or `general`
2. **Prioritize** it — `low`, `medium`, or `high`
3. **Respond** — write a professional reply

This is a real-world task performed daily by support teams, and a natural fit for LLM-based agents.

## Observation Space

| Field         | Type   | Values                             |
|---------------|--------|------------------------------------|
| `email_text`  | string | Full email body                    |
| `sender_type` | string | `customer` / `internal` / `system` |

## Action Space

| Field      | Type   | Values                               |
|------------|--------|--------------------------------------|
| `category` | string | `billing` / `technical` / `general`  |
| `priority` | string | `low` / `medium` / `high`            |
| `response` | string | Free-form reply text                 |

## Tasks

| ID     | Difficulty | Description |
|--------|------------|-------------|
| task_1 | Easy       | Simple general inquiry about support hours and live chat |
| task_2 | Easy       | Clear billing complaint — duplicate charge, obvious category |
| task_3 | Medium     | Angry customer — app crash + billing issue + 48h no support reply |
| task_4 | Medium     | System alert — production database lag, stale data, checkout impacted |
| task_5 | Hard       | Genuinely ambiguous — looks like billing but root cause is technical (upgrade not flipped) |

## Reward Function

| Component        | Weight | Condition                              |
|------------------|--------|----------------------------------------|
| Category correct | +0.4   | Exact match with expected category     |
| Priority correct | +0.3   | Exact match with expected priority     |
| Response quality | +0.3   | Hybrid score (see below)               |
| Empty response   | −0.2   | Penalty if response is blank           |

Total score range: **0.0 – 1.0** (continuous, with partial credit).

### Response Quality (hybrid, anti-gaming)

The response quality score combines three signals to defeat keyword-stuffing exploits:

1. **Keyword coverage (40%)** — distinct keywords present
2. **Length & coherence (20%)** — penalises too-short, no-punctuation, repetitive token, and ALL-CAPS responses
3. **Structural requirement (40%)** — must contain BOTH an acknowledgement (sorry/apologise/thank) AND an action/timeline phrase (will/investigate/refund/escalate/...)

This means a response like `"refund refund refund refund refund"` scores well below 0.4, while a genuine professional reply like `"We sincerely apologise for the duplicate charge. Our team will refund your account within 3-5 business days."` scores above 0.9.

## Baseline Scores

Model: `llama-3.1-8b-instant` via Groq (OpenAI-compatible endpoint)

| Task   | Difficulty | Score |
|--------|------------|-------|
| task_1 | Easy       | 0.92  |
| task_2 | Easy       | 0.95  |
| task_3 | Medium     | 0.91  |
| task_4 | Medium     | 0.94  |
| task_5 | Hard       | 0.65  |
| **Avg** |           | **0.87** |

Full baseline trace is committed at [baseline_output.txt](baseline_output.txt).
To reproduce:

```bash
API_BASE_URL=https://api.groq.com/openai/v1 \
HF_TOKEN=<your-key> \
MODEL_NAME=llama-3.1-8b-instant \
PYTHONPATH=. python3 inference.py
```

## Tests

24 unit tests covering grader robustness (including adversarial keyword-stuffing inputs) and environment lifecycle:

```bash
PYTHONPATH=. python3 -m pytest tests/ -v
```

## Setup

```bash
pip install -r requirements.txt
```

## Run the Server

```bash
# Local
PYTHONPATH=. uvicorn email_env.server.app:app --host 0.0.0.0 --port 7860

# Docker
docker build -t email-triage .
docker run -p 7860:7860 email-triage
```

## Run Inference

```bash
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
HF_TOKEN=hf_... \
PYTHONPATH=. python3 inference.py
```

## Example API Calls

### Reset (start a task)
```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_1"}'
```

### Step (submit an action)
```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "category": "billing",
    "priority": "high",
    "response": "We apologize for the duplicate charge. We will issue a full refund within 3-5 business days."
  }'
```

### Get State
```bash
curl http://localhost:7860/state
```

### List Tasks
```bash
curl http://localhost:7860/tasks
```

### Grade Directly
```bash
curl -X POST http://localhost:7860/grader \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "task_1",
    "category": "billing",
    "priority": "high",
    "response": "We apologize and will refund your account."
  }'
```

## Project Structure

```
.
├── Dockerfile           # Root Dockerfile for HF Spaces
├── requirements.txt
├── inference.py         # Mandatory inference script (uses OpenAI client)
├── README.md
└── email_env/
    ├── models.py        # Pydantic: Observation, Action, StepResult, State
    ├── tasks.py         # 5 tasks (easy → hard) with expected outputs
    ├── grader.py        # Deterministic scoring (0.0–1.0)
    ├── client.py        # Python HTTP client
    ├── baseline.py      # Alternate baseline using Groq
    ├── openenv.yaml     # OpenEnv spec metadata
    └── server/
        ├── environment.py  # Core env: reset(), step(), state()
        ├── app.py          # FastAPI server
        └── Dockerfile      # Alternate Dockerfile
```
