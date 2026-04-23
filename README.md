---
title: LeniencyBench
emoji: 📉
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
tags:
  - openenv
---

# LeniencyBench

An OpenEnv benchmark that measures (and trains out) a specific failure mode in deployed LLMs: **when a rule gets stricter, the model ignores it and keeps applying the looser rule it saw on the internet.**

The environment simulates a customer-support inbox where policies drift mid-episode. An agent receives 20 emails per episode, and at 2 fixed positions an *admin email* arrives announcing a rule change (e.g. "refund cap lowered from $100 to $50"). The agent must read, remember, and apply the new rule to subsequent tickets.

We call this the **leniency bias**. In our 8-episode baseline with Llama 3.1 8B: **0% accuracy on rules that tighten, 100% on rules that loosen.** LeniencyBench is the training target that closes that gap.

Built for the Meta PyTorch × Hugging Face OpenEnv Hackathon (Round 2).
**Bonus-prize fit:** Patronus AI (schema drift) + Scale AI (long-horizon business workflows).

## The leniency bias in one paragraph

A pretrained LLM has seen millions of customer-support conversations on the internet. Most of them default to "approve the refund", "apologize and resolve", "be lenient". When you deploy it and change a policy — say, tighten the refund cap — the model doesn't actually listen. It autopilots its internet prior. This is why agents built on generic LLMs silently fail the moment a real company's rules diverge from what the internet made them expect. LeniencyBench measures this directly, and is the training target that closes it.

## Why this environment is different

Most RL environments have static rules. This one makes the rule itself the thing that changes. Training on LeniencyBench produces agents that attend to policy updates across long contexts instead of autopiloting priors.

## Observation space

| Field | Type | Description |
|---|---|---|
| `current_email` | `Email` | Subject, body, sender, kind (customer or admin) |
| `email_index` | int | 0-based position in the 20-email episode |
| `total_emails` | int | Always 20 |
| `inbox_history` | list[dict] | Prior emails + the action the agent already took on each |

## Action space (6 discrete actions)

| Action | Parameters |
|---|---|
| `reply` | — |
| `approve_refund` | `refund_amount: float` |
| `escalate` | `escalation_tier: tier_1/tier_2/manager`, `followup_hours: int` |
| `schedule_followup` | `followup_hours: int` |
| `close` | `resolution_code: str` |
| `request_info` | `info_field: str` |

## Drift scenarios (3 types × 3 variants = 9)

- **Refund cap** — `$100 → $25 / $50 / $200`
- **Critical escalation routing** — `tier_2 → tier_1 / manager / no-change`
- **Critical SLA window** — `24h → 2h / 4h / 48h`

Each episode samples **two** drifts from different types (so they stack). Admin emails land at positions `[3, 11]`.

## Reward function (deterministic, 3-component, [0, 2])

| Component | Weight | What it scores |
|---|---|---|
| **Compliance** | 0 – 1.0 | Exact match on policy-dependent fields (refund amount, tier, SLA hours) |
| **Appropriateness** | 0 – 0.5 | Action *type* sensible for the email kind |
| **Drift-attention bonus** | 0 – 0.5 | +0.5 the FIRST time the agent nails a drift-sensitive step after each drift fires |

No LLM-as-judge anywhere in the reward path. Ground truth is pre-computed via a deterministic table lookup; reward is reproducible byte-for-byte across runs.

## Adversarial sanity check

Seven constant-action baselines (always-close, always-approve-$40, always-escalate-manager, etc.) all score under **41% of max** across 20 seeds. A perfect policy hits **~100%**. This ~60 pp gap is the training signal.

## Baseline numbers

| Model | Source | Overall reward | Drift-sensitive accuracy |
|---|---|---|---|
| Llama 3.1 8B | Groq API, 8 episodes | 23.1 / 30 (77%) | **12%** |
| Qwen 2.5 0.5B (raw) | Colab T4 | 0.62 / 2 per step | **0%** |
| Qwen 2.5 0.5B (post-SFT) | Colab T4, 1 epoch | 1.37 / 2 per step | **50%** |

3B training numbers will be produced onsite with HF compute credits.

## API

### `POST /reset`
Body: `{"seed": 0, "episode_id": "ep_0"}` (both optional).
Returns the initial `Observation`.

### `POST /step`
Body: a single `Action` JSON object.
Returns `StepResult { observation, reward, done, info }`.

### `GET /state`
Returns a snapshot of the current episode state.

### `GET /`
Returns a name/version/description blob.

## Example

```bash
# Start an episode
curl -X POST https://shreyas-garg-drift-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"seed": 42}'

# Submit an action
curl -X POST https://shreyas-garg-drift-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "approve_refund", "refund_amount": 40.0}'
```

## Run locally

```bash
pip install -r requirements.txt
PYTHONPATH=. uvicorn drift_env.server.app:app --host 0.0.0.0 --port 7860

# or in Docker:
docker build -t drift-env .
docker run -p 7860:7860 drift-env
```

## Training

End-to-end SFT warm-up + GRPO pipeline lives in `train.py` and `train_colab.ipynb` at the repo root. Uses Unsloth + HF TRL. Starts from Qwen 2.5 0.5B/3B, saves LoRA adapters. See the Colab notebook for a runnable setup.

## Project layout

```
.
├── Dockerfile               # HF Space entrypoint (port 7860)
├── openenv.yaml             # OpenEnv spec_version 1
├── pyproject.toml
├── requirements.txt
├── train.py                 # SFT + GRPO training loop
├── train_colab.ipynb        # Colab-runnable notebook
├── server/
│   └── app.py               # re-exports drift_env.server.app (validator convention)
└── drift_env/
    ├── models.py            # Pydantic Observation / Action / StepResult / State
    ├── policy.py            # PolicyState + 9 DriftEvent scenarios
    ├── emails.py            # 28 customer email templates
    ├── episodes.py          # Deterministic 20-email episode generator
    ├── grader.py            # 3-component deterministic reward
    ├── environment.py       # DriftEnv: reset / step / state
    ├── dataset.py           # Episodes -> training rows for SFT/GRPO
    ├── llm_agent.py         # OpenAI-client agent (used in eval_baseline.py)
    ├── prompts.py           # System + user prompt rendering
    ├── training/rewards.py  # TRL-compatible reward functions (3 independent)
    ├── server/app.py        # FastAPI server
    └── tests/               # 35+ unit + adversarial tests
```

## Links

- **Live Space**: [huggingface.co/spaces/shreyas-garg/drift-env](https://huggingface.co/spaces/shreyas-garg/drift-env)
- **GitHub**: [github.com/shreyas-garg/OpenEnv](https://github.com/shreyas-garg/OpenEnv)

## License

MIT.
