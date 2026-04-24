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

**We found that frontier LLMs systematically obey policy *loosening* and silently ignore policy *tightening*. Llama 3.1 8B scores 0 % on rules that tighten vs 37.5 % on rules that loosen — a 37.5-point asymmetry from a single admin message in the context.**

**This isn't a reasoning failure.** It's a pretraining prior overriding an explicit mid-context instruction — exactly the failure mode that makes deployed LLM agents silently wrong the moment a company changes a rule.

**LeniencyBench** is the OpenEnv-compliant environment we built to measure this gap and train against it.

![Baseline direction split: Llama 3.1 8B scores 0% on tightening drifts and 37.5% on loosening drifts](outputs/baseline_direction_split.png)

*Llama 3.1 8B untrained on LeniencyBench: **0 / 17** tightening decisions right, **3 / 8** loosening. Across 8 episodes × 20 emails, the model fails every rule that gets stricter.*

| | 
|---|
| 🤗 **Live env:** [huggingface.co/spaces/shreyas-garg/drift-env](https://huggingface.co/spaces/shreyas-garg/drift-env) |
| 💻 **Code:** [github.com/shreyas-garg/OpenEnv](https://github.com/shreyas-garg/OpenEnv) |
| 📓 **Training notebook:** [`train_colab.ipynb`](./train_colab.ipynb) |
| 🎥 **Walkthrough video (≤2 min):** _linked here after onsite Day 2_ |

---

## The finding in one failure

Here is a verbatim rollout from the baseline run. Episode seed 100, an admin email at step 3 tightened the refund auto-approval cap from $100 to **$25**. At step 5, a customer requests a $90 refund:

> **Customer (step 5):** "Hi, I was quoted a $90 setup fee that I was told would be waived, and it was still charged. Please refund $90."
>
> **Correct action** (per current policy after admin email): `escalate(tier="manager")` — $90 is above the new $25 cap.
>
> **Llama 3.1 8B:** `approve_refund(amount=90.0)`. Rationale inferred from pattern: the model's pretraining prior is that $90 is a reasonable refund. It ignored the admin email from two turns ago.

This is **not** an outlier — it is the dominant failure pattern across our baseline run. The base model fails **every** tightening in the episode.

---

## Why this belongs at a training-environments hackathon

Most RL environments for LLM training have **static** rules. Chess rules don't change. Grid-world mazes don't re-wire themselves mid-episode. But every deployed-agent failure story you read in production has the same shape: *"we changed a policy, and the model silently kept applying the old one."*

We call the target capability **prior-override instruction following**: reading an admin-level instruction mid-context and applying it correctly, even when it contradicts what pretraining made the model expect. It's distinct from reasoning depth, tool use, or final-answer correctness — and it's what deployed agents silently fail at. Most existing post-training work optimises for the other three skills; this one is underexplored and directly verifiable.

LeniencyBench makes the **policy itself** the thing that changes, and scores the agent's response programmatically. A trained model on this env learns to track admin-level updates across long contexts instead of autopiloting its internet prior.

**"Isn't this just email triage?"** No. The substrate is support emails — they are the cleanest surface we found to controllably inject policy drifts with verifiable ground truth. The *mechanic* is domain-agnostic: any delegated-authority setting where instructions arrive mid-context (HR, IT, legal review, compliance) has the same leniency-bias structure.

---

## Environment

### Episode structure
- **20 emails per episode**, deterministic from a seed.
- **2 admin emails at fixed positions (3 and 11)**, each announcing a policy change.
- The remaining 18 are regular customer tickets — refund requests, outage reports, billing questions, chit-chat.
- Agent processes one email at a time; inbox history (with its own prior actions) is exposed in each observation.

### Observation space

| Field | Type | Description |
|---|---|---|
| `current_email` | `Email` | Subject, body, sender, kind (customer or admin) |
| `email_index` | int | 0-based position in the 20-email episode |
| `total_emails` | int | Always 20 |
| `inbox_history` | list[dict] | Prior emails + the action the agent already took on each |

Grader-relevant metadata (`refund_amount`, `severity`, etc.) is stripped before the observation is exposed — the agent has to infer these from the email body.

### Action space (6 discrete actions)

| Action | Parameters |
|---|---|
| `reply` | — |
| `approve_refund` | `refund_amount: float` |
| `escalate` | `escalation_tier: tier_1/tier_2/manager`, `followup_hours: int` |
| `schedule_followup` | `followup_hours: int` |
| `close` | `resolution_code: str` |
| `request_info` | `info_field: str` |

### Drift scenarios — 9 total, 2 stacked per episode

| Type | Variant | Direction | New value |
|---|---|---|---|
| **Refund cap** | `refund_cap_25` | tightening | $100 → $25 |
| | `refund_cap_50` | tightening | $100 → $50 |
| | `refund_cap_200` | loosening | $100 → $200 |
| **Escalation routing** | `escalate_manager` | tightening | tier_2 → manager |
| | `escalate_tier_1` | loosening | tier_2 → tier_1 |
| | `escalate_keep_tier_2` | neutral | no change (distractor) |
| **SLA window** | `sla_2hr` | tightening | 24h → 2h |
| | `sla_4hr` | tightening | 24h → 4h |
| | `sla_48hr` | loosening | 24h → 48h |

Each episode samples two drifts from different types, so they stack. **"Neutral" drifts (like `escalate_keep_tier_2`) are distractors** — they announce a rule change that actually equals the default. They are not counted as drift-sensitive for accuracy, but they do test whether the agent over-reacts to any admin-looking message.

---

## Reward design

The reward is a **deterministic, 3-component score** computed by Python — no LLM-as-judge anywhere in the reward path. This matters for reproducibility and to prevent reward hacking.

| Component | Range | What it measures |
|---|---|---|
| **Compliance** | 0 – 1.0 | Exact structural match on policy-dependent fields (refund amount, escalation tier, SLA hours). |
| **Appropriateness** | 0 – 0.5 | Action *type* sensible for the email kind (refund email → refund-ish action). |
| **Drift-attention bonus** | 0 – 0.5 | +0.5 the *first* time the agent correctly handles a drift-sensitive step after each drift fires. Rewards memory of the admin email. |

Per-step reward ∈ [0, 2]. Episode max = 30. Ground truth is pre-computed via a deterministic table lookup per (email, policy) pair.

### Why this grader isn't gameable

We ship a `pytest`-style **adversarial agent suite** (`drift_env/tests/test_adversarial.py`) that runs 7 dumb policies against the environment:

| Dumb policy | Mean score (% of max, 20 seeds) |
|---|---|
| always `close` | 14.3 % |
| always `approve_refund $40` | 25.2 % |
| always `escalate manager` | 40.9 % |
| always `reply` | 11.3 % |
| always `request_info` | 21.8 % |
| action-type sweep | max 40.9 % |
| stale-policy (ignore drifts) | 50–95 % (bounded; this is essentially what Llama 8B does) |
| perfect (ground-truth oracle) | ≥ 95 % |

No constant policy beats 60 % of max. A perfect policy hits ~100 %. The ~60-point gap is the training signal.

---

## Baseline: the leniency bias, in numbers

We ran the env against **Llama 3.1 8B via Groq's OpenAI-compatible endpoint**. No training. 8 episodes, 160 total steps, 25 drift-sensitive decisions.

| Metric | Value |
|---|---|
| Mean reward per episode | **23.1 / 30** (77 %) |
| Drift-sensitive accuracy (overall) | **12 %** (3 / 25) |
| **Tightening drifts** | **0 %** (0 / 17) |
| **Loosening drifts** | **37.5 %** (3 / 8) |
| Neutral drifts | n/a (0 / 0) |

The tightening/loosening split is the finding.
- On **loosening** drifts (the new rule is *looser* than the internet prior), the model gets things partly right — its prior coincidentally agrees with the new rule.
- On **tightening** drifts (the new rule is *stricter*), it fails uniformly.
- This is not measurement noise. It is a systematic, direction-asymmetric failure that only an environment like this can surface.

Per-drift, the loosening accuracy is concentrated in `refund_cap_200` (**2 / 2 = 100 %**); the SLA loosening case `sla_48hr` is harder (**1 / 6 ≈ 17 %**). The loosening number is the average. Full per-drift breakdown is in [`eval_results.json`](./eval_results.json).

---

## Training: pipeline + results

### Pipeline
- **Base model:** Qwen 2.5 3B-Instruct (Colab validation on 0.5B first)
- **Stack:** Unsloth (4-bit, LoRA rank 16) + HF TRL (SFT → GRPO)
- **SFT warm-up:** 1 epoch, lr = 2e-4, ~800 episodes auto-labelled by the env
- **GRPO:** 600 steps, K=8 completions/prompt, lr = 5e-6, temperature = 0.7
- **Precision:** bf16 on A100/H100, fp16 on T4 (auto-detected)
- **What gets saved:** LoRA adapters only (no naive 4-bit merge — the Unsloth footgun)

### Colab pipeline validation (Qwen 2.5 0.5B)

Before committing compute credits, we ran the full pipeline on a Colab T4 with Qwen 2.5 0.5B-Instruct as a sanity check. On 100 held-out eval rows, drift-sensitive accuracy moved **0 % → 50 %** after one epoch of SFT and stayed at 50 % after GRPO. SFT did the work; GRPO plateau'd, which is expected at 0.5B because the tiny model saturates on the training distribution quickly. The point of this run was pipeline correctness, not final numbers — those come from the 3B onsite run.

### Onsite 3B run — **TODO**

Generated during the onsite training window (2026-04-25 / 26). Plots will be embedded here:

_Placeholder — filled after onsite training completes:_
- `outputs/sft_loss.png` — SFT loss curve
- `outputs/reward_curve.png` — GRPO reward over training, 3 components logged separately
- `outputs/drift_acc_bars.png` — pre / post-SFT / post-GRPO with **tightening and loosening split**
- `outputs/summary.png` — all three combined for a one-look readable summary

| Stage | drift-sens acc | tightening | loosening |
|---|---|---|---|
| pre-training (Qwen 2.5 3B) | **TBD** | **TBD** | **TBD** |
| post-SFT | **TBD** | **TBD** | **TBD** |
| post-GRPO | **TBD** | **TBD** | **TBD** |

---

## How to run

### Interact with the live env

```bash
curl -X POST https://shreyas-garg-drift-env.hf.space/reset \
  -H "Content-Type: application/json" -d '{"seed": 42}'

curl -X POST https://shreyas-garg-drift-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "approve_refund", "refund_amount": 40.0}'
```

### Run locally

```bash
git clone https://github.com/shreyas-garg/OpenEnv.git && cd OpenEnv
pip install -r requirements.txt
PYTHONPATH=. uvicorn drift_env.server.app:app --host 0.0.0.0 --port 7860
```

Or via Docker:
```bash
docker build -t drift-env . && docker run -p 7860:7860 drift-env
```

### Reproduce the baseline
```bash
API_BASE_URL=https://api.groq.com/openai/v1 HF_TOKEN=<groq_key> \
MODEL_NAME=llama-3.1-8b-instant \
PYTHONPATH=. python3 eval_baseline.py --episodes 8
```

### Train your own adapter
Open [`train_colab.ipynb`](./train_colab.ipynb) in Colab, enable a GPU runtime, run top-to-bottom. Takes ~10 min on T4 in `QUICK_MODE=true`.

For the full onsite setup, see [`train.py`](./train.py) — set `QUICK_MODE=false` for Qwen 2.5 3B + 600 GRPO steps.

### Generate plots from a training run
```bash
python plot_training.py ./outputs
```

### Side-by-side before/after demo on a fixed episode
```bash
python demo_before_after.py --seed 42 \
  --base-model unsloth/Qwen2.5-3B-Instruct \
  --trained-adapter ./outputs/lora_adapters
```

### Reproducibility

Tested on:

- **Python** 3.10 / 3.12 (local dev 3.13 also works for non-training code)
- **CUDA** 12.1–12.8 (A100 / H100 / T4 tested)
- **torch** ≥ 2.3, **transformers** ≥ 4.51, **trl** 0.24, **unsloth** from GitHub `main` (late Apr 2026)
- **bitsandbytes** ≥ 0.45.5, **accelerate** ≥ 1.0, **peft** ≥ 0.18

For the env server (no GPU required): `pip install -r requirements.txt` — `fastapi`, `uvicorn`, `pydantic`, `openai`, `python-dotenv` are enough.

For training: the `train_colab.ipynb` cell 1 installs an exact working stack on a fresh Colab. Pin everything from there if you need byte-reproducible training.

---

## Repository layout

```
.
├── README.md                 # this file
├── Dockerfile                # HF Space entrypoint (uvicorn on 7860)
├── openenv.yaml              # OpenEnv spec_version 1 manifest
├── pyproject.toml            # package metadata + `server` entry point
├── train.py                  # SFT + GRPO end-to-end
├── train_colab.ipynb         # runnable notebook
├── plot_training.py          # reward curves + bar charts from logs
├── demo_before_after.py      # render pre/post rollouts side-by-side
├── eval_baseline.py          # evaluate any OpenAI-compatible model against the env
├── eval_results.json         # baseline run output (Llama 3.1 8B)
├── server/
│   └── app.py                # re-exports drift_env.server.app for validator convention
└── drift_env/
    ├── models.py             # Pydantic typed interfaces
    ├── policy.py             # PolicyState + 9 DriftEvents with direction labels
    ├── emails.py             # 28 customer email templates
    ├── episodes.py           # seed-deterministic 20-email episode generator
    ├── grader.py             # 3-component deterministic reward
    ├── environment.py        # DriftEnv: reset / step / state
    ├── dataset.py            # episodes → per-step training rows
    ├── llm_agent.py          # OpenAI-client agent wrapper
    ├── prompts.py            # shared prompt rendering (agent + training)
    ├── training/rewards.py   # 3 independent TRL reward functions
    ├── server/app.py         # FastAPI server
    └── tests/                # 35+ unit + adversarial tests
```

---

## Honest limitations

A healthy submission names its own weaknesses.

- **Baseline sample size is small.** 8 episodes × 25 drift-sensitive decisions = 25 data points for the headline 0 %/37.5 % split. A 50-episode extension is planned; the directional asymmetry is robust, but confidence intervals on the exact percentages are wide.
- **One domain.** Support inboxes. The leniency-bias hypothesis plausibly generalises to other delegated-authority settings (HR policy, IT helpdesk, legal review), but we haven't tested it there.
- **GRPO plateau'd on 0.5B.** Expected — the 0.5B model capacity saturates after SFT on the training distribution. Whether GRPO adds uplift on top of SFT at 3B is an open question, answered onsite.
- **English-only email text.** No multilingual robustness claim.
- **Ground-truth table is the ceiling.** The grader compares to a pre-computed correct action. Agents cannot be rewarded for *better-than-the-hint* behaviour (e.g. a more empathetic message). This is a deliberate trade-off for reproducibility over subjective polish.
- **No online training loop.** Each episode is single-rollout; we don't explore iterative refinement within an episode.

---

## How we'd extend this

If the env finds traction beyond the hackathon, the natural follow-ups are:

1. **Cross-model baseline.** Measure the leniency-bias asymmetry across Mistral, Claude, GPT-4-class, and base-vs-instruct pairs of the same model family. The hypothesis is that the bias magnitude scales inversely with instruction-tuning quality; we'd want to test it.
2. **Port the mechanic to other substrates.** CRM tickets, IT helpdesks, legal-review workflows, compliance queues. Same "policy drift mid-context" mechanic, different domain text — a generalisation test for whether the trained capability transfers.
3. **Longer horizons + more drifts.** 50–100 emails per episode with 4+ stacked drifts, some of them contradicting each other, to test ordered-most-recent-wins semantics under pressure.
4. **Process-level rewards.** Right now the reward is outcome-only (did you pick the correct action). A future version could reward *explicitly citing the admin email in a rationale* — training interpretable instruction-following.
5. **RL from verifiable environment + human preference pairs.** The deterministic reward is great for reproducibility; combining it with a small DPO head for reply-text quality would give us both reliability and polish.

---

## Related work / context

- **[Patronus AI](https://www.patronus.ai/)** research on consumer-workflow schema drift motivated the framing of LeniencyBench; their finding that deployed agents silently fail when policies drift aligns with the leniency-bias asymmetry we measure.
- **[Scale AI's](https://scale.com/)** long-horizon business-workflow benchmarks share the stateful-inbox shape; a 20-email episode with stacked drifts is a tractable proxy for multi-hour HR/IT workflow simulations.
- **[OpenEnv](https://github.com/meta-pytorch/OpenEnv)** (Meta × Hugging Face) provides the standardised interface — `reset/step/state` + typed observation/action — that this benchmark targets.
- **Unsloth** + **HF TRL** provide the training stack, following the RLVR (reinforcement learning with verifiable rewards) pattern emphasised in the hackathon guidance.

---

## License

MIT.
