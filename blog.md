# LeniencyBench: an environment that measures (and trains out) leniency bias in LLMs

*~2 min read · written for the Meta PyTorch × Hugging Face OpenEnv Hackathon, Round 2 · April 2026*

---

## The finding

Frontier LLMs systematically obey policy *loosening* and silently ignore policy *tightening*.

Llama 3.1 8B, given a 20-email customer-support inbox where an admin email at position 3 announces a rule change, scores:

- **0% accuracy on tightening drifts** (e.g. "refund cap lowered from $100 to $25") — 0 out of 17 tested decisions correct.
- **37.5% on loosening drifts** ("cap raised to $200") — 3 out of 8.

That's a 37-point gap from a single mid-context message. Qwen 2.5 3B shows the same direction-asymmetric pattern: 0% on tightening, 21% on loosening.

This isn't a reasoning failure. It's a pretraining prior overriding an explicit operator instruction. When the new rule matches the model's "be lenient, approve refunds" baseline from the internet, the model goes along. When the new rule contradicts it, the model autopilots its prior — even when the operator clearly stated otherwise two turns ago.

I call this the **leniency bias**. **LeniencyBench** is the environment I built to measure and train it out.

## What's in the environment

LeniencyBench is OpenEnv-compliant — `reset / step / state` over an HTTP endpoint, typed Pydantic observation/action schemas, deployed on Hugging Face Spaces. Each episode is:

- 20 emails, deterministic from a seed
- 18 customer tickets (refund requests, outage reports, billing questions, chitchat)
- 2 admin emails at fixed positions (3 and 11) announcing policy changes
- 9 distinct drift events across 3 axes: refund caps, escalation tiers, SLA windows
- Each drift labeled `tightening`, `loosening`, or `neutral`

The agent picks one of 6 discrete actions per email (`reply / approve_refund / escalate / schedule_followup / close / request_info`), each with typed parameters. The reward is **deterministic Python** — no LLM-as-judge anywhere — combining three signals: policy compliance (0–1), action-type appropriateness (0–0.5), and a drift-attention bonus (0–0.5) for correctly applying a freshly-announced rule.

## Why the reward isn't gameable

Every constant policy I could write — "always escalate to manager," "always approve $40," "always close" — ceilings at 41% of max reward in the committed adversarial test suite. A perfect ground-truth policy hits ~100%. The 60-point gap is the training signal.

## The result

**One epoch of supervised fine-tuning on LeniencyBench's auto-generated labels closes the leniency bias on Qwen 2.5 3B:**

| Stage | Tightening | Loosening | Drift-sens overall |
|---|---|---|---|
| Pre-training | 0.0% | 21.4% | 11.8% |
| Post-SFT | **91.3%** | **71.4%** | **88.2%** |

200 held-out samples on episode seeds 10000–10039 — disjoint from the 0–799 training seeds. The result reproduces exactly across two independent runs (a10g + a100), confirming this is the env's signal, not run-to-run variance.

The trained LoRA adapter (~120 MB) is published at [`shreyas-garg/leniencybench-qwen3b-outputs`](https://huggingface.co/shreyas-garg/leniencybench-qwen3b-outputs). The full SFT log, eval snapshots, and training stdout are in the same repo.

## How to verify it without running training

The published model repo includes `evals.json` — the verbatim output of the held-out eval at all three stages. You can confirm the headline numbers in 30 seconds:

```python
from huggingface_hub import hf_hub_download
import json

p = hf_hub_download(
    "shreyas-garg/leniencybench-qwen3b-outputs", "evals.json", repo_type="model",
)
data = json.load(open(p))
print(data["pre"]["drift_acc_by_direction"])      # tightening: 0.0
print(data["post_sft"]["drift_acc_by_direction"]) # tightening: 0.913
```

## Why it matters

Most existing RL environments for LLM training have static rules. Chess rules don't change. Grid-world mazes don't re-wire mid-episode. But every deployed-agent failure story has the same shape: *"we changed a policy, and the model silently kept applying the old one."*

LeniencyBench measures the gap that prior-override instruction following has to close, and the SFT result is direct evidence the gap is closeable with cheap supervision. The env's mechanic generalizes — anywhere an operator-controlled instruction needs to override a pretraining prior (HR policy updates, IT helpdesk routing changes, compliance flags), the same training shape applies.

## Try it

- **Live OpenEnv server** — [`shreyas-garg/drift-env`](https://huggingface.co/spaces/shreyas-garg/drift-env)
- **Trained adapter + logs** — [`shreyas-garg/leniencybench-qwen3b-outputs`](https://huggingface.co/shreyas-garg/leniencybench-qwen3b-outputs)
- **Source code** — [GitHub](https://github.com/shreyas-garg/OpenEnv) · [HF mirror](https://huggingface.co/shreyas-garg/leniencybench)
- **Training notebook** — [`train_colab.ipynb`](https://huggingface.co/spaces/shreyas-garg/drift-env/blob/main/train_colab.ipynb)

The full README has architecture details, the adversarial-policy table, the train/eval seed split, an honest limitations section, and a related-work block citing prior work on knowledge conflict, lost-in-the-middle, and instruction following.

---

*If you're working on deployed-agent failure modes or post-training benchmarks for instruction following, I'd love to hear from you. The leniency bias measurement is one direction-asymmetric slice of a larger problem; the env mechanic transfers to other delegated-authority substrates (CRM, helpdesk, legal review) and could be extended with cross-model baselines or held-out drift types as a stronger generalization test.*
