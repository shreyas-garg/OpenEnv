---
license: mit
language:
  - en
base_model: unsloth/Qwen2.5-3B-Instruct
library_name: peft
tags:
  - openenv
  - lora
  - leniency-bias
  - rlvr
  - sft
---

# LeniencyBench — Qwen 2.5 3B SFT-trained adapter

LoRA adapter for **Qwen 2.5 3B-Instruct**, fine-tuned on [LeniencyBench](https://huggingface.co/spaces/shreyas-garg/drift-env), an OpenEnv-compliant environment that measures and trains out **leniency bias** in LLMs — the systematic failure to apply mid-context rule tightenings.

## Headline result

| Stage | Tightening accuracy | Loosening accuracy | Drift-sensitive overall |
|---|---|---|---|
| Pre-training (Qwen 2.5 3B base) | **0.0 %** (0/23) | 21.4 % (3/14) | 11.8 % (2/17) |
| **Post-SFT (this adapter)** | **91.3 %** (21/23) | **71.4 %** (10/14) | **88.2 %** (15/17) |

200 held-out samples on episode seeds 10000–10039 (disjoint from the 0–799 training seeds).

The result reproduces exactly across two independent training runs (a10g + a100), giving us confidence the gap closure is the env's signal, not run-to-run noise.

## What's in this repo

| File | What it is |
|---|---|
| `lora_adapters/` | **Final LoRA adapter** (post-SFT — this is what you load) |
| `lora_adapters_sft/` | Same adapter, kept under both names for clarity |
| `evals.json` | Pre / post-SFT eval snapshots with per-direction breakdown |
| `sft_log.json` | Full SFT training log (loss per step, ~1000 steps) |
| `v6_full_logs.txt` | Raw stdout from a parallel run that confirmed the same result |
| `baseline_direction_split.png` | Cross-model baseline plot |

## Loading the adapter

```python
from unsloth import FastLanguageModel
from peft import PeftModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-3B-Instruct",
    max_seq_length=4096,
    load_in_4bit=True,
)
model = PeftModel.from_pretrained(
    model, "shreyas-garg/leniencybench-qwen3b-outputs",
    subfolder="lora_adapters",
)
FastLanguageModel.for_inference(model)
```

## Training recipe

- **Base:** unsloth/Qwen2.5-3B-Instruct (4-bit)
- **LoRA:** rank 16, alpha 16, dropout 0, all 7 linear projections targeted
- **Optimizer:** SFTTrainer (HF TRL 0.24), lr 2e-4, batch 4, grad accum 4, bf16
- **Data:** 16,000 per-step samples drawn from 800 LeniencyBench episodes (seeds 0–799)
- **Compute:** 1 epoch in ~100 min on A100-SXM4-80GB
- **Held-out eval:** seeds 10000–10039, capped at 200 samples

## Read more

- **Live env (run it yourself):** [shreyas-garg/drift-env](https://huggingface.co/spaces/shreyas-garg/drift-env)
- **Source code (HF mirror):** [shreyas-garg/leniencybench](https://huggingface.co/shreyas-garg/leniencybench)
- **Source code (GitHub):** [shreyas-garg/OpenEnv](https://github.com/shreyas-garg/OpenEnv)

Built for the Meta PyTorch × Hugging Face OpenEnv Hackathon (Round 2).
