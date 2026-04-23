# Pitch Deck — Policy Drift OpenEnv

**Team:** Shreyas Garg (solo)
**Format:** 3 min pitch + 2 min Q&A
**Themes hit:** Theme 3.2 Personalized Tasks (primary), Theme 2 Long-Horizon (secondary)
**Bonus prizes targeted:** Patronus AI (schema drift) + Scale AI (long-horizon business workflow)

---

## SLIDE 1 — Hook (~20 sec)

### Headline
> **Agents break when the rules change.**

### Visual
Split-screen screenshot:
- Left: a policy doc that reads "**Refund cap: $100**" (with "$100" crossed out, "$50" written in red)
- Right: an AI assistant replying to a $75 refund with "**Approved!**" ❌

### Subtitle
`Policy-Drift OpenEnv — training LLMs to survive schema drift`

### Speaker notes (what you say verbatim)
"Imagine you joined a company on Monday. They tell you: refunds over fifty dollars need manager approval. On Tuesday, a customer asks for a seventy-five dollar refund. Do you approve it? Of course not — you remembered the rule.

Now imagine the AI version of you. It read the internet, which says refunds are fine. Nobody told it the new rule. It clicks approve. That's the failure mode I built an environment to solve."

---

## SLIDE 2 — Why this matters (~30 sec)

### Headline
> **Real companies change their rules every week.**

### Visual (bullet list with small logos)
- **Stripe** quietly shifts refund windows
- **Shopify** deprecates API endpoints on a rolling cadence
- **Airlines** update SLA policies mid-quarter
- **HR systems** change leave policies overnight

### Subtitle
"Every deployed AI agent faces this. None of the existing RL environments simulate it."

### Speaker notes
"The theme the hackathon gave us is world modeling for personal and professional tasks. The hardest thing about real-world tasks isn't complexity. It's that **the rules move**. Every existing OpenEnv environment I've seen has static rules. That's the gap."

---

## SLIDE 3 — The environment (the heart of the pitch, ~60 sec)

### Headline
> **20-email customer-support episode. Rules drift mid-episode. Agent must adapt.**

### Visual — the money shot
Horizontal timeline of 20 emails with icons:
```
[email][email][email][ADMIN]…[email][email]…[ADMIN]…[email][email][email]…
  1     2     3     4         5    6          12         13    14    20
                    ▲                          ▲
             "Refund cap                  "Critical incidents
             $100 → $50"                   → manager (was Tier 2)"
```
Overlay at turn 8:
```
📧 Customer: "Please refund my $75."
❌ Pre-trained agent: "Approved! $75 refunded."       (applied OLD $100 cap)
✅ Trained agent:    "Escalating to manager."         (applied NEW $50 cap)
```

### Key specs (bottom of slide, small font)
- **3 drift types × 3 variants = 9 scenarios**, randomized per episode
- **Admin emails** buried in the inbox; model must attend to them
- **6 discrete actions** with numeric parameters (refund amount, escalation tier, SLA hours)
- **Deterministic ground-truth lookup** — no LLM-as-judge in reward

### Speaker notes
"Every episode is twenty emails. Mostly normal customer tickets. But at position three, an admin email arrives: refund cap lowered from one hundred to fifty dollars. The agent has to read that, remember it, and apply it to customer emails later in the episode.

We have nine different policy drifts, randomized per episode. The agent never sees the same game twice. The only way to score well is to actually learn to attend to admin emails.

And critically: **the reward is a deterministic lookup table**, not an LLM judge. So our reward can't be gamed — I wrote seven adversarial tests and every dumb policy scores under 41%. A trained policy hits nearly 100%."

---

## SLIDE 4 — Results: training actually works (~45 sec)

### Headline
> **Drift-sensitive accuracy: [PRE]% → [POST-SFT]% → [POST-GRPO]%**

### Visual — reward curve
Two plots side by side (matplotlib output, straight from training):
- Left: **reward per training step** (upward trending line, 3 colored components: compliance + appropriateness + drift bonus)
- Right: **drift-sensitive accuracy over training** (the key metric — stepped bar chart pre / post-SFT / post-GRPO)

### Placeholder numbers to swap onsite
```
PLACEHOLDER (fill in onsite with 3B run):
  Pre-training  (Qwen 2.5 3B baseline):   drift-sens acc = ??%
  Post-SFT      (1 epoch, ~800 episodes): drift-sens acc = ??%
  Post-GRPO     (600 steps, K=8):         drift-sens acc = ??%
```

### Colab validation (0.5B, proof of pipeline)
Small corner annotation:
> "Pipeline validated on Qwen 2.5 0.5B in Colab: 0% → 50% drift-sensitive acc. Scaled numbers on 3B Qwen trained onsite."

### Speaker notes
"Here's the proof. Before training, the base model gets zero out of eight drift-sensitive questions. It defaults to its internet prior every time.

After one epoch of supervised fine-tuning on our environment's auto-generated labels, it jumps to fifty percent. GRPO on top pushes us to [X]%.

Each reward component is logged independently — you can see which part of the reward is driving the learning. In this case, compliance is where the gains come from, exactly as we designed it."

---

## SLIDE 5 — Why we win (~25 sec)

### Headline
> **Real-world relevant. Dual bonus-prize fit. Production-ready.**

### Three columns
**Patronus AI bonus ✅**
"Consumer workflows with schema drift" — this environment is literally that.

**Scale AI bonus ✅**
"Long-horizon business workflows in HR/IT" — 20-email episodes, multi-drift, stateful.

**OpenEnv compliant ✅**
Typed Pydantic models, reset/step/state, `openenv validate` passes, live HF Space.

### Bottom line
- **Live:** `huggingface.co/spaces/shreyas-garg/[URL]`
- **Code:** `github.com/shreyas-garg/OpenEnv`
- **Reward is verifiable + deterministic** — no LLM-as-judge, no cheatable text scoring

### Closing line
> "Agents don't just need to be smart. They need to keep up."

### Speaker notes
"To close — this is not a research demo. Patronus literally has a bonus prize for environments with schema drift, this is that environment. Scale AI wants long-horizon business workflows, this is that. The env is OpenEnv compliant, deployed, and the reward function is deterministic — no LLM-as-judge to game.

Agents don't just need to be smart. They need to keep up. Thank you."

---

## Q&A PREP (anticipated judge questions)

### Q1: "Isn't this just classification dressed up as RL?"
**A:** "It would be if the correct answer were static. But the same customer email has a **different** correct answer before and after the admin email. No pure classifier can solve that — you need memory across turns. The drift-attention bonus in our reward specifically rewards demonstrating that memory once per drift event."

### Q2: "How do you prevent the agent from cheating the reward?"
**A:** "Three defenses. One: the action space is discrete with numeric parameters — no free text to keyword-stuff. Two: reward is a deterministic lookup table, not an LLM judge. Three: I wrote seven adversarial tests — always-close, always-approve, always-escalate — none score above forty-one percent. Perfect policy hits one hundred. That gap is what the training signal lives in."

### Q3: "Why per-step rewards instead of end-of-episode?"
**A:** "Dense rewards give GRPO a cleaner gradient signal. And each step has a verifiable correct action given current policy — there's no reason to delay the signal."

### Q4: "Why SFT warm-up instead of pure GRPO?"
**A:** "Organizer guidance: pure GRPO from a base model often stalls because the model rarely produces correct rollouts early. SFT on auto-generated perfect labels gives GRPO a non-zero starting distribution to improve from. In our validation run, SFT alone got us zero to fifty percent. GRPO extends that gain."

### Q5: "How scalable is this to more drift types?"
**A:** "Very — it's just adding entries to a Python dict of drift events. I included three types (refund caps, escalation routing, SLA windows) with three variants each. A real deployment would have dozens. The reward function and grader are drift-type-agnostic."

### Q6: "What happens if two drifts contradict?"
**A:** "Most-recent-wins, which matches reality. Admin emails are processed in order; each applies its delta to current policy. The environment handles this in [episodes.py]."

### Q7: "Is the base model actually bad at this, or are you cherry-picking?"
**A:** "Llama 3.1 8B baseline via Groq: 12% on drift-sensitive steps across 25 examples spanning 8 episodes. The per-drift breakdown is even more telling — on drifts that *tighten* rules, base model scores 0%. On drifts that *loosen*, it scores 100% because the loose rule matches its internet prior. That's specification gaming in the wild. Full breakdown in eval_results.json."

### Q8: "Why not use a larger dataset? 500 episodes seems small."
**A:** "We can scale indefinitely — episode generator is deterministic in a seed. I used 800 for onsite and 50 for Colab validation. The real constraint is GRPO step count, not data volume."

### Q9: "What happens after 20 emails?"
**A:** "Episode ends. For the pitch we kept it at 20 — enough to force long-horizon memory (admin at 3, gotcha at 17) without blowing context. Can trivially extend to 50 or 100 on bigger models."

### Q10: "Any Q&A killer: what's the biggest weakness?"
**A:** "The drift-attention bonus is a sparse reward signal — only eight bonus-eligible steps per 100 training examples. In GRPO with K=8 rollouts, some batches see zero bonus opportunities. I'd want to either up-weight bonus-eligible rows in sampling or increase episode length to get more opportunities per sample. That's the first thing I'd fix in v2."

---

## TIMING (total 3:00)

| Slide | Time | Cumulative |
|---|---|---|
| 1: Hook | 0:20 | 0:20 |
| 2: Why matters | 0:30 | 0:50 |
| 3: Environment | 1:00 | 1:50 |
| 4: Results | 0:45 | 2:35 |
| 5: Why we win | 0:25 | 3:00 |

**Practice until you hit 2:55 comfortably.** Leave a buffer.

---

## DESIGN NOTES

- Dark theme, minimal color. Two colors max: one accent for positive (green), one for negative (red).
- Zero walls of text. If a slide has a paragraph, rewrite it as a bullet or drop it.
- Slide 3 should have **no more than 15 words of body copy** — it's a diagram slide.
- Slide 4 should have **zero text other than the headline + chart labels** — the chart is the point.
- Screenshots of actual emails > cartoon icons.
- For the `[X]%` numbers on slide 4, use a bold color (green) to draw the eye.

---

## TO-DO AFTER ONSITE TRAINING

1. Swap `[PRE]%`, `[POST-SFT]%`, `[POST-GRPO]%` on slide 4 with real 3B numbers.
2. Export the matplotlib reward curve to PNG, insert on slide 4.
3. Record a 10-second screencap: same episode, two models side-by-side, showing pre- vs post-training actions.
4. Update HF Space URL on slide 5 once deployed.
