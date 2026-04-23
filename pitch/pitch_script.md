# Pitch Script — 3 min

Read this aloud. Target: **2:50–3:00**. Practice until you don't need to look at it.

---

[**SLIDE 1 — 20s**]

"Imagine you joined a company on Monday. They tell you: refunds over fifty dollars need manager approval. On Tuesday, a customer asks for a seventy-five dollar refund. Do you approve it?

Of course not — you remembered the rule.

Now imagine the AI version of you. It read the internet. The internet says refunds are fine. Nobody told it the new rule. It clicks approve. That's the failure mode I built an environment to solve."

[pause; click to slide 2]

---

[**SLIDE 2 — 30s**]

"Real companies change their rules every week. Stripe updates refund windows. Shopify deprecates APIs. HR systems change leave policies overnight. This is the hardest thing about deploying AI agents — not complexity, not reasoning — it's that **the rules move**.

Every existing OpenEnv environment has static rules. That's the gap I'm filling."

[click]

---

[**SLIDE 3 — 60s**]

"Here's how it works. Every episode is twenty emails. Mostly regular customer tickets — refund requests, outage reports. But at position three, an admin email arrives:

*'Refund cap lowered from one hundred to fifty dollars, effective immediately.'*

A few turns later, a customer asks for a seventy-five dollar refund.

A pre-trained agent applies its internet prior. Approved. Seventy-five dollars gone.

A trained agent remembers the admin email, applies the new rule, and escalates to a manager.

I built **nine different policy drifts** across refund caps, escalation routing, and SLA windows — two of them fire per episode, randomized. The agent never sees the same game twice.

The critical design choice: **the reward is a deterministic lookup table**. Not an LLM judge. I wrote seven adversarial tests — constant-action policies, keyword-spam policies — none score above forty-one percent. A perfect policy hits one hundred. That sixty-point gap is where the training signal lives."

[click]

---

[**SLIDE 4 — 45s**]

"Here's the proof training works. Before any training, the base model gets zero out of eight drift-sensitive questions right. It confidently applies the old rule every time.

One epoch of supervised fine-tuning — using our environment's auto-generated correct-action labels — takes us to [POST-SFT]%.

GRPO on top pushes us to [POST-GRPO]%.

This is the reward curve. Three colored lines — compliance, appropriateness, and the drift-attention bonus — logged independently. You can see which component drives each phase of learning. That separation is what lets us debug and scale."

[click]

---

[**SLIDE 5 — 25s**]

"To close.

This is not a research demo. Patronus literally has a bonus prize for environments with consumer-workflow schema drift. This is that environment. Scale AI wants long-horizon business workflows in HR and IT. This is that too.

The environment is OpenEnv compliant. It's deployed. The reward is deterministic and cannot be gamed.

Agents don't just need to be smart. They need to keep up.

Thank you."

[stop talking. look up. make eye contact for Q&A.]

---

## Delivery notes

- **Pace**: slower than feels natural. 3 min is a lot of time. Don't rush.
- **Eye contact**: look at judges on lines 1, 3.1 (the money shot), and the close. Rest of the time you can glance at the slide.
- **Hands**: if you gesture, gesture wide on "every company" and "schema drift" — big ideas deserve big gestures. Fingers should not be in view if you're not actively gesturing.
- **The pause after slide 3**: let the seventy-five dollar refund joke land. Some judges will smirk. Let them.
- **Numbers**: emphasize each percentage. "Zero... to fifty... to [X]." Slow. Those are the numbers they remember.
- **The last line** ("Agents don't just need to be smart. They need to keep up.") — say it one beat slower than the rest. It's the tagline.

## Mindset

You're not selling a product. You're telling a story about a real problem with a clean technical solution. Judges have seen ten pitches today. What they remember is: **a crisp narrative + one striking number**.

Your crisp narrative: "rules move, agents break, this trains them."
Your striking number: the [POST] drift-sensitive acc percentage.

Deliver both. Leave.
