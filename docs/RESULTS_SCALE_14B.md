# Model-size check results: Qwen 2.5 14B vs 3B

Plan and predictions, written before any 14B game: [`PLAN_SCALE_14B.md`](PLAN_SCALE_14B.md).
Data: `results/v2_scale14b/` (14B) and `results/v2_main/` (3B, zero-noise half). Numbers below come
from `python experiments/analyze_scale14b.py`.

**Setup.** Only the model changed. Prompts (`prompts-v2`), menu, the reference player Stat, seeds
9000–9029, both seat orders and zero playing noise are identical to the 3B run. Each system played
60 games against Stat. The 120 games had 0 errors, 0 LLM fallbacks and 0 games stopped by the shot cap.

## 1. Win rate against Stat

| Model | SA (1 LLM call) | MA (team) |
|---|---|---|
| 3B | 13/60 = **21.7 %** [13.1, 33.6] | 15/60 = **25.0 %** [15.8, 37.2] |
| 14B | 19/60 = **31.7 %** [21.3, 44.2] | 21/60 = **35.0 %** [24.2, 47.6] |

95 % Wilson intervals in brackets.

## 2. Does the team help? (MA − SA, paired per seed)

Each seed gives two games against Stat, so per seed a system wins 0, 1 or 2. The table shows the difference, MA minus SA, averaged over the 30 seeds.

| Model | MA − SA (wins per seed) | 95 % bootstrap CI | Seeds MA better / equal / worse | Sign test |
|---|---|---|---|---|
| 3B | +0.067 | [−0.17, +0.30] | 7 / 18 / 5 | p = 0.77 |
| 14B | +0.067 | [−0.20, +0.33] | 9 / 15 / 6 | p = 0.61 |

The model effect, 14B − 3B paired per seed for the same system:

- SA: +0.20 wins per seed [−0.10, +0.47], p = 0.14
- MA: +0.20 wins per seed [−0.10, +0.50], p = 0.36

## 3. Cost per decision

| | Calls | Tokens in / out | Seconds (mean, uncached) |
|---|---|---|---|
| 3B SA | 1.02 | 1859 / 103 | 1.9 |
| 3B MA | 3.93 | 6802 / 605 | 8.2 |
| 14B SA | 1.00 | 1837 / 76 | 3.6 |
| 14B MA | 3.56 | 6171 / 503 | 21.5 |

Seconds are comparable within a model, not across models: 3B ran on an RTX 4060 laptop, 14B on a
cloud L4. At 14B the three specialists agreed on their first choice in 44 % of MA decisions (27 %
at 3B). When they agree, the coordinator call is skipped, so MA averages fewer calls at 14B.

## Predictions vs outcome

1. **Team effect stays within ±0.16 wins per seed at 14B. Held.** MA − SA is +0.067 at 14B,
   identical to 3B, and the interval includes zero. The objection "3B is too weak to use the
   specialist roles" is not supported: the larger model plays better, but the team does not add
   more on top of it. The interval is wide, though. With 30 seeds we cannot rule out a team
   effect of up to +0.33 wins per seed (about +17 points of win rate).
2. **Both systems win more at 14B. Right direction, not significant.** Both SA and MA gain
   +10 points (+0.20 wins per seed), but both paired intervals include zero.

**What this means for the report.** Going from 3B to 14B adds about +10 points of win rate. Adding
the team adds about +3 points, and does so at both sizes. One 14B call (SA, 31.7 %) wins more often
than the four-call 3B team (25.0 %). Extra compute looks better spent on a larger model than on
more agents. This is a direction only: no difference here is significant at 30 seeds. Stat still
wins about two games in three against every LLM system.

## Run notes

- One Google Cloud spot VM (`g2-standard-8`, 1 × L4, `us-central1-c`), project `cuemind-516017`,
  2026-11-27. Ollama `qwen2.5:14b-instruct`, 2 parallel slots, 2 game workers. All cloud resources
  were deleted after the run. Estimated cost is about $3; exact billing appears about a day later.
- The VM was preempted twice and the run resumed each time. Finished games are kept, and a game
  cut off mid-play is replayed from scratch. Decisions are seeded and cached by prompt, so a replay
  makes the same decisions. Seed 9000 repeats the 4-game smoke test, so its decisions came from the
  response cache. Seconds per decision therefore use uncached decisions only (338 of 350 for SA,
  334 of 356 for MA).
- Code at commit `37e892d` (the plan commit) on the VM; no code or prompt changes during the run.
  Raw shot logs: `results/raw_logs_scale14b.zip`.
