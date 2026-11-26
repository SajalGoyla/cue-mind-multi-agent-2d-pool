# CueMind v2 plan (approved 2026-11-25)

v2 replaces the v1 positions benchmark (tag `v1-positions-benchmark`) with a games-only study. Decision record: D-029 in `docs/decisions.md`. Branch: `v2-games`.

## Why v1 was dropped (measured on the v1 data)

- **Circular benchmark.** The oracle graded with more runs of the same simulator and noise model, using the same ingredients the LLMs and B2 read; B2's score correlated 0.70 with oracle values over 1,460 test candidates.
- **Too noisy for close calls.** Median oracle SE 0.20; median best-vs-second gap 0.18; the best shot was clearly separated in only 17 of 180 positions. The SA–MA gap (0.013) was far below this and below sample-to-sample variation (~0.1, E10).
- **"L0" was not simulation-free.** The menu and its order came from simulation (at L0 SA picked S01 in 52 % of positions, B2 in 87 %).
- **Specialists misbehaved.** The safety specialist recommended a safety in 96 % of positions. (The plan's claim that offense and position gave identical scores held in only 9 of 180 E1 decisions.)
- **Games.** Bot-vs-bot win rates barely moved with policy (B2 vs B1 52–54 %); 0 of 469 games hit the 150-shot cap; ~21 shots per game.

## Research question

Given the same shot menu and no simulated forecasts, does a team of specialised LLM agents play better eight-ball than a single LLM agent, measured by full games against the same simulation-based reference player (Stat)? Secondary: what does the team cost in calls, tokens and latency? **Expected outcome stated up front:** with 60 games per pairing only large differences are detectable (95 % CI about ±13 points), so "no detectable difference" is the most likely result.

## Systems (same model for all LLM agents: Qwen 2.5 3B Instruct via Ollama, temperature 0.7, seeded)

| System | How it chooses |
|---|---|
| **Stat** (reference) | Uses the forecasts attached to the menu (3 noisy simulations per option at low noise, averaged) and the D-011 objective (unchanged). It has more information than the LLMs, stated in the report |
| **SA** | One call: rules + offense, position and safety knowledge + the menu → `shot_id`, `rationale` |
| **MA v2** | Offense, Position, Safety specialists (rules + own knowledge + menu) each return a ranked shortlist of 3 ids with ≤ 20-word reasons; if all three rank the same shot first it is played without the coordinator; otherwise the coordinator (rules + all knowledge + menu + nominations) chooses from the union → `shot_id`, `rationale`, `followed` |
| **B1** | Easiest pot, medium stun (pilot only) |

Fairness: SA's prompt contains every strategy block the specialists see (tested); same model, temperature and seeds; the same geometry-only fallback rule when an answer is unusable.

## Shot menu (all players receive the same menu for the same state and decision)

1. Every viable direct pot for the shooter's legal balls (ghost-ball geometry) × 3 speeds (soft, medium, firm) × 3 spins (draw, stun, follow) = 9 variants per pot, plus legal safeties (the noise-free legality check is a validity filter only; its result is never shown).
2. Every variant is forecast 3 times at low noise and averaged.
3. Keep at most 15: the best variant of every legal ball that has a pot first (all of the team's balls are represented), then the best remaining variants up to 12 pots, then up to 3 safeties.
4. Shuffle with a per-decision seed (`rng_for(seed, shot_index, "menu")`) so the order carries no information.
5. LLMs see one line per option: ball, pocket, speed, spin, cut angle, ease word. **No forecast numbers.** The menu *selection* uses simulation; the report says so.

## Experiments

| Step | What | Pass / decision rule |
|---|---|---|
| 1. Pilot (bots only) | Stat vs B1 (seats swapped) and Stat vs Stat (one game per seed), 30 seeds, playing noise `none` and `low` | Stat vs B1 ≥ 60 % with the Wilson CI above 50 % shows that decisions matter. Both noise conditions are run in the main study regardless (team decision); a failing condition is reported as such |
| 1b. Floor check | Stat vs SA, 5 seeds × 2 seats, both noise levels | If SA wins < 10 %, SA vs MA is undetectable in that condition: report it |
| 2. Prompt development | Stat vs SA and Stat vs MA on 3 dev seeds (low noise), at most 2 rounds; then freeze `prompts-v2` | Dev seeds 8500+ never overlap main seeds |
| 3. Main | Stat vs Stat, Stat vs SA, Stat vs MA on the same 30 seeds (9000–9029), seats swapped, common random numbers, playing noise `low` and `none` | — |

## Metrics (only these)

- **Primary:** win rate against Stat for SA and MA (Wilson 95 % CI); paired SA vs MA per seed (each system's score over its two games against Stat; bootstrap CI of the difference, sign test).
- **Secondary (same games):** pot success, fouls per game, balls potted per inning, game length, safety rate, fallback rate, LLM calls / tokens / latency per decision, MA unanimity rate.
- **Qualitative:** 3–4 moments from real game logs (candidates listed automatically in `results/v2_main/moments.md`).

## Cut list (done on the branch; everything remains in the v1 tag)

Removed: positions benchmark and oracle, L1/L2 descriptions for LLMs, SA-vote, critic, MA ablations, B0, the Claude/Vertex client and budget guard, v1 configs, data and results, v1 experiment scripts. Kept: `sim/`, `game/`, candidate generation (+ forecasting as Stat's tool), Stat, B1, SA, MA v2, tournament runner, stats, the response cache, the Ollama client, the queue runner with the Ollama watchdog and a keep-awake request. v1 report drafts moved to `docs/archive_v1/`.

Fixes: the state text now matches the rules (only pocketing the 8 with a foul, or in the wrong pocket, or early, loses); the menu order encodes nothing; demo games will be committed to `demo/games/`.

## Optional later (after the main results)

LoRA: one adapter imitating Stat's choices from B-vs-B games (raw menu → Stat's shot), applied to SA and the coordinator; rerun Stat vs SA-LoRA and Stat vs MA-LoRA. Report only.

## Top risks

1. **Zero playing noise lets seat order decide games** (quick check: Stat vs B1 50 %, breaker 70 %). Mitigation: both conditions are reported; low noise is the main condition.
2. **Floor effect** (Stat too strong). Mitigation: the floor check; report honestly.
3. **Low power** (±13 points at 60 games). Mitigation: paired design with common random numbers; the null result is stated as a hypothesis in advance.
4. **The 3B model ignores instructions** (invalid ids, poor shortlists). Mitigation: constrained JSON output, one corrective retry, logged fallbacks; two prompt rounds on dev seeds.
5. **Long runs stopped by memory pressure or sleep.** Mitigation: detached process, Ollama watchdog (D-028), keep-awake request, resumable runs.
