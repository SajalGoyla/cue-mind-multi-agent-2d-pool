# Decision log

Each entry records one design decision: the context, what we chose, the alternatives, and the consequences. Newest entries go at the bottom. Reference entries by ID (e.g. "D-004") in code comments, commits, and the report.

| ID | Date | Decision | Status |
|---|---|---|---|
| D-001 | 2026-11-20 | Use pooltool as the physics simulator | Accepted |
| D-002 | 2026-11-20 | Own 8-ball rules layer instead of pooltool's `_EightBall` | Accepted |
| D-003 | 2026-11-20 | LLM systems choose among simulated candidates by `shot_id` | Accepted |
| D-004 | 2026-11-20 | Common random numbers for execution noise | Accepted |
| D-005 | 2026-11-20 | Shared, non-LLM break and ball-in-hand policies | Accepted |
| D-006 | 2026-11-20 | Candidate menu: pots × power × spin, safeties, contact fallback | Accepted |
| D-007 | 2026-11-20 | Geometry-only (L0) pruning is round-robin over pot geometries | Accepted |
| D-008 | 2026-11-20 | Deterministic rack and cue ball on the head spot | Accepted |
| D-009 | 2026-11-20 | Policy failures never crash a game: shared fallback rule | Accepted |
| D-010 | 2026-11-20 | Simulation access levels L0/L1/L2 and two-stage pruning | Accepted |
| D-011 | 2026-11-20 | B2 search-bot objective | Accepted (weights provisional until E0) |
| D-012 | 2026-11-20 | Positions shown to LLMs in diamonds and named zones | Accepted |
| D-013 | 2026-11-20 | Assign rack ball numbers ourselves (cross-process determinism) | Accepted |
| D-014 | 2026-11-20 | Retry pooltool failures with a negligible aim nudge | Accepted |
| D-015 | 2026-11-20 | Pin `httpx` to the stable 0.28 line | Accepted |
| D-016 | 2026-11-20 | LLM layer: structured outputs, cache, budget, fake LLM, one retry | Accepted |
| D-017 | 2026-11-20 | Multi-agent graph in LangGraph; coordinator sees all strategy knowledge | Accepted |
| D-018 | 2026-11-20 | Tournament protocol: paired seeds, swapped seats, resumable, process pool | Accepted |
| D-019 | 2026-11-20 | Positions benchmark: frozen annotated menus, oracle EIV, decision-level metrics | Accepted |
| D-020 | 2026-11-20 | Demo: Streamlit + Plotly, offline / live / cache-only modes | Accepted |
| D-021 | 2026-11-20 | Default 10 worker processes (memory bound) | Accepted |
| D-022 | 2026-11-20 | Hand-made hard cases with oracle-drafted, human-reviewed labels | Accepted |
| D-023 | 2026-11-20 | Budget exhaustion stops a game instead of falling back | Accepted |
| D-024 | 2026-11-20 | Keep medium noise as the main setting; decision-level benchmark carries the main comparison | Accepted |
| D-025 | 2026-11-20 | Local open-weight model (Qwen 2.5 7B via Ollama) for the main LLM experiments | Accepted |
| D-026 | 2026-11-20 | LLM experiments at concurrency 2 (Ollama 2 slots, model fully in VRAM) | Accepted |

---

## D-001: Use pooltool as the physics simulator

- **Context.** We need headless, deterministic pool physics with arbitrary ball placement, independent aim/power/spin, and event logs.
- **Decision.** `pooltool-billiards==0.6.0` (Apache-2.0). The feasibility spike (`../cuemind-feasibility/FEASIBILITY_REPORT.md`, outside this repo) verified all required capabilities: ~25 ms per shot, bit-identical determinism, and 98 % precision for its ghost-ball pot helper.
- **Alternatives.** Poolgame-AI-shot-prediction ("Repo B"): rejected (no licence, no physics, poor bank/kick accuracy). Writing our own physics: too slow to build.
- **Consequences.** Panda3D must be installed (from its archive index) even though nothing is rendered. The first simulation in a process compiles Numba kernels (measured 64 s on the first run after install, ~2–3 s afterwards), so every process calls `warm_up()`.

## D-002: Own 8-ball rules layer instead of pooltool's `_EightBall`

- **Context.** The positions benchmark and mid-game experiments start games from arbitrary positions with groups already assigned. pooltool's `_EightBall` cannot be copied (`NotImplementedError`), places the cue ball at a fixed spot after fouls, and does not expose an easy way to set group assignment.
- **Decision.** `src/cuemind/game/rules.py` implements a simplified WPA-style called-shot 8-ball (rules listed in the module docstring and in `architecture.md`). pooltool is used only for physics and events. `judge()` is a pure function of (state, events, call), so it is fully unit-tested with synthetic events.
- **Alternatives.** Wrap `_EightBall` (the plan's first option): hard to start from mid-game states, cannot be copied for hypothetical shots.
- **Consequences.** Our rules differ from official WPA rules in documented ways: no push-outs; the table stays open after the break; an 8 pocketed on the break is re-spotted; the turn continues only when the *called* ball drops in the *called* pocket; a safety always ends the turn. The same rules apply to every system, so the comparison stays fair. Event extraction uses exact ball ids (avoids pooltool's substring quirk where `"1"` matches `"11"`).

## D-003: LLM systems choose among simulated candidates by `shot_id`

- **Context.** LLMs that output raw angle/power pocket almost nothing (PoolBench).
- **Decision.** Code generates and simulates candidate shots; every policy (bot or LLM) returns a `shot_id` from the menu.
- **Consequences.** Illegal or nonsensical actions are impossible by construction; the only difference between systems is which candidate they choose. The generator's quality caps every system (a limitation to report).

## D-004: Common random numbers for execution noise

- **Context.** Without noise, simulate-then-pick is an oracle and the second player wins every game (spike, 4/4). With noise, luck adds variance that can swamp differences between systems.
- **Decision.** Noise draws come from `rng_for(seed, shot_index, "execute")` (`sim/noise.py`). Each draw always consumes exactly four normals, so streams stay aligned. Presets: none / low (0.25°, 3 %, 0.01 R) / medium (0.5°, 5 %, 0.02 R) / high (1.0°, 10 %, 0.04 R).
- **Consequences.** Two systems choosing the same shot at the same point of the same seeded game get the same execution error. Paired games with swapped seats then compare decisions, not luck.

## D-005: Shared, non-LLM break and ball-in-hand policies

- **Context.** The research question is about shot selection. Placement skill and break quality would confound it.
- **Decision.** Break: aim at the apex ball at 8 m/s with ±1° seeded jitter. Ball in hand: evaluate a 5 × 9 grid of free spots and place the cue ball where the best direct pot is geometrically easiest. Both are deterministic and identical for every system.
- **Consequences.** Ball-in-hand placement costs ~45 pot-geometry evaluations (≈ 0.1–1.4 s depending on how many balls remain).

## D-006: Candidate menu

- **Decision.** Pots: each viable direct pot (pooltool `ai.pot`, max cut 80°) × power {soft 1.5, medium 2.5, firm 4.0 m/s} × spin {draw b = −0.3, stun 0, follow +0.3}. Safeties: contact shots at cut {0°, ±35°} × {soft, medium} on each legal ball, kept only if a noise-free simulation is legal. Contact-only fallback: aim at each legal ball's centre, used only when nothing else exists.
- **Alternatives.** Banks/kicks: postponed (cut-line in the plan; mirror-reflection banks potted only 1–13 % in the spike).
- **Consequences.** Safety generation runs up to 6 simulations per legal ball (≈ 0.2–1 s). Safety validity uses the simulator even at access level L0; no simulation *statistics* are shown at L0.

## D-007: Geometry-only (L0) pruning is round-robin over pot geometries

- **Context.** A first version kept the first 12 pot variants, and since each geometry has 9 variants, the menu covered only about one target ball (caught while writing the code).
- **Decision.** At L0, variants are taken round-robin over pot geometries (ordered by geometric ease) using a fixed variant priority (medium-stun first), up to 12 pots + 3 safeties. With simulation statistics (L1/L2), pruning uses `p_pot` instead (D-010).
- **Consequences.** The L0 menu covers up to 12 different target/pocket choices.

## D-008: Deterministic rack and cue ball on the head spot

- **Context.** pooltool's rack shuffles ball order with Python's `random` and jitters positions with NumPy's global RNG, and places the cue ball randomly behind the head string, so "same seed" did not give the same rack.
- **Decision.** `rack_state(seed)` seeds both RNGs and puts the cue ball on the head spot (W/2, L/4).
- **Consequences.** A seed fully determines the opening position, which paired games need.

## D-009: Policy failures never crash a game

- **Decision.** If a policy raises or returns an unknown `shot_id`, the runner uses the shared fallback rule (highest `p_pot` if available, else highest geometric ease; pots before safeties) and logs `fallback=True` with the reason. Unexpected engine errors end the game with `status="error"` and the traceback in the result.
- **Consequences.** Fallback rates are a reported metric per system (fairness contract item 7).

## D-010: Simulation access levels and two-stage pruning

- **Context.** RQ3 varies how much simulation information the decision maker gets. Annotating every generated candidate with rollouts is too slow (≈ 100 pot variants × 8 rollouts per decision).
- **Decision.** Three levels (`candidates/annotate.py`, `pipeline.py`):
  - **L0** geometry only; **L1** + one noise-free preview (pots?, scratch?, foul, cue end, number of next pots); **L2** + `n_rollouts = 8` noisy rollouts using the execution noise model (the player knows their own skill).
  - L2 statistics: `p_pot`, `p_scratch`, `p_foul`, `p_keep_turn`; `next_ease` = mean ease of my best next pot *over rollouts where I keep the turn*; `opp_threat` = mean ease of the opponent's best pot *over rollouts where the turn passes*, with a foul (ball in hand) or a lost game counted as 1.0. Conditional means keep the quantities interpretable on their own.
  - Two-stage pruning at L1/L2: a geometry-ordered working set (18 pots round-robin + 4 safeties) is simulated, then the menu keeps pots with `p_pot ≥ 0.10`, the best variant per (target, pocket, spin), the top 12 by `p_pot`, and the 3 safeties with the lowest `opp_threat + p_foul`. If fewer than 2 candidates survive, the best remaining working candidates are added so there is always a choice.
  - Rollout random streams are keyed by (game seed, shot index, candidate key, rollout index), so statistics do not depend on menu order (tested).
- **Alternatives.** Rollouts on all candidates (too slow); pruning by B2's objective (would pre-select B2's answer and bias the comparison).
- **Consequences.** Measured ≈ 2–3 s per L2 decision on a 6-ball table, single-threaded; games are parallelised across processes (32 logical CPUs available). With n = 8, probabilities come in steps of 12.5 %; the oracle (Phase 4) uses more rollouts.

## D-011: B2 search-bot objective

- **Decision.** `EV = p_pot·(1 + 0.5·next_ease) − 1.5·p_scratch − 1.0·(p_foul − p_scratch) − 0.6·(1 − p_keep_turn)·opp_threat` over the same annotated menu the LLMs see (`policies/search_bot.py`). Scratches are counted once (the foul term excludes them).
- **Consequences.** B2 is a strong, transparent non-LLM baseline with identical information. Weights may be adjusted once on dev games during E0 (so that B2 clearly beats B1), then frozen; any change is logged here.

## D-012: Positions shown to LLMs in diamonds and named zones

- **Context.** LLMs reason poorly over raw metre coordinates.
- **Decision.** `candidates/describe.py` renders positions as diamonds (x 0–4 across the short rail, y 0–8 along the long rail) plus a zone name ("upper half, on the right rail"), and gives each candidate a one-line description containing only the information allowed at the current access level. Ease values are also given as words (easy ≥ 0.7, moderate ≥ 0.5, hard > 0).
- **Consequences.** The same text is used by SA and MA (fairness), and it is shown in the demo UI for explainability.

## D-013: Assign rack ball numbers ourselves (cross-process determinism)

- **Context.** In the first 20-game run, replaying a seed in a new process produced a different game. Bisecting showed the rack's ball *positions* were identical but the ball *numbers* differed: pooltool picks numbers with `random.choice(list(remaining))` where `remaining` is a `set`, and set iteration order depends on Python's per-process hash seed (`PYTHONHASHSEED`). Paired games and resumable experiments require exact reproducibility.
- **Decision.** `rack_state(seed)` keeps pooltool's (seeded) positions but assigns numbers itself with `random.Random(f"rack-{seed}")`: the 8 in the middle of the third row, one solid and one stripe in the back corners, the rest shuffled.
- **Verification.** Full L2 games are now bit-identical under `PYTHONHASHSEED` = 1, 2, 3; a subprocess regression test pins the rack hash across hash seeds.
- **Consequences.** Runs before this fix (only `results/phase1/`) used racks that cannot be reproduced exactly; they were only a smoke test.

## D-014: Retry pooltool failures with a negligible aim nudge

- **Context.** One of 20 games ended with an internal `AssertionError` raised inside `pt.simulate` (degenerate collision geometry) during safety generation.
- **Decision.** `sim.execute.simulate` retries with the aim nudged by ±1e-4° and 3e-4° (far below any execution noise) and raises `SimulationError` if all attempts fail. Safety generation drops such a candidate; a failed L2 rollout is scored conservatively as a foul (turn over, threat 1.0). The real executed shot in a game goes through the same retry; a total failure ends the game with `status="error"` (counted and reported).
- **Update (E0).** A second pooltool failure mode appeared in E0 (1 of the first 42 games): `ZeroDivisionError` inside the pot-occlusion helpers when a ball sits in a pocket mouth, plus NaN aim angles from zero-length vectors. `direct_pots` now rejects such (ball, pocket) pairs as `degenerate_geometry` and drops any pot with a non-finite angle or cut. Regression test added.
- **Consequences.** Games no longer abort on a single pooltool failure. `simulate` now always returns a simulated copy and never mutates its input.

## D-015: Pin `httpx` to the stable 0.28 line

- **Context.** Installing pooltool requires `--prerelease=allow` (its Panda3D pin is a dev build). That flag also pulled `httpx==1.0.dev6`, which breaks `langgraph-sdk` (`httpx.HTTPStatusError` no longer exists). The Anthropic SDK uses its own `httpx2` package and is unaffected.
- **Decision.** Pin `httpx==0.28.1` in `requirements.txt`.

## D-016: LLM layer design

- **Decision.**
  - `llm/client.py`: `ClaudeVertexClient` calls `messages.parse(output_format=<pydantic schema>, output_config={"effort": ...})` with the system prompt marked `cache_control: ephemeral` (prompt caching). It checks `stop_reason` (`refusal`, `max_tokens` → error), catches `RateLimitError` / `APIStatusError` / `APIConnectionError` separately, and records input, output, cache-read and cache-write tokens and an estimated cost (`experiments/configs/prices.yaml`; Vertex prices must be confirmed).
  - `llm/cache.py`: SQLite response cache keyed by SHA-256 of (model, effort, system, user, schema, repeat index). Failures are never cached. The same file holds a spending ledger: `Budget(run_id, max_usd)` is shared by all worker processes and stops a run cleanly at the limit (cached calls are free, so runs resume).
  - `FakeLLM` (deterministic, reads the candidate table) powers every test and an offline demo mode; it can also emit invalid ids or no output to test failure handling.
  - `agents/common.call_with_id_check`: if the returned `shot_id` is not in the menu (or nothing parsed), the model is asked once more with the list of valid ids; after that the shared fallback rule applies (D-009).
  - Prompts live in `llm/prompts/*.md`; `promptlib.prompt_version()` hashes them and every decision logs the hash.
- **Alternatives.** LangChain's `ChatAnthropicVertex`: less direct control over caching, token accounting and retries. JSON-in-text parsing: less reliable than structured outputs.
- **Consequences.** Temperature cannot be set on these models, so live outputs are not bit-reproducible; the cache makes replays exact and SA-vote uses `repeat_idx` 0–3 to draw independent samples.

## D-017: Multi-agent graph in LangGraph; coordinator sees all strategy knowledge

- **Decision.** `agents/graph_multi.py` builds `START → {offense, position, safety} → coordinator → (critic → coordinator_retry) → END`. Specialists run in one LangGraph superstep (concurrently) with only their own strategy block; reports are merged with a dict-union reducer. The coordinator's system prompt contains the rules and all three strategy blocks (it must weigh them); the single agent's prompt contains exactly the same blocks, so MA never has knowledge that SA lacks (tested). The critic may veto once. Specialists can be removed for ablations (`ma-no-offense`, etc.).
- **Consequences.** MA makes 4 calls per decision (5 with the critic, plus any id-retry). SA-vote (4 samples, majority vote, ties by mean confidence) is the compute-matched control.

## D-018: Tournament protocol

- **Decision.** `eval/tournament.py`: for each seed and pairing (X, Y), X breaks in one game and Y in the other; both games share rack, break jitter, rollout and noise streams. Games run in a process pool (each worker warms the JIT once); each finished game appends one summary line to `results/<exp>/games.jsonl` (with per-seat metrics), and full shot logs go to `results/<exp>/raw/` (git-ignored). Re-running skips finished games and retries errored ones.
- **Statistics** (`eval/stats.py`): Wilson CI for win rates; paired seed score (0–2, minus 1) with a bootstrap CI and a two-sided sign test; Wilcoxon signed-rank with rank-biserial effect size and Holm correction for decision-level comparisons.

## D-019: Positions benchmark

- **Context.** Full games are expensive with LLMs and noisy (a 50-seed tournament can only detect large win-rate gaps). The plan puts most statistical weight on a decision-level benchmark.
- **Decision.** `eval/positions.py`, `eval/oracle.py`, `experiments/build_dataset.py`, `experiments/run_positions.py`:
  - Positions come from B2-vs-B1 games (2 per game) plus 20 % random valid layouts; dev 60 / test 180, test stratified by difficulty.
  - Each position stores **one frozen menu** built at L2 (candidates keep geometry, preview and rollouts). For L0/L1 runs the same menu is re-described with fields hidden, so access levels differ only in information, not in the shots on offer.
  - **Oracle (EIV):** 24 noisy executions per candidate; +1 per own ball legally pocketed, inning continued by the B1 rule for up to 2 shots, −1 per foul, −0.75 × opponent threat when the turn passes, ±5 for winning/losing on the 8. B1 (not B2) continues the inning so the oracle does not agree with B2 by construction.
  - Metrics per decision: regret (best EIV − chosen EIV), top-1 (within one standard error of the best), normalised score (0 = random choice, 1 = best), plus fallback rate, calls, tokens, cost and latency; broken down by difficulty and situation tag.
  - `DecisionContext.sample_idx` gives independent LLM samples for the decision-variance experiment (E10) without disturbing the cache of the main runs.
- **Consequences.** The oracle is an approximation of game value; position-level regret will be checked against game outcomes (E2) and the limitation reported.

## D-020: Demo design

- **Decision.** `demo/app.py` (Streamlit + Plotly), three tabs: *Decide a shot* (pick a demo or benchmark position, see the candidate menu drawn on the table, run SA and MA side by side with specialist cards, oracle panel for benchmark positions, execute the chosen shot with noise and watch the animation), *Replay a game* (step through a logged game with each decision's rationale and specialist reports; animate any shot), *Results* (summary tables and figures). Modes: offline fake LLM (no network), live Claude with the response cache (cached answers are labelled), cache-only.
- **Reliability.** Errors are shown as messages, never stack traces; replay and results tabs work without network; a smoke test (`tests/smoke/test_demo_app.py`, Streamlit `AppTest`) runs the app offline end to end.

## D-021: Default 10 worker processes

- **Context.** The development machine has 32 logical CPUs but 15.7 GB RAM. E0 with 24 workers left < 1 GB free, and concurrently started processes failed with "paging file is too small" / `MemoryError`.
- **Decision.** Configs default to 10 workers (each ~0.5 GB). Increase only on machines with more memory.

## D-022: Hand-made hard cases

- **Decision.** `data/positions/hard_cases.yaml` describes positions in diamond coordinates with a category (safety-only, shape trap, endgame, cluster, scratch risk, ball in hand) and the intended lesson. `experiments/build_hard_cases.py` builds benchmark records (48 oracle rollouts each), renders a PNG and an oracle table per case into `data/positions/hard_cases/`, and *drafts* labels: expected intent = the best-EIV shot's type; acceptable shots = within one standard error of the best. Labels stay `draft` until a team member reviews them and sets `reviewed_by`.
- **Finding while drafting.** A position where the only legal ball is fully snookered (every straight line blocked) yields a one-option menu (the contact fallback, which fouls): the generator has **no kick shots** (cut-line, D-006). Such positions are excluded from the benchmark by the ≥ 2-candidate rule; in games they force a foul. This is a generator limitation to report (failure category "best shot not in menu").
- **Consequences.** 9 draft cases exist (2 safety-only, 2 shape, 2 endgame, 1 cluster, 1 scratch risk, 1 ball in hand). The plan asks for ~25; more can be added to the YAML.

## D-023: Budget exhaustion stops a game

- **Context.** The runner turns any policy exception into the fallback shot (D-009). If the budget ran out mid-game, the rest of the game would silently be played by the fallback rule.
- **Decision.** `BudgetExceeded` is re-raised by the runner; the game (or tournament job) stops and is resumed later from the response cache. Tested.

## D-024: Main noise level and where the statistical weight goes

- **Evidence (E0, E0b; `results/summary.md`).** B2 (search) vs B1 (geometric), paired seeds, both seat orders: low noise 53 % [41, 65] (60 games), medium 54 % [44, 63] (100 games), high 52 % [39, 64] (60 games); none significant. B2 pots more often at low and medium noise (0.75 vs 0.69; 0.61 vs 0.53). B2 beats random play clearly (79 % at medium).
- **Interpretation.** Under these rules and this menu, better shot selection changes game outcomes only slightly; outcomes are dominated by execution luck and by who gets the table in a good layout. Lowering the noise does not make skill differences show up in win rates, so there is no reason to move away from the planned medium setting (a plausible amateur).
- **Decision.** Keep medium noise for the dataset, the oracle and all main experiments. Treat the positions benchmark (regret against the oracle) as the primary evidence for RQ1/RQ2; report game win rates (E2) as secondary evidence with honest power statements (with ~50 games per pairing only very large gaps are detectable).
- **Also measured.** Profiling one L2 decision (8 balls): 222 pooltool simulations account for ~85 % of the time (~34 ms each); our code's overhead is small. Speed-ups would require fewer rollouts or more processes, and process count is limited by RAM (D-021), so settings stay as planned.

## D-025: Local open-weight model for the main LLM experiments

- **Context.** Claude on Vertex AI returns HTTP 429 (quota) for every call on the course project, and the team asked to save Google Cloud credits for later. The original proposal planned to run most experiments on an open-weight model and confirm key results on an API model.
- **Decision.** Main LLM experiments use **Qwen 2.5 7B Instruct (Q4_K_M, 7.6 B parameters) served locally by Ollama 0.40** on the laptop's RTX 4060 (8 GB). `llm/client.py: OllamaClient` uses `/api/chat` with the pydantic JSON schema as `format` (constrained decoding), temperature 0.7 for every system, `seed = 1000 + repeat_idx` (independent but reproducible samples for SA-vote and E10), context 8192 tokens, and the same response cache. The server runs with `OLLAMA_NUM_PARALLEL=4` so MA's three specialists are decoded concurrently. `CUEMIND_LLM=ollama` is now the default backend; Claude remains available (`llm: claude`) for the confirmation experiment (E8) once quota exists.
- **Measured (first live check).** SA ≈ 3.7 s and ~2.3 k input tokens per decision; MA ≈ 32 s, 4 calls, ~9.4 k input and ~1.4 k output tokens per decision; the first call of a session pays ~70 s to load the model. Cost is zero; tokens and latency are still reported, and the cost analysis for RQ2 uses tokens and latency (API prices can be applied to the token counts afterwards).
- **Observed quirk.** The model sometimes gives confidence on a 0–100 scale; `clamp01` reads values in (1, 100] as percentages.
- **Consequences.** Results describe a small open model; whether they transfer to a frontier model is tested later with Claude (E8) and stated as a limitation until then.

## D-026: Concurrency 2 for LLM experiments

- **Context.** The first experiment queue (E4 onwards) was stopped by Claude Code's memory guard when free RAM fell critically low. With `OLLAMA_NUM_PARALLEL=4` and an 8192-token context the model needs 6.8 GB, of which ~0.6 GB spilled from the 8 GB GPU into system RAM, on top of 4 concurrent decision threads.
- **Decision.** Ollama runs with 2 parallel slots (model 5.5 GB, entirely in VRAM) and `scripts/run_queue.py --concurrency 2` passes 2 threads to positions runs and 2 workers to tournaments. Measured: 8 GB RAM free under load.
- **Consequences.** Lower throughput per run but no memory pressure. Per-decision latencies reported from E1 (4-way load) and later runs (2-way load) are not directly comparable; latency comparisons are made within one run.
