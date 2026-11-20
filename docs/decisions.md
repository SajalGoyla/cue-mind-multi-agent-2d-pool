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
- **Consequences.** Games no longer abort on a single pooltool failure. `simulate` now always returns a simulated copy and never mutates its input.
