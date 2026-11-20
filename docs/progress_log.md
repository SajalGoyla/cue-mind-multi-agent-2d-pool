# Progress log

One entry per working session: what was done, how it was tested, where outputs are, open problems. Newest at the bottom.

## 2026-11-20: Phase 0 and Phase 1 (scaffold, game engine)

**Done**
- Repo scaffold per the plan: `src/cuemind/` package (`pyproject.toml`, editable install), `tests/`, `docs/`, `.env.example`.
- Environment: Python 3.12 venv via uv; pooltool 0.6.0, anthropic 2.12.0, langgraph 1.2.24, pydantic 2.14, streamlit 1.67 (pinned in `requirements.txt`).
- Simulation layer: `TableState`, validation, hashing, deterministic rack; stroke execution and event extraction; noise model with common random numbers.
- Game engine: own 8-ball rules layer (D-002), seeded break, shared ball-in-hand placement, game runner with JSONL logging and fallback handling.
- Candidates (geometry level L0): direct pots × power × spin, safeties, contact fallback, round-robin pruning (D-007).
- Baselines B0 (random) and B1 (geometric).

**Tested**
- `pytest -q`: 26 tests (unit + integration) pass.
- CLI: `python -m cuemind.game.runner --p1 b1 --p2 b1 --games 2` played two full games to a legal 8-ball finish (19 and 24 shots, ~15 s each, ~0.7 s per shot).
- 20-game B1-vs-B1 run at medium noise: see `results/phase1/` (summary appended below when finished).

**Measured**
- First pooltool simulation after install: 64 s (Numba JIT); later processes ~2–3 s.
- Direct-pot geometry when warm: ~1 ms (2 balls), ~30 ms (15 balls).

**Issues found and fixed**
- pooltool rack was not reproducible for a seed (uses both `random` and NumPy RNGs) → seed both, fixed cue ball on the head spot (D-008).
- First pruning version covered only ~1 target ball → round-robin pruning (D-007).
- Group assignment initially used the first potted ball instead of the called ball → fixed and unit-tested.

**20-game B1-vs-B1 smoke run (medium noise, L0)**: `results/phase1/b1_vs_b1_20games.jsonl`
- 19/20 finished with a legal 8-ball ending; 1 ended with a pooltool-internal `AssertionError` (fixed, D-014).
- Mean 20.2 shots per game (min 3, max 35); ~16 s per game.
- Fouls over 20 games: 27 scratches, 16 wrong first contact, 1 scratch on the break, 1 no contact.
- Replaying the failing seed in a new process gave a different game, which exposed cross-process non-determinism in the rack (fixed, D-013). This run is a smoke test only; its racks are not exactly reproducible.

## 2026-11-20: Phase 2 (annotations, pruning, descriptions, B2) and robustness fixes

**Done**
- Access levels L1 (preview) and L2 (8 noisy rollouts) with two-stage pruning (D-010).
- Plain-language state summaries and candidate descriptions in diamond coordinates (D-012).
- B2 search bot (D-011); policy registry names `b0`, `b1`, `b2`.
- Deterministic rack numbering across processes (D-013); pooltool failure retry (D-014).

**Tested**
- `pytest -q`: 38 tests pass. New: rollout probability validity, determinism and order independence, a hanging-ball sanity check, access-level text, B2 = argmax EV, fallback rule, a short L2 game, rack hash across `PYTHONHASHSEED` values (subprocess), simulate retry/give-up, failed rollouts scored as fouls.
- Same L2 game bit-identical under `PYTHONHASHSEED` 1, 2, 3.

**Measured**
- L2 decision preparation ≈ 2.2 s on a 6-ball table (6 rollouts), single-threaded.

**Lesson (tooling)**
- On Windows, Python's default file encoding is cp1252: helper scripts must pass `encoding="utf-8"` (or run with `PYTHONUTF8=1`). One doc was truncated by this and rewritten; all files were re-checked as valid UTF-8.

## 2026-11-20: Phase 3 infrastructure and Phase 5 LLM layer

**Done**
- Tournament runner (D-018) with experiment configs `experiments/configs/e0.yaml`, `e0_noise.yaml` and CLI `experiments/run_tournament.py`; statistics and metrics modules.
- LLM layer (D-016): Vertex client with structured outputs, prompt caching, response cache, cross-process budget guard, FakeLLM.
- Prompts (`llm/prompts/*.md`), SA, SA-vote, MA (LangGraph, D-017), MA+critic, MA ablations; all registered by name.
- `scripts/check_llm.py` live check; local `.env` (git-ignored) points to the existing service-account key by path.

**Tested**
- 57 tests pass offline, including prompt fairness (SA prompt contains every specialist's knowledge; each specialist only its own), retry → fallback, SA-vote sampling, MA fan-out/coordination, ablation and critic paths, cache hit/offline mode, budget guard, cost formula, tournament run + resume + summary, stats against known values, degenerate-geometry regression.

**Blocked: live LLM calls**
- `scripts/check_llm.py` fails with HTTP 429 `RESOURCE_EXHAUSTED: Quota exceeded for aiplatform.googleapis.com/global_online_prediction_requests_per_base_model` for both `anthropic-claude-haiku` and `anthropic-claude-sonnet` on the `global` endpoint. Regional endpoints (`us-east5`, `europe-west1`) return 404 (model not enabled). Action for the team: request a quota increase for Claude on Vertex AI in the GCP console (IAM & Admin → Quotas, metric `global_online_prediction_requests_per_base_model`) or enable the models in a region, then re-run `python scripts/check_llm.py`.

**E0 (running)**
- 300 games (B2–B1, B2–B0, B1–B0; 50 paired seeds; medium noise; L2). One early game hit a pooltool `ZeroDivisionError` (fixed, D-014 update); re-running the command re-plays errored games.

## 2026-11-20: Phase 4 (positions benchmark) and Phase 8 (demo) code

**Done**
- Oracle (EIV) and positions benchmark code (D-019); dataset builder and positions runner CLIs; configs for E1, E4, E6, E10 and a dev-split pilot.
- `sim/trajectories.py` and `viz/table_plot.py` (Plotly table with diamonds, candidate aim lines, animated replays).
- Streamlit demo (D-020) with four demo positions in `demo/sample_inputs.json` (hand-made illustrations; to be replaced by positions mined from real logs).
- README rewritten as a tutorial (setup, quick start, live check, experiment commands, costs, troubleshooting); `data/README.md`.

**Tested**
- New tests: oracle determinism and sanity (a hanging pot is worth more than any safety), frozen-menu information hiding by access level, scoring, random positions validity, tags; demo smoke test (offline end to end).
- The tournament integration test failed twice with `MemoryError` while E0 was using the machine (environmental, D-021); to be re-run after E0.

**Open / next** (superseded by the next entry)

## 2026-11-20: E0 results, hard cases, fixes

**E0: bot tournaments, medium noise, L2 menus, 50 paired seeds (100 games per pairing)** — `results/e0/`, `results/summary.md`, `results/figures/e0_win_rates.png`
- All 300 games finished (0 errors after the D-014 fix and a resume; 0 shot-cap draws); 20–23 shots per game; ~1 foul per player per game.
- B2 vs B1: 54 % [44, 63], paired seed score +0.08, sign-test p = 0.56 (not significant).
- B2 vs B0: 79 % [70, 86], p < 0.001. B1 vs B0: 61 % [51, 70], p = 0.06.
- Pot success: B2 0.61, B1 0.53, B0 0.43–0.47. B2 plays safeties 17–20 % of the time.
- **Interpretation.** At medium noise game outcomes are dominated by execution luck: even random choice wins 21–39 %, and the strong search bot cannot be separated from the simple geometric bot in 100 games. Game-level comparisons between LLM systems (E2) will therefore have low power at this noise level; the decision-level benchmark (E1) must carry the main comparison, as planned. Next: E0b calibrates low/high noise before the dataset is built, because the dataset's annotations and oracle depend on the noise level.

**Other work**
- 9 draft hard cases with rendered PNGs and oracle tables (`data/positions/hard_cases/`, D-022); labels need human review.
- Offline end-to-end check of the positions runner on the hard cases (bots + FakeLLM SA/MA/SA-vote at L0 and L2): all summary columns produced.
- Fixes: budget exhaustion now stops a game (D-023); B0 in the positions runner now gets a per-position random stream (it previously always picked the same menu slot).
- `experiments/analyze.py` regenerates `results/summary.md` and figures (palette validated for colour-vision deficiency; every bar carries a direct label).
- E2 config (`experiments/configs/e2.yaml`) ready for when LLM access works.

**Tested**
- Full suite: 65 tests pass (`pytest -q`, ~2 min).

**Open / next**
- E0b (running): B2 vs B1 at low and high noise; choose the main noise level.
- Build the positions dataset at the chosen noise; dev pilot; E1/E4 once Vertex quota is available.
- Human review of hard-case labels; TA feedback into `docs/ta_feedback.md`.
