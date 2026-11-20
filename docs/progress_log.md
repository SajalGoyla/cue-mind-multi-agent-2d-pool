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

**Open / next**
- Phase 3: parallel tournament runner (paired seeds, swapped seats, resumable), E0 baseline tournaments, noise calibration, B2 weight check.
