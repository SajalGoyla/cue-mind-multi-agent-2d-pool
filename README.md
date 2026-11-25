# CueMind: Multi-Agent LLM Strategic Decision-Making for 2D Pool

Course project for **COMS E6998-007: Introduction to LLM Based Generative AI Systems** (Columbia University, Fall 2026).

**Team:** Sajal Kumar Goyla · Shambhawi Bhure · Alimurtaza Mustafa Merchant

## What this is

When an LLM has to make a sequence of decisions with long-term consequences, does a team of specialised agents decide better than a single agent? CueMind studies this with full games of eight-ball pool in the [pooltool](https://github.com/ekiefl/pooltool) physics simulator, where every game ends in a clear win or loss.

LLMs are poor at the numerical side of pool, so the work is split:

- **Code lists the candidate shots from geometry** (target ball, pocket, speed, spin, cut angle, a difficulty word). Every player gets the same menu for the same position.
- **The LLM systems choose a shot by its id**, judging from the geometry and the position alone. They are never told what a shot will do.
- **B, the reference player, simulates every candidate privately** before choosing, so it is the strong yardstick every system plays against.

| System | Description |
|---|---|
| **B** | Simulates each candidate (once without noise, 8 times with noise) and picks the highest expected value |
| **SA** | Single LLM agent: one call per shot |
| **MA** | Offense, position and defense specialists each shortlist up to 3 shots in parallel; a coordinator picks one of the shortlisted shots (skipped when all three rank the same shot first). LangGraph |
| B1 | Easiest pot by geometry. Used only in the pilot, to check that decisions matter in games |

**Evaluation is game play only:** B vs B, B vs SA and B vs MA on the same seeds, both seat orders, with common random numbers. SA and MA are compared seed by seed through their results against the same B. Metrics: win rate with a 95 % interval, pot success, balls per inning, fouls, and LLM calls, tokens and latency per decision.

**Why this design (v2, decision D-029).** The first version ranked decisions on a positions benchmark graded by a simulated "oracle" and showed the LLMs simulated statistics. Both came from the same simulator and noise model, so the benchmark mostly measured how closely a chooser weighed the simulator's numbers the way the oracle did. v2 removes the oracle and every simulated forecast from the LLM input, and evaluates only by games. The v1 code and results remain on `main` (commit `6c96b93`).

## Repository layout

```
src/cuemind/
  sim/          table state, stroke execution, noise, trajectories
  game/         8-ball rules, break, ball in hand, game runner
  candidates/   shot generation, the raw menu, B's private simulation statistics
  policies/     B, B1, single agent, multi-agent, registry
  agents/       LangGraph multi-agent graph, shared retry logic
  llm/          Ollama client and fake LLM, response cache, schemas, prompts (prompts/*.md)
  eval/         tournaments, metrics, statistics
  viz/          2D table rendering (Plotly, matplotlib)
demo/           Streamlit demo (app.py), demo positions, demo games
experiments/    run_tournament.py, analyze.py, configs/*.yaml
results/        experiment outputs (summaries committed; raw shot logs git-ignored)
docs/           architecture, decision log, progress log, report/presentation outlines
tests/          unit, integration and demo smoke tests
scripts/        check_llm.py (live model check), ollama_watchdog.py
```

Start with [`docs/architecture.md`](docs/architecture.md) (how a decision flows through the system) and [`docs/decisions.md`](docs/decisions.md) (why things are the way they are; D-029 describes v2).

## Setup

Prerequisites: Python 3.12 or 3.13, [uv](https://github.com/astral-sh/uv), git; for LLM games, [Ollama](https://ollama.com/download).

Windows / Linux:
```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt \
    --extra-index-url https://archive.panda3d.org/simple/ --prerelease=allow --index-strategy unsafe-best-match
uv pip install --python .venv/Scripts/python.exe -e .
```
(On Linux use `.venv/bin/python`. On Windows set `PYTHONUTF8=1`; pooltool prints Unicode.)

macOS (the pinned Panda3D dev build is not available there; PyPI's 1.10.16 works):
```bash
uv venv --python 3.12 .venv
grep -v '^panda3d==' requirements.txt > .venv/requirements-mac.txt
uv pip install --python .venv/bin/python -r .venv/requirements-mac.txt "panda3d==1.10.16" --prerelease=allow
uv pip install --python .venv/bin/python -e .
```

The first simulation after installing takes up to a minute while Numba compiles; later processes take 2–3 s. **Run every command from the repository root** (the LLM response cache path is relative).

## Quick start (no model needed)

```bash
pytest -q                                                   # all tests, offline (fake LLM)
python -m cuemind.game.runner --p1 b --p2 b1 --games 1      # one full game between two bots
python -m cuemind.game.runner --p1 sa --p2 ma --llm fake    # one game with the offline fake LLM
streamlit run demo/app.py                                   # demo; "Offline (fake LLM)" needs nothing else
```

## Local LLM

```bash
ollama pull qwen2.5:7b-instruct          # ~4.7 GB
OLLAMA_NUM_PARALLEL=2 ollama serve       # PowerShell: $env:OLLAMA_NUM_PARALLEL="2"; ollama serve
python scripts/check_llm.py              # one structured call, then one SA and one MA decision
```

`CUEMIND_OLLAMA_MODEL` selects another model. All responses are cached in `results/cache/llm_cache.sqlite`, so re-running anything is instant and exact.

## Experiments

| Step | What | Command |
|---|---|---|
| 1. Pilot (bots, free) | B vs B1 and B vs B at noise `none` and `low`: picks the noise level where decisions visibly matter | `python experiments/run_tournament.py --config experiments/configs/pilot_bots.yaml` |
| 2. Dev | Prompt development and floor check on dev seeds (B vs SA, B vs MA); then freeze prompts (`git tag prompts-v2`) | `python experiments/run_tournament.py --config experiments/configs/dev.yaml` |
| 3. Main | B vs B, B vs SA, B vs MA on 30 seeds × 2 seat orders | `python experiments/run_tournament.py --config experiments/configs/main.yaml` |
| Analysis | All tables and figures → `results/summary.md`, `results/figures/` | `python experiments/analyze.py` |

Set the noise level chosen by the pilot in `dev.yaml` and `main.yaml`. Add `--seeds 1 --llm fake` to any command for an offline dry run. Re-running a command resumes where it stopped. Results go to `results/<exp_id>/` (`games.jsonl`, `summary.csv`, `config_used.json`; per-shot logs in `raw/`, git-ignored). To show a game in the demo's replay tab, copy its raw log into `demo/games/`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No solution found ... panda3d` during install | Windows/Linux: use the Panda3D archive index flags above. macOS: use the macOS commands |
| `UnicodeEncodeError` on Windows | `set PYTHONUTF8=1` (or `PYTHONIOENCODING=utf-8`) |
| `AttributeError: module 'httpx' has no attribute 'HTTPStatusError'` | A pre-release httpx was installed; `uv pip install "httpx==0.28.1"` |
| `connection: is ollama serve running` | Start the model server: `ollama serve` |
| A decision returns an old answer after a code change | The response cache returned it (same prompt). Point `CUEMIND_LLM_CACHE` to a new file |
| `MemoryError` / "paging file is too small" | Too many worker processes for your RAM; lower `--workers` |
| Ollama's memory grows during long runs (seen on Windows) | Run `python scripts/ollama_watchdog.py` alongside the experiment |
| First run is very slow | Numba JIT compilation (once per install, ~1 min) |
| `Assertion failed: thread != nullptr ... threadWin32Impl.cxx` when the demo or a test exits | Harmless Panda3D message at interpreter shutdown |

## Credits

Physics: [pooltool](https://github.com/ekiefl/pooltool) (Apache-2.0, E. Kiefl). Pot geometry uses `pooltool.ai.pot`. Orchestration: LangGraph. Model: Qwen 2.5 7B Instruct via Ollama.

## References

1. E. Kiefl. *Pooltool: A Python package for realistic billiards simulation.* JOSS 9(101):7301, 2024.
2. P. Patra and D. Kumar. *PoolBench.* ICML 2026 Workshop on New Frontiers in Game-Theoretic Learning.
3. C. Archibald et al. *Computational pool: A new challenge for game theory pragmatics.* AI Magazine 31:33–41, 2010.
4. A. P. Smit et al. *Should we be going MAD?* ICML, PMLR 235, 2024.
5. S. Memery et al. *CueTip: An interactive and explainable physics-aware pool assistant.* arXiv:2501.18291, 2025.
