# CueMind: Multi-Agent LLM Strategic Decision-Making for 2D Pool

Course project for **COMS E6998-007: Introduction to LLM Based Generative AI Systems** (Columbia University, Fall 2026).

**Team:** Sajal Kumar Goyla · Shambhawi Bhure · Alimurtaza Mustafa Merchant

> This is **v2** (branch `v2-games`): a games-only study. The earlier positions-benchmark version is preserved under the tag `v1-positions-benchmark`; why it was replaced is explained in [`docs/PLAN_V2.md`](docs/PLAN_V2.md).

## Question

Given the same shot menu, and without simulated forecasts, does a **team** of specialised LLM agents play better eight-ball than a **single** LLM agent? Both play full games in the [pooltool](https://github.com/ekiefl/pooltool) physics simulator against the same simulation-based reference player. Secondary: what does the team cost in calls, tokens and time?

## How a shot is decided

1. **Shot generator (code):** every possible direct pot × 3 speeds × 3 spins, plus safe shots, is forecast 3 times with realistic stroke errors; the top 15 are kept (every legal ball represented) and shuffled.
2. **Players:**
   - **Stat** (reference) picks using the forecasts.
   - **SA** (single agent) sees only geometry (ball, pocket, speed, spin, cut angle, difficulty) and picks one shot.
   - **MA** (team): Offense, Position and Safety specialists each nominate their 3 best shots; if all three put the same shot first it is played, otherwise a coordinator chooses among the nominations.
3. **pooltool** plays the shot (with or without stroke noise) and the 8-ball rules decide the outcome.

All LLM agents use **Qwen 2.5 3B Instruct**, running locally with Ollama.

## Repository layout

```
src/cuemind/
  sim/          table state, stroke execution, noise, trajectories
  game/         8-ball rules, break, ball in hand, game runner
  candidates/   shot generation, forecasting, the v2 menu (menu_v2.py), descriptions
  policies/     Stat (search_bot.py), B1, single agent, multi-agent, registry
  agents/       LangGraph team graph (shortlists → coordinator)
  llm/          Ollama client, response cache, schemas, prompts (prompts/*.md)
  eval/         tournaments, metrics, statistics
  viz/          2D table rendering
demo/           Streamlit demo (app.py), demo positions, demo games
experiments/    run_tournament.py, analyze.py, configs/
scripts/        check_llm.py, run_queue.py, ollama_watchdog.py
docs/           PLAN_V2.md, architecture, decisions, progress log
tests/          unit, integration and demo smoke tests
```

## Setup

Prerequisites: Python 3.12, [uv](https://github.com/astral-sh/uv), [Ollama](https://ollama.com).

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt \
    --extra-index-url https://archive.panda3d.org/simple/ --prerelease=allow --index-strategy unsafe-best-match
uv pip install --python .venv/Scripts/python.exe -e .
cp .env.example .env

ollama pull qwen2.5:3b-instruct
OLLAMA_NUM_PARALLEL=2 ollama serve      # PowerShell: $env:OLLAMA_NUM_PARALLEL="2"; ollama serve
```

On macOS/Linux use `.venv/bin/python`. On Windows set `PYTHONUTF8=1`. The first simulation after installing takes up to a minute (Numba compiling).

## Quick start

```bash
pytest -q                                   # all tests, offline (fake LLM)
python scripts/check_llm.py                 # one live single-agent and one team decision
streamlit run demo/app.py                   # demo: Decide a shot · Replay a game · Results
```

## Running the study

| Step | Command | Notes |
|---|---|---|
| Pilot (bots only) | `python experiments/run_tournament.py --config experiments/configs/v2_pilot.yaml` | CPU only |
| Floor check, prompt dev, main run, analysis | `python scripts/run_queue.py --concurrency 2 --watchdog-mb 3000` | GPU + CPU; resumable; keeps Windows awake; restarts Ollama if its memory grows |
| Analysis only | `python experiments/analyze.py` | writes `results/summary.md` and `results/figures/` |

Every run resumes where it stopped: finished games in `results/<exp>/games.jsonl` are skipped.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No solution found ... panda3d` during install | Use the Panda3D archive index flags shown above |
| `UnicodeEncodeError` on Windows | `set PYTHONUTF8=1` |
| `connection: is ollama serve running` | Start `ollama serve` and pull the model |
| Runs stop for low memory | Lower `--concurrency`; keep `OLLAMA_NUM_PARALLEL=2`; the queue's watchdog restarts Ollama when it grows |
| `Assertion failed: thread != nullptr ... threadWin32Impl.cxx` at exit | Harmless Panda3D message at interpreter shutdown |

## Credits

Physics: [pooltool](https://github.com/ekiefl/pooltool) (Apache-2.0, E. Kiefl). Orchestration: LangGraph. Model: Qwen 2.5 3B Instruct via Ollama.

## References

1. E. Kiefl. *Pooltool: A Python package for realistic billiards simulation.* JOSS 9(101):7301, 2024.
2. P. Patra and D. Kumar. *PoolBench.* ICML 2026 Workshop on New Frontiers in Game-Theoretic Learning.
3. C. Archibald et al. *Computational pool: A new challenge for game theory pragmatics.* AI Magazine 31:33–41, 2010.
4. A. P. Smit et al. *Should we be going MAD?* ICML, PMLR 235, 2024.
5. S. Memery et al. *CueTip: An interactive and explainable physics-aware pool assistant.* arXiv:2501.18291, 2025.
