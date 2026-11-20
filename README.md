# CueMind: Multi-Agent LLM Strategic Decision-Making for 2D Pool

Course project for **COMS E6998-007: Introduction to LLM Based Generative AI Systems** (Columbia University, Fall 2026).

**Team:** Sajal Kumar Goyla · Shambhawi Bhure · Alimurtaza Mustafa Merchant

## What this is

When an LLM has to make a sequence of decisions with long-term consequences, does splitting each decision across specialized agents beat a single agent that has the same tools? CueMind studies this with eight-ball pool in the [pooltool](https://github.com/ekiefl/pooltool) physics simulator, where every game ends in a clear win or loss.

LLMs are poor at the numerical side of pool, so the work is split:

- **Code does geometry and physics.** A shared generator proposes candidate shots (target, pocket, aim, speed, spin); pooltool simulates them, with realistic stroke errors, to estimate how often each shot pots, scratches or fouls, where the cue ball ends, and what the opponent gets if it misses.
- **The LLM system chooses.** It picks one candidate by its `shot_id` and explains why. Illegal or nonsensical actions are impossible by construction.

| System | Description |
|---|---|
| B0 random | Random candidate (floor) |
| B1 geometric | Easiest-looking pot (non-LLM reference) |
| B2 search | Highest expected value over the same simulated menu (strong non-LLM reference) |
| SA | Single LLM agent, one call per shot |
| SA-vote | Single agent, 4 independent samples + majority vote (compute-matched control) |
| MA | Offense, position and safety specialists in parallel + a coordinator (LangGraph) |
| MA+critic, MA ablations | Extension and ablations |

Every comparison holds the candidates, information, model, strategy knowledge, execution noise and fallback rule fixed across systems (see the fairness contract in [`docs/EXECUTION_PLAN.md`](docs/EXECUTION_PLAN.md) §2.3).

**Research questions.** RQ1: do specialized agents choose better shots and win more than a single LLM? RQ2: when does collaboration help (offense, position, defense), and is it worth the extra cost? RQ3: how do simulation access and execution noise change the answer?

## Repository layout

```
src/cuemind/
  sim/          table state, stroke execution, noise, trajectories
  game/         8-ball rules, break, ball in hand, game runner
  candidates/   shot generation, simulation annotations (L0/L1/L2), pruning, descriptions
  policies/     B0/B1/B2 bots, single agent, multi-agent, registry
  agents/       LangGraph multi-agent graph, shared retry logic
  llm/          Vertex client, response cache + budget, schemas, prompts (prompts/*.md)
  eval/         tournaments, positions benchmark, oracle, metrics, statistics
  viz/          2D table rendering (Plotly)
demo/           Streamlit demo (app.py) and demo positions
experiments/    experiment scripts and YAML configs
data/           positions benchmark (see data/README.md)
results/        experiment outputs (summaries committed; raw shot logs git-ignored)
docs/           execution plan, architecture, decision log, progress log
tests/          unit, integration and demo smoke tests
scripts/        check_llm.py (live API check)
```

Start with [`docs/architecture.md`](docs/architecture.md) (how a decision flows through the system), [`docs/decisions.md`](docs/decisions.md) (why things are the way they are) and [`docs/progress_log.md`](docs/progress_log.md) (what has been done and measured).

## Setup

Prerequisites: Python 3.12, [uv](https://github.com/astral-sh/uv), git. For live LLM runs: a Google Cloud project with Claude enabled on Vertex AI and a service-account key **stored outside this repository**.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt \
    --extra-index-url https://archive.panda3d.org/simple/ --prerelease=allow --index-strategy unsafe-best-match
uv pip install --python .venv/Scripts/python.exe -e .
cp .env.example .env        # then fill in GOOGLE_APPLICATION_CREDENTIALS and GCP_PROJECT_ID
```

On macOS/Linux use `.venv/bin/python` instead of `.venv/Scripts/python.exe`. On Windows set `PYTHONUTF8=1` (pooltool prints Unicode). The first simulation after installing takes up to a minute while Numba compiles; later processes take 2–3 s.

## Quick start (no API key needed)

```bash
pytest -q                                              # unit, integration and demo smoke tests (offline)
python -m cuemind.game.runner --p1 b2 --p2 b1 --games 1 # one full game between two bots
streamlit run demo/app.py                              # demo; choose "Offline (fake LLM)" in the sidebar
```

## Live LLM check

```bash
python scripts/check_llm.py --quick   # one structured call (fractions of a cent)
python scripts/check_llm.py           # plus one single-agent and one multi-agent decision
```

All LLM responses are cached in `results/cache/llm_cache.sqlite`, so re-running anything is free and exact. Every LLM experiment takes `--max-usd`; the run stops cleanly at the limit and resumes from the cache.

## Experiments

| Exp | What | Command |
|---|---|---|
| E0 | Bot tournaments (B2/B1/B0), 50 paired seeds | `python experiments/run_tournament.py --config experiments/configs/e0.yaml` |
| E0b | Noise calibration (low/high) | `python experiments/run_tournament.py --config experiments/configs/e0_noise.yaml` |
| Data | Build the positions benchmark | `python experiments/build_dataset.py --config experiments/configs/dataset.yaml` |
| Pilot | Prompt iteration on the dev split | `python experiments/run_positions.py --config experiments/configs/dev_pilot.yaml --max-usd 2` |
| E1 | Main decision-level comparison | `python experiments/run_positions.py --config experiments/configs/e1.yaml --max-usd 5` |
| E4 | Simulation-access ablation (L0/L1/L2) | `python experiments/run_positions.py --config experiments/configs/e4.yaml --max-usd 5` |
| E6 | Specialist ablations | `python experiments/run_positions.py --config experiments/configs/e6.yaml --max-usd 5` |
| E10 | Decision variance (3 samples) | `python experiments/run_positions.py --config experiments/configs/e10.yaml --limit 30` |
| E2 | Full games SA vs MA vs B2 (LLM) | `python experiments/run_tournament.py --config experiments/configs/e2.yaml` |
| Hard cases | Build/refresh hand-made cases + PNGs for review | `python experiments/build_hard_cases.py` |
| Analysis | All tables and figures → `results/summary.md`, `results/figures/` | `python experiments/analyze.py` |
| Errors | Highest-regret LLM decisions for error analysis | `python experiments/failure_cases.py --exp e1` |
| Demo assets | Pick S1–S4 sample runs from real logs, export PNGs | `python scripts/export_demo_assets.py --exp e1` |

Add `--llm fake` to any positions experiment to run it offline. Tournaments and dataset building are CPU-only (no API cost); use `--workers` to match your machine (each worker needs ~0.5 GB of RAM). Results go to `results/<exp_id>/` (`summary.csv`, `games.jsonl` or `decisions.jsonl`, `config_used.json`).

## Costs

Simulation is free (CPU). LLM cost per decision is logged with token counts. Price estimates use `experiments/configs/prices.yaml` (Anthropic list prices; **Vertex AI prices are set separately and must be confirmed**). With Claude Haiku 5.5 the full study is expected to cost a few tens of US dollars; see `docs/EXECUTION_PLAN.md` Appendix C.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No solution found ... panda3d` during install | Use the Panda3D archive index flags shown above |
| `UnicodeEncodeError` on Windows | `set PYTHONUTF8=1` (or `PYTHONIOENCODING=utf-8`) |
| `AttributeError: module 'httpx' has no attribute 'HTTPStatusError'` | A pre-release httpx was installed; `uv pip install "httpx==0.28.1"` |
| HTTP 429 `Quota exceeded ... global_online_prediction_requests_per_base_model` | Request a Vertex AI quota increase for the Claude model, or wait and resume (runs resume from the cache) |
| HTTP 404 `Publisher model ... not found` | The model is not enabled in that region/project; use `VERTEX_REGION=global` or enable the model |
| `MemoryError` / "paging file is too small" | Too many worker processes for your RAM; lower `--workers` |
| First run is very slow | Numba JIT compilation (once per install, ~1 min) |

## Credits

Physics: [pooltool](https://github.com/ekiefl/pooltool) (Apache-2.0, E. Kiefl). Pot geometry uses `pooltool.ai.pot`. Orchestration: LangGraph. Models: Claude via Google Cloud Vertex AI.

## References

1. E. Kiefl. *Pooltool: A Python package for realistic billiards simulation.* JOSS 9(101):7301, 2024.
2. P. Patra and D. Kumar. *PoolBench.* ICML 2026 Workshop on New Frontiers in Game-Theoretic Learning.
3. C. Archibald et al. *Computational pool: A new challenge for game theory pragmatics.* AI Magazine 31:33–41, 2010.
4. A. P. Smit et al. *Should we be going MAD?* ICML, PMLR 235, 2024.
5. S. Memery et al. *CueTip: An interactive and explainable physics-aware pool assistant.* arXiv:2501.18291, 2025.
