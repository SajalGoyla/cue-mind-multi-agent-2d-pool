# CueMind: Multi-Agent LLM Strategic Decision-Making for 2D Pool

Course project for **COMS E6998-007: Introduction to LLM Based Generative AI Systems** (Columbia University, Fall 2026).

**Team:** Sajal Kumar Goyla · Shambhawi Bhure · Alimurtaza Mustafa Merchant

> **Status:** early development. The execution plan is in place; code, data, and results will be added in phases. See [`docs/EXECUTION_PLAN.md`](docs/EXECUTION_PLAN.md).

## Overview

When an LLM has to make a sequence of decisions with long-term consequences, does splitting each decision across specialized agents beat a single agent that has the same tools? CueMind studies this question with full games of eight-ball pool in the [pooltool](https://github.com/ekiefl/pooltool) physics simulator, where every game ends in a clear win or loss.

LLMs are poor at the numerical side of pool (aim angle, power), so CueMind splits the work:

- **Code handles the geometry and physics.** A shared shot generator proposes feasible candidate shots (target ball, pocket, angle, power, spin), and pooltool simulates their likely outcomes.
- **The LLM makes the judgment.** It picks one candidate by its `shot_id`, so it can never produce an illegal or nonsensical action.

The systems we compare get the same candidates, information, model, and simulation budget:

| System | Description |
|---|---|
| Geometric bot | Plays the easiest-looking pot (non-LLM reference) |
| Search bot | Simulates every candidate and keeps the best (non-LLM reference) |
| Single agent | One LLM call per shot |
| Multi-agent | Offense, position, and safety specialists, plus a coordinator that picks the shot (LangGraph) |

## Research questions

1. **RQ1.** Given identical candidates and simulation information, do specialized agents win more games than a single LLM?
2. **RQ2.** When does collaboration help most (offense, positioning, or defense), and is the benefit worth the extra inference cost?
3. **RQ3.** How do access to the simulation tool and shot-execution noise affect the results?

## Planned repository layout

```
src/cuemind/     core logic: simulation, game rules, candidates, policies, LLM agents, evaluation
demo/            interactive 2D demo (Streamlit) with per-agent reasoning traces
experiments/     evaluation scripts and configs
data/            positions benchmark and how it is generated
results/         metrics, figures, logs
docs/            execution plan, architecture, tutorial, progress log
tests/           unit, integration, and demo smoke tests
```

## Setup (planned)

- Python 3.12 with [uv](https://github.com/astral-sh/uv)
- `pooltool-billiards==0.6.0` (needs the Panda3D archive index at install time)
- Claude models through Google Cloud Vertex AI (`anthropic[vertex]`)

Full install, demo, and experiment commands will be added here as the code lands. Credentials go in a local `.env` file (git-ignored) and must never be committed.

## Timeline

| Date | Milestone |
|---|---|
| 2026-12-07 | Final presentation and live demo |
| 2026-12-15 | Final report and repository due |

## References

1. E. Kiefl. *Pooltool: A Python package for realistic billiards simulation.* JOSS 9(101):7301, 2024.
2. P. Patra and D. Kumar. *PoolBench.* ICML 2026 Workshop on New Frontiers in Game-Theoretic Learning.
3. C. Archibald et al. *Computational pool: A new challenge for game theory pragmatics.* AI Magazine 31:33–41, 2010.
4. A. P. Smit et al. *Should we be going MAD?* ICML, PMLR 235, 2024.
5. S. Memery et al. *CueTip: An interactive and explainable physics-aware pool assistant.* arXiv:2501.18291, 2025.
