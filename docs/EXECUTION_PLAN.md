# CueMind: End-to-End Execution Plan (for an LLM coding agent)

> **Superseded for v2 (2026-11-25).** This plan describes v1 (positions benchmark, oracle, access levels, SA-vote, critic, Claude on Vertex). The current design is decision **D-029** in [`decisions.md`](decisions.md) and [`architecture.md`](architecture.md). Kept for history; the v1 code and results are on `main` at `6c96b93`.

> **Who this is for.** This file is written to be handed to an LLM coding agent (e.g. Claude Code) as its working brief. It is self-contained: an agent that follows it phase by phase should produce the full system, the dataset, all evaluations, the demo UI, the sample runs for the class presentation, and the material for the report.
>
> **Status on 2026-11-20.** We have the one-page proposal (`Final_Project_Proposal_LLM_Based_Gen_AI.pdf`), the course rubric (`COMSE6998_007_Fall_2026_project_deliverables_doc.pdf`), and a working feasibility spike (`cuemind-feasibility/`, see its `FEASIBILITY_REPORT.md`). No production code exists yet. The older `plan.md` (2026-10-15) is **superseded** by this file wherever they differ; this file follows the final proposal.
>
> **Hard deadlines.** Presentation + live demo: **Mon 2026-12-07** (Prof. Chen Wang's slot; confirm on the sign-up sheet). Report + GitHub repo: **Tue 2026-12-15**. Late = zero credit. We are behind the proposal's timeline, so the schedule in §14 is compressed and has explicit cut-lines.

---

## Table of contents

0. Rules of engagement for the executing agent
1. Project summary (what we are building and why)
2. Research questions, hypotheses, and the fairness contract
3. Tech stack, environment, credentials
4. Repository layout
5. Phase 0: scaffold and port the spike
6. Phase 1: game engine (state, rules, noise, break, ball-in-hand)
7. Phase 2: candidate generator and simulation annotations
8. Phase 3: non-LLM baselines and baseline evaluation
9. Phase 4: the evaluation dataset (positions benchmark)
10. Phase 5: LLM layer, single agent, multi-agent (LangGraph)
11. Phase 6: experiments (main comparison, ablations, RQ1–RQ3)
12. Phase 7: analysis, statistics, figures, failure analysis
13. Phase 8: demo UI, sample runs, presentation, report
14. Schedule, work split, cut-lines
15. Testing strategy (unit / integration / smoke / regression)
16. Risk register
17. Future work (for the report and slides)
18. Definition of done (final checklist against the rubric)
Appendix A: data contracts (JSON schemas)
Appendix B: prompt templates
Appendix C: cost and runtime budget
Appendix D: command cheat-sheet

---

## 0. Rules of engagement for the executing agent

Read these before doing anything. They override convenience.

1. **Work phase by phase.** Each phase ends with a *Definition of Done (DoD)*. Do not start phase N+1 until phase N's DoD tests pass, except where §14 marks work as parallel.
2. **Run the tests after every meaningful change** (`pytest -q`). Never leave the repo with failing tests at the end of a session. If a test is wrong, fix the test and say why in the commit message.
3. **Never touch credentials.** The GCP service-account key lives at `Final Project/testing-api-key/*.json`. Do **not** open, print, copy, or commit it. Reference it only through the `GOOGLE_APPLICATION_CREDENTIALS` environment variable loaded from a git-ignored `.env`. `.gitignore` must exclude `.env`, `*.json` keys under any `*api-key*` folder, `results/raw/`, and the LLM cache DB.
4. **Spend money only with approval.** Every command that calls a paid LLM must go through the `--max-usd` guard (§10.2). Before any run whose estimated cost exceeds **US$5**, stop and ask the human. Unit and integration tests must use the `FakeLLM`; they must never call the real API.
5. **Do not modify third-party code** (`cuemind-feasibility/third_party/`). Work around upstream quirks in our code and pin them with tests (the spike already lists them).
6. **Never fabricate results.** Every number in a figure, table, slide, or the report must come from a file in `results/` produced by a script in the repo. If a run fails, record the failure; do not fill the gap by hand.
7. **Freeze before you test.** Prompts and agent logic are tuned **only** on the `dev` split. Once the main experiments start (§11), prompts are frozen and tagged in git (`prompts-v1`). Any later prompt change means re-running the affected experiments.
8. **Log everything a decision touched** (state, candidates, prompts, raw responses, parsed outputs, tokens, latency, fallback flags, executed noise, outcome). The demo and the failure analysis depend on these logs.
9. **Keep the fairness contract (§2.3).** If a change would give one system information, compute, or knowledge the other lacks, it is a bug.
10. **Commit small and often** with clear messages. Do not push to GitHub or create PRs unless the human asks. Do not rewrite git history.
11. **Ask the human** when: a decision in §2.4 ("decisions to confirm") is still open; a cost guard trips; the TA feedback text is needed (§13.6); a result contradicts the hypothesis badly enough that the framing should change; you need hand-labelled ground truth (§9.4).
12. **Keep a progress log** in `docs/progress_log.md`: date, what was done, which tests/runs, where the outputs are, open problems. One short entry per working session.

---

## 1. Project summary

**CueMind** asks: *when an LLM must make a sequence of decisions with long-term consequences, does splitting each decision across specialized agents beat a single agent, when both get the same tools?* The testbed is eight-ball pool in the `pooltool` physics simulator: each game ends in a win or a loss, so outcomes are unambiguous.

**Design principle (from PoolBench and CueTip):** LLMs are bad at the numerical side of pool (raw angle/power output pockets nearly nothing), so **code does the geometry and physics, and the LLM does the judgment**. A shared geometric shot generator proposes feasible candidate shots (target ball, pocket, aim angle, power, spin); pooltool simulates them; the LLM system **picks one candidate by `shot_id`**. Illegal or nonsensical actions are therefore impossible by construction, and the only thing that differs between systems is *which* candidate they choose.

**Systems compared** (same model, same candidates, same information, same simulation budget):

| ID | System | LLM? | Role in the study |
|---|---|---|---|
| B0 | Random-candidate bot | no | Floor / sanity check |
| B1 | Geometric bot: easiest pot by geometry score, medium power, stun | no | Reference baseline from the proposal |
| B2 | Search bot: simulates every candidate (noisy rollouts), picks max expected heuristic value | no | Strong non-LLM reference baseline from the proposal |
| SA | Single agent: one LLM call per decision | yes | **Primary baseline** |
| MA | Multi-agent: Offense + Position + Safety specialists in parallel → Coordinator | yes | **Main system** |
| SA-vote | Compute-matched single agent: 4 independent SA samples, majority vote (ties → highest mean confidence) | yes | Controls for "MA just uses 4× more calls" (Smit et al., *Should we be going MAD?*) |
| MA+critic | MA plus a critic that can veto the coordinator once | yes | Extension (only if time) |

**Deliverables** (from the rubric): live interactive 2D demo with per-agent reasoning traces; a reproducible evaluation benchmark (positions dataset + game harness); results with baselines, confidence intervals, ablations, and failure analysis; a report in the ACM template; and a clean GitHub repo with tests and a step-by-step tutorial.

---

## 2. Research questions, hypotheses, fairness contract

### 2.1 Research questions (verbatim from the proposal)

- **RQ1.** Given identical geometric candidates and simulation information, do specialized agents win more games than a single LLM?
- **RQ2.** When does collaboration help most (offense, positioning, or defense), and does its benefit justify the extra inference cost?
- **RQ3.** How do simulation-tool access and shot-execution noise affect the results?

### 2.2 Hypotheses (state them in the slides and report, then test them)

- **H1.** MA achieves lower decision regret than SA on the positions benchmark and a higher win rate in paired games. *Expected effect is modest*; a null result is a valid, reportable finding (cite Smit et al.).
- **H2.** MA's advantage concentrates in **position-play** and **safety** situations (where trade-offs exist), not in easy-pot situations (where both pick the obvious shot).
- **H3.** MA costs about 4–5× SA in tokens and 1.5–3× in latency (specialists run in parallel). SA-vote at matched compute closes part of the gap.
- **H4.** Without simulation annotations (geometry only), both LLM systems degrade sharply and the MA–SA gap shrinks, because specialists have less to reason about. More execution noise increases the value of safety play and therefore of the safety specialist.

**Assumptions to state explicitly:** the shared candidate generator is good enough that the best shot is usually in the menu; execution noise is a reasonable proxy for an amateur player; a single model (Claude Haiku 5.5 by default) is representative enough for the main study, and key results are confirmed on one stronger model.

### 2.3 Fairness contract (invariants; enforce them in code and in tests)

For any comparison between SA, SA-vote and MA (and B2 where relevant):

1. **Same candidate list** for a given state (deterministic generator, seeded). Assert by hashing the candidate list in each decision log record and comparing across systems in paired runs.
2. **Same annotations** (same simulation access level, same rollout budget, same rollout seeds).
3. **Same non-shot policies:** break, ball-in-hand placement, and pocket-call logic are shared code and are **not** LLM-controlled.
4. **Same execution noise draws** (common random numbers): the noise for shot *k* of game seed *s* comes from `rng(seed=hash(s, k))`, identical for both systems.
5. **Same model, same effort, same max_tokens, same output schema style.**
6. **Same domain knowledge:** SA's system prompt contains the union of the strategy knowledge given to the three specialists (offense, position, safety guidance). MA must not win because it was simply told more about pool.
7. **Same fallback rule** when the output is invalid after one retry: pick the candidate with the highest `p_pot` (ties → geometry score). Fallback rate is reported per system.
8. **Prompts frozen** before test-split runs (git tag).

### 2.4 Decisions to confirm with the team (defaults are what the agent should use if nobody answers)

| # | Decision | Default |
|---|---|---|
| D1 | Bulk-experiment model | `claude-haiku-5-5` via Vertex AI (cheap; matches the proposal's "run most experiments on a cheaper model") |
| D2 | Confirmation model | `claude-sonnet-5-5` via Vertex AI (already verified working in `testing-api-key/test_models.py`) on a subset |
| D3 | Open-weight model (proposal mentions one) | Optional P2: a local model via Ollama behind the same `LLMClient` interface. Skip if no GPU/time; explain the change in the report |
| D4 | Game variant | Full 15-ball 8-ball for headline games; mid-game positions benchmark carries most of the statistical weight |
| D5 | Main noise level | Medium: φ σ = 0.5°, V0 σ = 5 %, tip offset σ = 0.02 R |
| D6 | Effort setting | `output_config.effort = "medium"` for every LLM call in every system (ablation: `low`) |
| D7 | Total LLM budget | Ask the human. Appendix C estimates ≈ US$15–30 for the full Haiku study and ≈ US$40–80 for the Sonnet confirmation (check current Vertex pricing) |

---

## 3. Tech stack, environment, credentials

- **Python 3.12**, managed with `uv` (the spike's setup is verified on this machine, Windows 11).
- **Simulator:** `pooltool-billiards==0.6.0` (Apache-2.0). Needs the Panda3D archive index at install time (see the spike README). First simulation after install takes ~30 s (Numba JIT); warm up once per process with `pt.simulate(pt.System.example())`.
- **LLM access:** Anthropic Python SDK with the Vertex client: `pip install "anthropic[vertex]" google-auth`. Client: `AnthropicVertex(project_id=..., region="global")`. Model IDs on Vertex for current models are the bare IDs (`claude-haiku-5-5`, `claude-sonnet-5-5`), no date suffix.
- **Structured outputs:** use `client.messages.parse(..., output_format=<PydanticModel>)` and read `response.parsed_output`. (Structured outputs and prompt caching are both supported on Vertex.) Always check `response.stop_reason` before reading output (`refusal` and `max_tokens` → treat as invalid output).
- **Model-API notes that affect the design** (verify against the installed SDK; do not guess):
  - Haiku 5.5 and Sonnet 5.5 **reject non-default `temperature`/`top_p`/`top_k`** (400). We therefore **cannot set temperature 0**: outputs are not bit-reproducible. Mitigations: an on-disk response cache (replays are exact), repeated runs on a subset to measure decision variance, and reporting that variance.
  - Thinking is adaptive and on by default; control depth with `output_config={"effort": "medium"}`. Do not send `budget_tokens`. Do not use assistant prefill (400).
  - Prompt caching: put the long static strategy text in the `system` block with `cache_control: {"type": "ephemeral"}`; volatile content (state, candidates) goes in the user message. Verify with `usage.cache_read_input_tokens > 0` on repeated calls; if it stays 0, the prefix is below the model's minimum cacheable length: note it, do not pad artificially.
  - Token accounting: record `usage.input_tokens`, `usage.output_tokens`, `usage.cache_read_input_tokens`, `usage.cache_creation_input_tokens` for every call.
  - Retries: construct the client with `max_retries=5`; catch `anthropic.RateLimitError`, `anthropic.APIStatusError`, `anthropic.APIConnectionError` separately (most specific first). Message Batches are **not** available on Vertex, so throughput comes from bounded concurrency.
- **Orchestration:** `langgraph` (the proposal commits to LangGraph). One graph for SA (single node) and one for MA (fan-out to 3 specialist nodes, join into coordinator), so both systems share the same plumbing. Check the installed LangGraph version's API for parallel branches and state reducers before writing the graph.
- **Other libraries:** `pydantic>=2`, `numpy`, `scipy`, `pandas`, `matplotlib`, `plotly`, `streamlit`, `pyyaml`, `python-dotenv`, `tqdm`, `pytest`, `pytest-timeout`, `imageio` (GIF export). Pin exact versions in `requirements.txt` after the first successful install (`uv pip freeze`).

`.env.example` (committed; the real `.env` is git-ignored):

```
GOOGLE_APPLICATION_CREDENTIALS=C:/path/to/service-account.json   # never commit the key
GCP_PROJECT_ID=your-project-id
VERTEX_REGION=global
CUEMIND_MODEL=claude-haiku-5-5
CUEMIND_EFFORT=medium
CUEMIND_MAX_USD=5
CUEMIND_LLM_CACHE=results/cache/llm_cache.sqlite
PYTHONIOENCODING=utf-8
```

---

## 4. Repository layout

The repository is `Final Project/cue-mind-multi-agent-2d-pool/` (a clone of https://github.com/SajalGoyla/cue-mind-multi-agent-2d-pool). All work happens there; `Final Project/cuemind-feasibility/` stays outside the repo as read-only reference. The tree below shows the repo root as `cuemind/` for brevity. Layout follows the course's suggested organisation:

```
cuemind/
├── README.md                    # setup, demo launch, experiment commands, costs, troubleshooting
├── requirements.txt             # pinned
├── pyproject.toml               # package metadata, pytest config
├── .env.example
├── .gitignore
├── src/cuemind/
│   ├── __init__.py
│   ├── config.py                # dataclasses + YAML loading for all settings
│   ├── sim/
│   │   ├── state.py             # TableState <-> pt.System, validation, hashing
│   │   ├── execute.py           # apply a shot (with noise) to a system, extract outcome
│   │   ├── noise.py             # NoiseModel(phi_sd, v0_rel_sd, tip_sd), CRN seeding
│   │   └── trajectories.py      # continuous=True trajectories for the UI
│   ├── game/
│   │   ├── rules.py             # wrapper over pooltool _EightBall: legality, turns, winner, shot cap
│   │   ├── break_shot.py        # deterministic seeded break
│   │   ├── ball_in_hand.py      # shared placement policy
│   │   └── runner.py            # play a game between two Policy objects; JSONL logging
│   ├── candidates/
│   │   ├── direct.py            # pooltool.ai.pot ghost-ball pots
│   │   ├── variants.py          # power x spin grid
│   │   ├── safety.py            # safety / contact candidates
│   │   ├── bank.py              # (P2) 1-cushion bank with simulation aim correction
│   │   ├── annotate.py          # access levels L0/L1/L2: features + noisy rollouts
│   │   ├── prune.py             # neutral pruning to <= K candidates
│   │   └── describe.py          # plain-language rendering for prompts
│   ├── policies/
│   │   ├── base.py              # Policy protocol: decide(DecisionContext) -> Decision
│   │   ├── random_bot.py        # B0
│   │   ├── geometric_bot.py     # B1
│   │   ├── search_bot.py        # B2
│   │   ├── single_agent.py      # SA, SA-vote
│   │   └── multi_agent.py       # MA, MA+critic
│   ├── llm/
│   │   ├── client.py            # LLMClient protocol; ClaudeVertexClient; FakeLLM; (P2) OllamaClient
│   │   ├── cache.py             # sqlite response cache keyed by request hash + repeat index
│   │   ├── budget.py            # token/cost accounting + --max-usd guard
│   │   ├── schemas.py           # pydantic output models
│   │   └── prompts/             # *.md templates (versioned, frozen at prompts-v1)
│   ├── agents/
│   │   ├── graph_single.py      # LangGraph for SA
│   │   └── graph_multi.py       # LangGraph for MA (+critic)
│   ├── eval/
│   │   ├── positions.py         # positions benchmark runner (decision-level)
│   │   ├── oracle.py            # expected-inning-value ground truth
│   │   ├── tournament.py        # paired seeded games, seat swap, parallel workers
│   │   ├── metrics.py           # all metrics from logs
│   │   └── stats.py             # Wilson CI, bootstrap, sign test, Wilcoxon, Holm
│   └── viz/
│       ├── table_plot.py        # Plotly/Matplotlib 2D table, candidate arrows, trajectories
│       └── figures.py           # report/slide figures from results/
├── demo/
│   ├── app.py                   # Streamlit demo
│   ├── sample_inputs.json       # curated demo positions + seeds
│   └── recordings/              # backup screen recording (git-LFS or link)
├── experiments/
│   ├── configs/*.yaml           # one file per experiment (E0…E8)
│   ├── build_dataset.py
│   ├── run_baselines.py
│   ├── run_positions.py
│   ├── run_tournament.py
│   └── analyze.py
├── data/
│   ├── README.md                # how positions are generated, schema, splits, licence
│   ├── positions/dev.jsonl
│   ├── positions/test.jsonl
│   └── positions/hard_cases.yaml
├── results/                     # metrics CSV/JSON, figures; raw logs in results/raw (git-ignored, zipped release)
├── notebooks/analysis.ipynb     # optional; must reproduce figures from results/
├── docs/
│   ├── architecture.md
│   ├── tutorial.md
│   ├── ta_feedback.md
│   ├── progress_log.md
│   └── failure_analysis.md
└── tests/
    ├── unit/…
    ├── integration/…
    └── smoke/test_demo_app.py
```

---

## 5. Phase 0: scaffold and port the spike (≈ 0.5 day)

**Tasks**

1. In the repo clone, create the layout above, `pyproject.toml` (package `cuemind` under `src/`), and `.env.example`. (`README.md`, `.gitignore` and this plan already exist from the initial commit; extend them.)
2. Create the venv and install:
   ```bash
   uv venv --python 3.12 .venv
   uv pip install --python .venv/Scripts/python.exe pooltool-billiards==0.6.0 \
       --extra-index-url https://archive.panda3d.org/simple/ --prerelease=allow --index-strategy unsafe-best-match
   uv pip install --python .venv/Scripts/python.exe "anthropic[vertex]" google-auth langgraph pydantic \
       numpy scipy pandas matplotlib plotly streamlit pyyaml python-dotenv tqdm pytest pytest-timeout imageio
   uv pip freeze --python .venv/Scripts/python.exe > requirements.txt
   ```
3. Port (copy and clean up, do not import from) these spike modules: `integration/adapters.py` → `sim/state.py`; `simulate_candidates.py` → `sim/execute.py` + parts of `candidates/annotate.py`; the `pooltool_pot` part of `candidate_generator.py` → `candidates/direct.py`; `play_game.py` → `game/runner.py`; `visualize.py` → `viz/table_plot.py`. **Drop all Repo B code** (no licence; the spike recommends against it).
4. Port `tests/test_integration.py` and keep the upstream-quirk tests (substring `is_ball_pocketed`, cue `b=0.25` default, pocketed-ball occlusion).
5. Write `llm/client.py` with the `LLMClient` protocol and a `ClaudeVertexClient`, plus a one-off script `scripts/check_llm.py` that sends a 1-line prompt to the configured model through the client (reads `.env`), prints the reply and token usage. The human runs it once to confirm credentials.

**DoD:** `pytest -q` passes; `python -m cuemind.game.runner --p1 b1 --p2 b1 --games 1` plays a full game headlessly; `scripts/check_llm.py` succeeds (run by the human).

---

## 6. Phase 1: game engine (≈ 1 day)

### 6.1 State

`TableState` (pydantic, Appendix A.1): ball positions in metres (pooltool frame), cue ball, shooter's group (`open|solids|stripes`), legal targets, balls remaining per side, `ball_in_hand` flag, shot number, game id. Functions: `from_system`, `to_system`, `validate` (overlaps, off-table, NaN, missing cue: reuse spike validation), `state_hash` (stable SHA-256 of rounded positions + metadata).

### 6.2 Rules (`game/rules.py`)

Wrap pooltool's `_EightBall` ruleset (`pt.get_ruleset(pt.GameType.EIGHTBALL)`):
- Before executing a shot set `shot_constraints.ball_call` / `pocket_call` from the chosen candidate (for safeties: `ball_call` = target, `pocket_call` = nearest pocket, mark as safety in our log).
- After the shot call `process_and_advance(system)` and read `shot_info` (`legal`, `reason`, `turn_over`, `game_over`, `winner`).
- `_EightBall.copy()` is not implemented: never copy the ruleset; judge hypothetical shots with `ruleset.eight_ball.is_legal(shot, constraints, break_shot=False)` on a copied `System` (as the spike does).
- **Shot cap:** 150 shots → draw (scored 0.5 each). Log `status="shot_cap"`.
- **Mid-game start:** support starting a game from a saved `TableState` (groups assigned, chosen shooter). Implement by building the `System` from the state and setting the ruleset's player/group/turn fields; if pooltool does not allow setting group assignment directly, implement a minimal 8-ball rules layer ourselves (`game/rules_lite.py`: open table, group assignment on first legal pot, fouls = scratch / wrong first contact / no rail after contact / no contact, 8-ball early = loss, 8-ball + scratch = loss) and use pooltool only for physics and event extraction. **Decide this in the first two hours of Phase 1 by testing; document the choice in `docs/architecture.md`.**

### 6.3 Noise (`sim/noise.py`)

`NoiseModel(phi_sd_deg, v0_rel_sd, tip_sd)`; `apply(shot, rng)` returns the executed shot: φ + N(0, φ_sd), V0·(1 + N(0, v0_rel_sd)) clipped to [0.3, 6.0] m/s, a/b + N(0, tip_sd) clipped to ±0.5. Presets: `none` (0,0,0), `low` (0.25°, 3 %, 0.01), `medium` (0.5°, 5 %, 0.02), `high` (1.0°, 10 %, 0.04). **CRN seeding:** `rng_for(game_seed, shot_index, purpose)` with `purpose ∈ {"execute","rollout"}` via `numpy.random.default_rng(hash_to_int(...))`.

### 6.4 Break (`game/break_shot.py`)

Deterministic: aim at the apex ball, V0 = 8 m/s, φ jitter ~ U(−1°, 1°) from the game-seed RNG. Same break for both systems in a paired game. If the break scratches or nothing drops, the ruleset handles turn order.

### 6.5 Ball-in-hand (`game/ball_in_hand.py`), shared and non-LLM

After a foul, the incoming shooter places the cue ball. Policy (identical for every system): sample a 7×13 grid of positions over the table (skip overlaps, keep ≥ 1.1 R from cushions), run the direct-pot generator from each, score each spot by `max geometry_score` among its candidates (cheap, no rollouts), place the cue ball at the best spot. Must be fast (< 0.5 s). Write a test that the placement never overlaps a ball.

### 6.6 Game runner (`game/runner.py`)

`play_game(policy_p1, policy_p2, seed, noise, start_state=None, max_shots=150, log_path)`. Each turn: build `TableState` → generate + annotate candidates (§7) → `policy.decide(ctx)` → execute the chosen shot with noise → rules → log one JSONL record (Appendix A.4). Policies receive a `DecisionContext` (state, candidates, game history summary: last 3 shots, score). Catch policy exceptions → fallback rule (§2.3-7), never crash the game.

**DoD:** 20 games B1 vs B1 with medium noise complete with no crashes; all end `finished` or `shot_cap`; logs validate against the schema; tests for noise determinism (same seed → same executed shot), CRN (two policies choosing the same shot at the same index get the same noise), ball-in-hand non-overlap, shot cap, mid-game start.

---

## 7. Phase 2: candidate generator and annotations (≈ 1.5 days)

### 7.1 Generation (deterministic given state)

1. **Direct pots** (`candidates/direct.py`): pooltool `ai.pot` ghost-ball pots for every legal target × pocket, with the spike's rejection checks (pocket path blocked, no room for ghost, jaw in way, cue path blocked, cut > 80°). Filter to on-table balls only (upstream occlusion quirk).
2. **Variants** (`candidates/variants.py`): each viable pot × power {soft 1.5, medium 2.5, firm 4.0 m/s} × spin {draw b = −0.3, stun b = 0, follow b = +0.3} = up to 9 variants. Always pass `a, b, theta` explicitly (pooltool's default b = 0.25).
3. **Safeties** (`candidates/safety.py`): for each legal target, contact shots at φ aimed at a full/half/thin hit (cut 0°, ±30°, ±60°) × power {soft, medium}. Keep those that, in a noise-free simulation, are legal (first contact legal, rail after contact). These are labelled `shot_type="safety"` and called with no intention of potting.
4. **Banks** (P2, cut-line): 1-cushion bank candidates with simulation-based aim correction (sweep φ ± 2° in 0.25° steps, keep the best). Only if Phase 6 is on schedule.

### 7.2 Annotations (simulation access levels; the variable for RQ3)

| Level | Name | What the agents see per candidate |
|---|---|---|
| **L0** | Geometry only | target, pocket, shot type, power/spin labels, cut angle, distances, `required_precision_deg` (pooltool), geometry score. No simulation |
| **L1** | Single preview | L0 + one noise-free simulation: does it pot, scratch, foul; predicted cue-ball end position (zone + coordinates); number of viable pots from there |
| **L2** | Monte-Carlo (main setting) | L0 + N = 8 noisy rollouts (same noise model as execution): `p_pot`, `p_scratch`, `p_foul`, `p_keep_turn`, mean cue end + spread, `next_best_p_pot_geom` (ease of my best next shot from the mean end position), `opp_threat_if_turn_passes` (ease of the opponent's best shot if I miss or play safe), `cluster_contact` flag |
| **L3** | On-demand tool (P2) | Starts at L1; each agent may call `simulate(shot_id)` up to B = 3 times per decision to obtain the L2 stats for that shot. Total rollouts capped at the L2 budget |

**Simulation budget:** with K ≤ 15 candidates and N = 8 rollouts, ≤ 120 rollouts per decision (~2–4 s single core; parallelise rollouts with a process pool). B2 uses exactly the same annotations, so the "search" baseline and the LLMs see identical information.

### 7.3 Pruning (`candidates/prune.py`), neutral and documented

Before the LLM sees the menu: keep at most **K = 12 pot candidates + 3 safety candidates**. Rule: drop pots with `p_pot < 0.10`; then keep the best-`p_pot` variant for each (target, pocket, spin) triple; then the top 12 by `p_pot`; then the 3 safeties with the lowest `opp_threat`. Pruning uses only `p_pot` / `opp_threat`, **not** the B2 objective, so the menu does not pre-select B2's answer. If fewer than 3 candidates survive, add the spike's labelled `contact_only` fallback shots. Shot IDs are reassigned after pruning (`S01…S15`), stable and deterministic.

### 7.4 Plain-language descriptions (`candidates/describe.py`)

LLMs reason poorly over raw coordinates. Render each candidate as one compact table row plus a one-line description, e.g.
`S04 | pot 3 → top-right corner | cut 22° | medium, draw | p_pot 0.81 | scratch 0.02 | cue ends: centre table (leaves 5 → side, easy) | if missed: opponent has an easy shot`.
Describe positions using **diamond coordinates** (short rail 0–4, long rail 0–8) and zone names (head/foot, left/right rail, centre). Also produce a short state summary: groups, balls left for each side, whose turn, ball-in-hand, game phase (open / mid / endgame on the 8).

**DoD:** unit tests for each generator (straight-in pot found; blocked ball rejected; safety candidates legal in noise-free sim; determinism: same state → same candidate list and hash); annotation smoke test on the 10 spike tables (L0/L1/L2 complete, values in [0,1]); timing report written to `results/timing_candidates.json` (target: mean decision prep ≤ 4 s on 15-ball positions with parallel rollouts).

---

## 8. Phase 3: non-LLM baselines and baseline evaluation (≈ 0.5 day + overnight runs)

### 8.1 Policies

- **B0 random:** uniform choice over the pruned menu (seeded).
- **B1 geometric:** highest `geometry_score` pot at medium power, stun; if none, the best safety by geometry. (This is the proposal's "plays the easiest pot".)
- **B2 search:** maximise `EV = p_pot·(1 + 0.5·next_best_p_pot_geom) − 1.5·p_scratch − 1.0·p_foul − 0.6·(1 − p_keep_turn)·opp_threat_if_turn_passes` over all candidates, safeties included. Weights live in config; tune them **only** on a few dev games so that B2 clearly beats B1; then freeze.

### 8.2 Experiment E0: baseline sanity and noise calibration (no LLM, free)

- Tournaments with paired seeds and swapped seats: B2 vs B1, B2 vs B0, B1 vs B0, 100 games each (50 seeds × 2 seat orders), at noise `low`, `medium`, `high`.
- Purpose: validate the harness; calibrate noise so that at `medium` games last ~20–45 shots, fouls happen, and B2 beats B1 clearly but not 100 %. If B2 wins ~100 % at medium noise, raise noise; if games exceed the shot cap often, lower it.
- Outputs: `results/e0_baselines/summary.csv` (win rate ± Wilson CI, shots/game, fouls/game, pot success), and a short note in `docs/progress_log.md` with the chosen noise level.

**DoD:** E0 complete, noise level confirmed (update D5 if changed), harness has run ≥ 600 games without a crash.

---

## 9. Phase 4: the evaluation dataset (positions benchmark) (≈ 1 day, overlaps Phase 5)

No external dataset exists for this task; **we build our own benchmark** and document it in `data/README.md`. It has three parts:

### 9.1 Game seeds (for full games)

`data/seeds.yaml`: 60 game seeds. Seeds 0–9 are **dev** (pilot games), 10–59 are **test**.

### 9.2 Mid-game positions

1. **Sampled from realistic play:** run B2-vs-B1 games at medium noise from seeds 1000–1199 and snapshot the state at random turns (excluding the break shot). This yields a realistic distribution of positions.
2. **Random placements:** 20 % of positions from uniformly random valid placements with random group assignment (covers unusual layouts).
3. Keep only positions with ≥ 2 candidates after pruning. De-duplicate by state hash.
4. **Difficulty groups** from L2 annotations: `easy` (best `p_pot` ≥ 0.80), `medium` (0.40–0.80), `hard` (< 0.40 or no pot candidate, i.e. safety situations). Stratify: equal thirds.
5. **Situation tags** (computed, used for RQ2): `offense` (an easy pot exists and position is not at stake), `position` (≥ 2 pots with `p_pot` ≥ 0.6 that differ in next-shot ease by ≥ 0.3), `defense` (best `p_pot` < 0.4 or opponent near the 8). A position can carry several tags.
6. **Splits:** `dev.jsonl` = 60 positions (prompt iteration only), `test.jsonl` = 180 positions (60 per difficulty). Store the **frozen candidate menu and annotations** inside each record so every system sees byte-identical inputs.

### 9.3 Ground-truth value per candidate (oracle) (`eval/oracle.py`)

For each test position and each candidate compute an **expected inning value (EIV)** with M = 24 noisy rollouts: execute the candidate with noise; if the shooter keeps the turn, continue the inning greedily with the B1 policy (cheap: geometry + one noise-free sim per step) for up to H = 2 further shots; score = legal balls pocketed by the shooter in this inning (+5 if the 8 is legally pocketed to win, −5 for a loss-causing foul on the 8) − 1.0 per foul − 0.75·(opponent's best geometric pot ease when the turn passes). Report EIV mean and standard error. Using B1 (not B2) for continuation avoids circularity with the B2 baseline. Run with a process pool overnight; cache results in the dataset file.

**Decision metrics derived from it:** regret = `max_c EIV(c) − EIV(chosen)`; top-1 accuracy = chosen within 1 standard error of the best; normalised score = `(EIV(chosen) − EIV(random)) / (EIV(best) − EIV(random))`.

### 9.4 Hand-picked hard cases (`data/positions/hard_cases.yaml`), ~25 positions

Write a small YAML format (ball ids → diamond coordinates, groups, shooter) and a script that renders each to PNG for checking. Categories (≈ 4–5 each):
1. **Safety-only:** every pot is blocked or < 15 %; correct intent = safety.
2. **Shape trap:** the easiest pot leaves no next shot or a likely scratch; a slightly harder pot leaves a good next shot.
3. **Endgame:** shooter or opponent is on the 8; risk management matters.
4. **Cluster:** a cluster must be broken now or later.
5. **Scratch risk:** the natural pot sends the cue ball toward a pocket at the default speed/spin.
6. **Ball in hand** (placement is shared code, but the shot choice after placement matters).

Labels: `expected_intent` and an `acceptable_shot_ids` set. **The agent drafts these positions and proposes labels from EIV; a human team member must review and confirm every label** (add `reviewed_by:` in the YAML). These cases feed the case studies, the demo, and the slides.

**DoD:** `dev.jsonl`, `test.jsonl`, `hard_cases.yaml` committed with oracle values; `data/README.md` documents generation commands, seeds, schema (Appendix A.2), difficulty and tag definitions, and the fact that labels in hard cases are human-reviewed; dataset statistics table in `results/dataset_stats.csv`.

---

## 10. Phase 5: LLM layer, single agent, multi-agent (≈ 2 days)

### 10.1 LLM client (`llm/client.py`)

```python
class LLMClient(Protocol):
    def complete(self, *, system: str, user: str, schema: type[BaseModel],
                 agent_name: str, request_tag: str) -> LLMResult: ...
# LLMResult: parsed (BaseModel | None), raw_text, stop_reason, usage dict,
#            latency_s, cached (bool), error (str | None)
```

- `ClaudeVertexClient`: `AnthropicVertex(project_id, region)`; `messages.parse(model, max_tokens=4000, system=[{"type":"text","text":system,"cache_control":{"type":"ephemeral"}}], messages=[{"role":"user","content":user}], output_format=schema, output_config={"effort": EFFORT})`. Check `stop_reason`; on `refusal` or `max_tokens` → `parsed=None, error=...`. Hard timeout 60 s per call.
- **Response cache** (`llm/cache.py`): sqlite keyed by SHA-256 of (model, effort, system, user, schema name, `repeat_idx`). On hit, return the stored result with `cached=True` and the original usage/latency (also record that it was a cache hit). This makes every run resumable and makes the demo's replay mode exact.
- **Budget guard** (`llm/budget.py`): per-run accumulator of tokens and estimated USD using a price table in `experiments/configs/prices.yaml` (fill in current Vertex prices; mark the date). Abort the run cleanly when `--max-usd` is exceeded; the run can be resumed thanks to the cache.
- **FakeLLM:** deterministic, picks by a configurable rule (e.g. highest `p_pot`), returns schema-valid outputs with fake usage. Also a mode that returns invalid JSON / unknown `shot_id` to test the retry + fallback path.
- **Concurrency:** a process-level semaphore (default 4 in-flight calls) + SDK retries (`max_retries=5`).

### 10.2 Output schemas (`llm/schemas.py`, pydantic)

```python
class SingleDecision(BaseModel):
    shot_id: str
    intent: Literal["offense", "position", "safety"]
    rationale: str            # <= 80 words
    confidence: float         # 0..1

class Assessment(BaseModel):
    shot_id: str
    score: int                # 0..10 from this specialist's perspective
    note: str                 # <= 20 words

class SpecialistReport(BaseModel):
    agent: Literal["offense", "position", "safety"]
    assessments: list[Assessment]   # one per candidate shown
    top_choice: str
    summary: str              # <= 60 words

class CoordinatorDecision(BaseModel):
    shot_id: str
    intent: Literal["offense", "position", "safety"]
    rationale: str            # <= 80 words; must reference which specialist views it weighed
    confidence: float
    overruled: list[Literal["offense", "position", "safety"]]   # specialists whose top choice it rejected

class CriticVerdict(BaseModel):        # extension
    approve: bool
    reason: str
    alternative_shot_id: str | None
```

**Validation in code** (not only by schema): `shot_id` must be in the current menu; otherwise retry once with a short corrective user message ("S17 is not a valid id; valid ids are …"). Still invalid → fallback rule (§2.3-7) and `fallback=True` in the log.

### 10.3 Single agent (SA, SA-vote) (`policies/single_agent.py`, `agents/graph_single.py`)

One call per decision. System prompt = rules of 8-ball + **union of the three specialists' strategy guidance** (Appendix B) + output instructions. User message = state summary + candidate table + last-3-shots history. **SA-vote:** 4 independent calls (cache `repeat_idx` 0–3), majority vote on `shot_id`, ties → highest mean confidence. SA-vote's call count matches MA's 4 calls.

### 10.4 Multi-agent (MA) (`policies/multi_agent.py`, `agents/graph_multi.py`)

LangGraph state graph:

```
prepare ──► offense ──┐
        ├─► position ─┼─► coordinator ──► validate ──► (critic, if enabled) ──► END
        └─► safety ───┘
```

- **Offense specialist:** potting probability, called-pocket risk, ball-in-hand exploitation, how many balls can be run.
- **Position specialist:** cue-ball end position, next-shot ease, cluster breaking, speed/spin choice.
- **Safety specialist:** foul/scratch risk, what the opponent gets if we miss, when a safety beats a low-percentage pot, endgame risk on the 8.
- All three run **in parallel** on the same user message; each sees only its own system prompt.
- **Coordinator:** sees the state summary, the candidate table, and the three `SpecialistReport`s; weighs them by game context (balls left, who is closer to the 8, ball in hand). Returns `CoordinatorDecision`.
- **Critic (extension, MA+critic):** sees the decision + candidate table; vetoes only on clear blunders (e.g. `p_scratch` > 0.3 when an alternative < 0.1 exists, playing at the 8 early). On veto, the coordinator is called once more with the critic's reason. Max one veto per decision.
- **Prompt caching:** each agent's static system prompt is cached separately; long-game context stays short because each decision is stateless apart from the 3-shot history (this is the proposal's "caching to keep prompts short over long games").

### 10.5 Prompt iteration protocol (dev split only)

1. Run SA and MA on the 60 dev positions with FakeLLM (pipeline check), then with Haiku (cost ≈ cents).
2. Read 15–20 traces per system: invalid ids? ignoring annotations? specialists all agreeing trivially? coordinator just copying offense?
3. Fix prompts symmetrically (any knowledge added to a specialist is also added to SA's prompt). At most 3 iteration rounds; record each round's dev metrics in `results/prompt_iterations.csv` (this is the report's "optimization procedure" evidence).
4. Freeze: `git tag prompts-v1`.

**DoD:** unit tests for prompt rendering (all candidates present, ids correct, no `None` leaks), schema validation, retry + fallback via FakeLLM bad modes, LangGraph fan-out/join produces 3 reports + decision with FakeLLM, budget guard aborts at the limit, cache hit returns identical decision; an integration test plays one full game SA vs MA with FakeLLM; a real-API smoke run of 5 dev positions for SA and MA (human-approved, < US$0.50) with traces saved to `results/smoke/`.

---

## 11. Phase 6: experiments

All experiments are config files in `experiments/configs/`, run by scripts that write to `results/<exp_id>/` (`decisions.jsonl` or `games.jsonl`, `summary.csv`, `config_used.yaml`, `cost.json`, git commit hash, model ID, prompt tag). Run cheap decision-level experiments first; full games second.

### 11.1 Experiment matrix

| ID | What | Systems | Data | Setting | Answers | Priority |
|---|---|---|---|---|---|---|
| E0 | Baselines + noise calibration | B0, B1, B2 | 100 paired games per pairing | noise low/med/high | sanity | P0 (done in Phase 3) |
| **E1** | **Main decision-level comparison** | B0, B1, B2, SA, MA, SA-vote | test positions (180) + hard cases (25) | L2, medium noise | RQ1, RQ2 | **P0** |
| **E2** | **Main game-level comparison** | SA vs MA head-to-head; SA vs B2; MA vs B2 | 25 seeds × 2 seat orders = 50 games per pairing | L2, medium noise, full rack from break | RQ1 | **P0** |
| E3 | Mid-game game starts | SA vs MA | 20 test positions × 2 seat orders (game continues from the position) | L2, medium | RQ1 (lower variance) | P1 |
| E4 | Simulation-access ablation | SA, MA | test positions | L0, L1, L2 (+ L3 if built) | RQ3 | **P0** (L0/L2), P1 (L1), P2 (L3) |
| E5 | Noise ablation | SA, MA, B2 | test positions re-annotated + oracle recomputed per noise level; 20 paired games at low/high | noise low/high | RQ3 | P1 |
| E6 | Agent ablations | MA without position; without safety; without offense (coordinator gets 2 reports) | test positions | L2, medium | RQ2 | P1 |
| E7 | Cost/effort trade-off | SA, MA at effort `low` vs `medium` | test positions | L2 | RQ2 | P2 |
| E8 | Model confirmation | SA, MA with Sonnet 5.5 | test positions (or a 90-position stratified subset) + 10 paired games | L2, medium | robustness | P1 |
| E9 | Critic extension | MA+critic vs MA | test positions + hard cases | L2 | extension | P2 |
| E10 | Decision variance | SA, MA | 30 test positions × 3 repeats | L2 | reliability | P1 |

### 11.2 Protocol details

- **Paired games:** for seed *s*, game A has system X as player 1 (breaks), game B swaps seats; both use the same break jitter and the same CRN noise stream. Unit of analysis = seed pair (score ∈ {0, 0.5, 1, 1.5, 2} for X).
- **Positions:** each system decides once per position (E10 handles repeats). The executed outcome is not needed; EIV of the chosen candidate is looked up from the oracle.
- **Parallelism:** games across processes (e.g. 6 workers); LLM concurrency capped globally.
- **Resumability:** each finished game/decision is appended immediately; rerunning skips done items (key = exp id + seed/position id + seat + repeat).
- **Cost logging:** per decision: calls, input/output/cached tokens, USD estimate, wall-clock latency (end-to-end and per agent).

---

## 12. Phase 7: analysis, statistics, figures, failure analysis

### 12.1 Metrics (`eval/metrics.py`)

Game-level: win rate (draw = 0.5), paired score difference, pot success rate (called ball potted / pot attempts), balls per inning, fouls per game by type, scratch rate, safety rate and safety success (opponent fails to pot on the next shot), game length, fallback rate.
Decision-level: mean regret, top-1 accuracy, normalised score, intent accuracy on hard cases, agreement with B2, broken down by difficulty and situation tag.
Cost: LLM calls/decision, tokens in/out/cached per decision and per game, USD per decision and per game, latency p50/p95 (per agent and end-to-end), cache hit rate.
Multi-agent behaviour: how often specialists disagree; how often the coordinator overrules each specialist; which specialist's top choice the coordinator follows, by situation tag.

### 12.2 Statistics (`eval/stats.py`)

- Win rates: Wilson 95 % CI.
- Paired games: bootstrap CI over seed pairs (10,000 resamples) of the mean paired score; two-sided sign test on seed-pair outcomes.
- Positions: paired Wilcoxon signed-rank on per-position regret (SA vs MA, SA-vote vs MA); bootstrap CI of the mean regret difference; effect size (rank-biserial).
- Multiple comparisons: Holm correction over the primary family {MA vs SA, MA vs SA-vote, MA vs B2} on regret.
- **Report sample sizes and the minimum detectable difference.** With 50 games per pairing, only large win-rate gaps (~25 points) are detectable; say so, and lean on the positions benchmark for precision.

### 12.3 Figures (`viz/figures.py`, all generated from `results/`)

1. Architecture diagram (SA vs MA pipeline, shared candidate generator) for slides and report.
2. Bar chart: mean regret ± CI per system (E1), with B0/B1/B2 reference lines.
3. Regret by situation tag (offense / position / defense) per system (RQ2).
4. Win-rate table + forest plot of paired score differences (E2, E3).
5. Cost–quality Pareto plot: normalised score vs USD per decision (and vs latency) for SA, SA-vote, MA, MA ablations (RQ2).
6. Simulation-access ablation: regret at L0 / L1 / L2 for SA and MA (RQ3).
7. Noise ablation: regret / win rate vs noise level; safety rate vs noise level (RQ3).
8. Specialist influence: heatmap of "coordinator followed specialist X" by situation tag.
9. Example table renders with candidate arrows for case studies.

### 12.4 Failure analysis (`docs/failure_analysis.md`)

Sample the 30 highest-regret decisions for SA and MA each. Tag each with a failure category: (a) ignored a large scratch/foul risk; (b) chose a pot when a safety dominated; (c) over-cautious safety when an easy pot existed; (d) misread position/zone; (e) coordinator followed the wrong specialist / averaged incompatible advice; (f) invalid output → fallback; (g) generator limitation (best shot not in menu, e.g. a bank). Report counts per system and 3–4 annotated examples with traces. For each category say what we changed or would change.

**DoD:** `python experiments/analyze.py` regenerates every table and figure from `results/` in one command; `results/summary.md` lists the headline numbers with CIs and the exact files they came from.

---

## 13. Phase 8: demo UI, sample runs, presentation, report

### 13.1 Demo app (`demo/app.py`, Streamlit + Plotly)

Keep it focused (the rubric rewards simplicity and reliability over features). One page, a sidebar, three tabs:

**Sidebar:** model (Haiku/Sonnet), mode `Live` / `Cached replay`, noise preset, simulation access level (L0/L2), cost spent this session.

**Tab 1: "Decide a shot" (core live demo)**
1. Choose a position: dropdown of curated demo positions (`demo/sample_inputs.json`) or "random mid-game position (seed)".
2. Table view (Plotly top-down 2D table, diamonds, pockets, numbered balls coloured by group). Candidate shots drawn as faint arrows; hovering a row in the candidate table highlights its arrow.
3. Button "Run Single agent and Multi-agent" → spinner with stage text ("Offense agent thinking…"). Side by side: SA card (choice, intent, rationale, tokens, latency) vs MA cards (three specialist cards with top choice, scores, summary; coordinator card with choice, rationale, overruled list).
4. Button "Execute shot" for either choice → animated trajectory (pooltool `continuous=True`, frames rendered as a Plotly animation) with the noise draw shown, and the outcome (potted / foul / turn).
5. Oracle panel (toggle): EIV of each candidate, so the audience sees whether the choice was good.

**Tab 2: "Replay a game":** load a logged game (`results/e2_*/games.jsonl` or a demo game), step through shots with prev/next/play, show each decision's trace (agent cards) and the animation. Needs no API.

**Tab 3: "Results":** the 4–5 headline figures and the summary table, loaded from `results/`.

**Reliability requirements:** every live call has a 45 s timeout and on failure falls back to the cached response for that position, **clearly labelled "cached result"**; the app starts without network access (replay and results tabs work offline); errors show a friendly message, never a stack trace; precompute and cache LLM responses for all demo positions the day before (so even "live" will usually be instant, and the label says whether it was a fresh call or cached; be honest about it in the demo).

**Launch:** `streamlit run demo/app.py`. Smoke test: `tests/smoke/test_demo_app.py` using `streamlit.testing.v1.AppTest` with FakeLLM: app loads, a position renders, a decision runs, replay tab loads a fixture game.

### 13.2 Sample runs for the class presentation (`demo/sample_inputs.json`)

Pick from **real logs after E1/E2**, never invent them. Mine `results/` for positions that illustrate:
- **S1 Agreement on an easy pot:** both systems pick the same obvious shot (shows the pipeline; fast).
- **S2 Position play:** MA picks a slightly harder pot with a much better leave, SA picks the easiest pot (only if such a case really exists; otherwise show the reverse honestly).
- **S3 Safety:** no good pot; the safety specialist drives the decision.
- **S4 Failure case:** a high-regret MA decision (e.g. coordinator followed the wrong specialist), to show honest error analysis.
- **S5 Endgame on the 8:** replay of the final shots of a real E2 game.
For each: position id, why it was chosen, expected talking points, and the cached traces. Write `scripts/export_demo_assets.py` to export PNGs and GIFs of each sample run for the slides.

**Backup recording:** a 2–3 minute screen recording of the real app running S1–S3 and a replay, made by a human on 2026-12-05, labelled "recorded backup" when shown.

### 13.3 Demo script (≈ 4–5 minutes; rehearse twice)

1. (20 s) The user/task: an AI pool strategist that picks shots; physics by code, judgment by LLM.
2. (60 s) S1 live: show candidates, SA vs MA cards, execute the shot.
3. (90 s) S2 or S3 live: point at the specialist that changed the decision; show the oracle panel.
4. (60 s) Replay tab: last shots of a real game with traces.
5. (30 s) Results tab: one headline chart; latency/cost numbers from measurements.
Pre-flight checklist (run 1 hour before): `scripts/check_llm.py`, app smoke test, cache present, laptop on power, Streamlit already launched, backup recording on the desktop.

### 13.4 Presentation slides (≈ 12–14 slides; map to the rubric's 3 × 10 %)

**Problem statement (10 %)**
1. Title, team, one-line claim.
2. Problem domain: 8-ball in 60 s; why sequential decisions with long-term consequences matter for agent systems in general.
3. State of the art and limits: PoolBench (LLMs output raw angles → 0 pots in 300 shots), CueTip (LLM explains a neural policy; LLM-chosen play wins only 26–32 %), MAD paper (multi-agent often fails to beat a strong single agent).
4. Why GenAI here, and what a non-generative approach (B2 search) already does; research questions RQ1–3 and hypotheses H1–H4; scope (simulated play, one ruleset, one table, no human study).

**Major contributions (10 %)**
5. Architecture: shared candidate generator + simulation annotations + SA vs MA (diagram); why choose-by-`shot_id`.
6. The fairness contract (same candidates, knowledge, noise, budget, compute-matched control): this is our methodological contribution.
7. Benchmark: positions dataset with oracle EIV, difficulty and situation tags, hand-picked hard cases; paired-seed game harness with common random numbers.
8. Engineering: LangGraph fan-out, structured outputs, caching, cost guard, failure handling; measured latency and cost.

**Evaluation (10 %)**
9. Main results: regret and win rate with CIs (E1, E2) vs B0/B1/B2.
10. RQ2: where MA helps (by situation tag) + cost–quality Pareto.
11. RQ3: simulation access and noise ablations.
12. Case studies: one success, one failure (with traces); error-analysis categories.
13. Limitations and future work (§17), prioritised by what we observed.
14. Demo (switch to the app).

### 13.5 Final report (ACM Master Article template, LaTeX; due 2026-12-15)

Sections and what fills them (all numbers from `results/summary.md`):
1. **Abstract.**
2. **Introduction:** problem, RQs, contributions (bulleted), headline finding.
3. **Related work:** pool AI and computational billiards (Archibald et al.; pooltool), LLMs in pool (PoolBench, CueTip), LLM multi-agent debate/collaboration and its critiques (Smit et al.), tool-augmented LLM agents (code computes, LLM judges). *Critically compare*: say what each assumes and how it informed our design.
4. **Method:** environment and rules; candidate generator and annotation levels; systems (B0–B2, SA, SA-vote, MA, critic); prompts (in an appendix); fairness contract.
5. **Benchmark and experimental setup:** dataset construction, splits, oracle definition, hard cases (human-reviewed), game protocol, noise model, models and versions, effort, costs, hardware.
6. **Results:** E1, E2 (RQ1); situation breakdown, ablations, cost (RQ2); access and noise (RQ3); variance (E10); model confirmation (E8).
7. **Analysis and discussion:** failure categories, why results occurred, implications for multi-agent design, when a single agent is preferable.
8. **Limitations:** simulated noise ≠ human play; one model family; oracle approximates game value; candidate generator ceiling; no user study (say explicitly that our evaluation is scenario-based, not user-based).
9. **Future work** (§17). 10. **Conclusion.** **References.** **Appendix:** prompts, schemas, extra tables, TA-feedback response.
Also include a short "**Response to TA feedback**" paragraph (the rubric checks it).

### 13.6 TA feedback

`docs/ta_feedback.md` must record the feedback from the October TA meeting and how we addressed it. **The agent does not know this feedback: ask the human to paste it in.** Reference it in the report and on the limitations/contributions slides.

### 13.7 README and tutorial (rubric: 5 % for the tutorial)

`README.md` sections: overview + architecture figure; prerequisites (Python 3.12, uv, GCP project with Vertex AI Claude access, service-account JSON outside the repo); install commands (tested on a clean clone); `.env` setup; **quick start in 3 commands** (tests, one baseline game, launch the demo in cached mode without any API key); running experiments (one command per experiment, expected runtime and cost); reproducing figures; sample inputs and expected outputs; costs (per experiment, from `cost.json`); known limitations; troubleshooting (Panda3D index, Numba first-run delay, `PYTHONIOENCODING`, Vertex 429/403 errors, missing cache); licence and third-party credits (pooltool Apache-2.0).

---

## 14. Schedule, work split, cut-lines

Today is **Fri 2026-11-20**. 17 days to the demo. Member roles (adjust names as the team decides):
- **M1 Simulation & data:** Phases 1, 2, 4 (engine, candidates, annotations, dataset, oracle).
- **M2 Agents:** Phase 5 (LLM layer, prompts, SA/MA graphs, prompt iteration), E4/E6/E7 runs.
- **M3 Evaluation & UI:** Phase 3, tournament harness, stats/figures (Phase 7), demo app (13.1).
Everyone: slides and report sections for their part; daily 15-minute sync; one integration merge per day.

| Dates | M1 | M2 | M3 | Milestone / gate |
|---|---|---|---|---|
| Nov 20–21 | Phase 0 port; Phase 1 engine | LLM client, cache, budget, FakeLLM, schemas | Runner logging schema, metrics/stats skeleton with tests | **G1:** full B1-vs-B1 game runs; `check_llm.py` passes |
| Nov 22–23 | Phase 2 generator + L0/L1/L2 annotations | Prompts v0, SA graph, MA graph on FakeLLM | B0/B1/B2 policies; E0 overnight | **G2:** E0 done, noise fixed |
| Nov 24–25 | Phase 4 dataset build + oracle (overnight) | Dev-set real runs, prompt iteration rounds 1–2 | Tournament runner (paired, CRN, resumable, parallel); viz table plot | **G3:** dataset frozen; SA/MA run end-to-end on dev |
| Nov 26–27 | Hard cases drafted; human review | Prompt round 3; **freeze `prompts-v1`** | Demo app v0 (tabs 1–2 with FakeLLM) | **G4:** prompts frozen, hard-case labels reviewed |
| Nov 28–30 | Support runs; L1 / noise re-annotation for E5 | **E1, E4 (L0/L2)** on test | **E2** games (SA vs MA, SA vs B2, MA vs B2) | **G5:** P0 experiments complete |
| Dec 1–2 | E5, E10 | E6, E3 | Analysis script, headline figures, results tab | **G6:** `results/summary.md` with CIs |
| Dec 3–4 | Failure analysis with M2 | E8 subset with Sonnet (budget approval) | Demo v1: caching of demo positions, sample runs S1–S5, export GIFs | **G7:** demo feature-complete |
| Dec 5–6 | Slides (problem, benchmark) | Slides (architecture, agents) | Slides (results); backup recording | **G8:** two full rehearsals with timer |
| **Dec 7** | | | | **Presentation + live demo** |
| Dec 8–11 | Report: method, benchmark | Report: agents, related work; E9 critic if time | Report: results, figures; repo cleanup, README/tutorial | |
| Dec 12–14 | Fresh-clone install test on another machine; final edit pass for one voice; check every number against `results/` | | | **G9:** tests green, README verified |
| **Dec 15** | | | | **Report + repo submitted** |

**Cut-lines if behind** (drop in this order, and record the drop in the report's limitations): E9 critic → L3 on-demand tool → banks → E7 effort sweep → E8 reduced to positions only → E3 mid-game games → E5 games (keep E5 positions) → reduce E2 to 15 seeds × 2. **Never cut:** E1, E2 (≥ 15 seed pairs), E4 L0-vs-L2, the fairness contract, tests, the live demo.

---

## 15. Testing strategy

| Layer | What | Examples |
|---|---|---|
| Unit | Pure functions and small components | state round-trip (`to_system(from_system(s)) == s`); validation rejects overlaps/NaN; noise determinism and CRN; candidate determinism + hash; pruning keeps ≤ 15 and only valid ids; descriptions contain every id; schema parsing; budget guard; cache hit/miss; stats functions against known values (Wilson CI for 7/10, bootstrap on a toy set) |
| Integration | Components together, FakeLLM only | full game B1 vs B2; full game SA vs MA with FakeLLM; invalid-output FakeLLM triggers retry then fallback and is logged; positions runner on 5 dev positions; tournament resume after a simulated crash |
| Regression | Pin upstream quirks and earlier bugs | pooltool `b=0.25` default; substring `is_ball_pocketed`; pocketed balls in occlusion; ball-in-hand overlap |
| Smoke | Demo and entry points | `AppTest` for `demo/app.py`; each `experiments/*.py --dry-run` parses its config and processes one item with FakeLLM |
| Live (manual, marked `@pytest.mark.live`, skipped by default) | Real API | `check_llm.py`; 5-position real smoke run |
| Fairness | The contract in §2.3 | in a paired run, candidate-list hashes are identical across systems for each decision; SA prompt contains every strategy paragraph present in specialist prompts; noise draws equal at equal (seed, shot index) |

Use `pytest-timeout` (60 s default; long tests marked `slow`). Target: the default suite runs in < 3 minutes.

---

## 16. Risk register

| # | Risk | Likelihood | Impact | Mitigation / fallback |
|---|---|---|---|---|
| R1 | We are behind schedule (only the spike exists) | High | High | Compressed plan, parallel tracks, daily gates, explicit cut-lines (§14); decision-level benchmark gives results even if full games are few |
| R2 | MA does not beat SA (or loses) | Medium | Medium | Framed as a controlled comparison from the start; negative results with CIs and a cost analysis are valid findings; RQ2 breakdown still informative |
| R3 | Everything ties because the candidate menu makes choices easy | Medium | High | Hard cases, situation tags, difficulty strata; regret metric is sensitive even when win rates tie; check in E0/dev that B0 vs B2 differ substantially (if not, the menu is too easy and pruning or noise must change) |
| R4 | LLM cost or rate limits (Vertex 429s) | Medium | Medium | Haiku for bulk, cache, budget guard, bounded concurrency, resumable runs; request a Vertex quota increase early |
| R5 | Non-reproducible LLM outputs (no temperature control) | Certain | Low–Med | Cache for exact replays; E10 measures variance; report it |
| R6 | pooltool rules limitations (no real ball-in-hand, no ruleset copy, mid-game start) | Medium | Medium | Shared ball-in-hand policy; `is_legal` on copied systems; `rules_lite.py` fallback decided early in Phase 1 |
| R7 | Simulation too slow for L2 + oracle | Low–Med | Medium | Process-pool rollouts; reduce N to 6 or K to 10; oracle overnight; break capped with `max_events` |
| R8 | Prompt overfitting to test data | Medium | High | Dev/test split, frozen prompts, git tag, symmetric prompt edits |
| R9 | Unfair comparison (knowledge or compute asymmetry) | Medium | High | Fairness contract + fairness tests; SA-vote compute-matched control |
| R10 | Live demo failure (network, API, laptop) | Medium | High (30 % of grade) | Cached responses with honest labels; offline replay tab; backup recording; pre-flight checklist; rehearsals |
| R11 | Oracle EIV is a poor proxy for winning | Medium | Medium | Validate: correlation between position-level regret and game outcomes; report both metrics; state the limitation |
| R12 | Credential leak | Low | High | Key outside repo, `.env` git-ignored, pre-commit check for `private_key` strings |
| R13 | Team integration conflicts | Medium | Medium | Shared data contracts (Appendix A) fixed on day 1; daily merges; tests as the contract |
| R14 | LLM ignores annotations or hallucinates ids | Medium | Medium | Code-side id validation, retry, fallback, fallback-rate metric; descriptions in plain language |

---

## 17. Future work (prioritise by what the results show; keep this list honest)

1. **Richer action space with LLM-requested variants:** let agents propose a modified shot ("S04 softer with more draw"), simulated on demand (the L3 tool, extended).
2. **Learned or search-based value function** combined with LLM judgment (e.g. LLM chooses among top-k of a deeper search), and comparison with CueTip-style neural policies.
3. **Better generator:** banks, kicks, combination shots, multi-cushion position play: the generator's ceiling bounds both systems.
4. **Opponent modelling and match-level strategy** (adapting to the opponent's skill).
5. **Open-weight models and smaller models** to test whether specialization helps weaker models more (the proposal's original model plan).
6. **Debate/critique variants:** multi-round discussion, critic agent, confidence-weighted aggregation, and their cost curves.
7. **Human-in-the-loop coaching** UI (explain the chosen shot to a learner) and a small user study of explanation usefulness.
8. **Real-table transfer:** detect a real table from a photo and recommend shots; calibrate the noise model to human data.
9. **Generalisation:** other cue sports (9-ball) and other sequential decision games, to see whether the specialization result transfers.

---

## 18. Definition of done (final checklist mapped to the rubric)

**Presentation (30 %)**
- [ ] Problem, background, SOTA limits, GenAI justification, target users (coaches/learners, agent-system researchers)
- [ ] RQs, hypotheses, assumptions, scope stated
- [ ] Architecture with reasons for design choices; integration story (formats, failures, context)
- [ ] Results vs baselines with CIs; ablations; cost/latency trade-offs; case studies incl. a failure; limitations & prioritised future work

**Demo (30 %)**
- [ ] Live end-to-end run on real inputs; trace visible; outcome shown
- [ ] No crashes/hangs; clear loading states; graceful fallback with honest labels
- [ ] Backup recording prepared and labelled
- [ ] Latency claims match measured numbers

**Repository (15 %)**
- [ ] Modular `src/` (core), `demo/` (UI), `experiments/` (eval) separation; documented interfaces
- [ ] Unit, integration, smoke tests pass; error handling for invalid inputs and API failures
- [ ] Data + generation instructions; configs, model IDs, prompts, seeds recorded; results with scripts that regenerate them
- [ ] Tested README/tutorial: setup, demo launch, experiment commands, costs, troubleshooting; no secrets committed

**Report (15 %)**
- [ ] ACM template; logical flow; proper citations (pooltool, PoolBench, CueTip, Archibald et al., Smit et al., LangGraph, Claude models)
- [ ] Critical related-work comparison; methodology detailed enough to reproduce; prompt-iteration procedure described
- [ ] Results with figures and uncertainty; implications within the evidence; TA feedback addressed
- [ ] Final editing pass: one voice, consistent terminology, every number traced to `results/`

---

## Appendix A: data contracts

### A.1 `TableState`

```json
{
  "schema_version": 1,
  "game_id": "e2_seed17_A",
  "shot_number": 12,
  "shooter": "P1",
  "shooter_group": "solids",            // open | solids | stripes
  "ball_in_hand": false,
  "cue": [0.4953, 0.6000],              // metres, pooltool frame (origin bottom-left)
  "balls": {"1": [0.31, 1.20], "3": [0.72, 1.55], "8": [0.50, 1.65], "11": [0.20, 0.90]},
  "legal_targets": ["1", "3"],
  "remaining": {"P1": ["1", "3"], "P2": ["11"]},
  "history": [{"shooter": "P2", "shot_id": "S03", "intent": "offense", "outcome": "missed"}]
}
```

### A.2 Position record (`data/positions/*.jsonl`)

```json
{
  "position_id": "test_0042",
  "split": "test",
  "source": "midgame_sample",            // midgame_sample | random | hand
  "difficulty": "medium",
  "tags": ["position"],
  "state": { "...": "TableState" },
  "noise_preset": "medium",
  "candidates_hash": "sha256:…",
  "candidates": [ { "...": "Candidate (A.3) with L0/L1/L2 annotations" } ],
  "oracle": {"S01": {"eiv": 1.42, "se": 0.11}, "S02": {"eiv": 0.35, "se": 0.09}},
  "expected_intent": null,               // set for hand cases only
  "acceptable_shot_ids": null,           // hand cases only
  "reviewed_by": null                    // hand cases only
}
```

### A.3 Candidate

```json
{
  "shot_id": "S04",
  "shot_type": "pot",                    // pot | safety | bank | contact_only
  "target_ball": "3",
  "pocket": "rt",
  "pocket_label": "top_right",
  "phi_deg": 63.214, "V0": 2.5, "a": 0.0, "b": -0.3, "theta": 0.0,
  "power_label": "medium", "spin_label": "draw",
  "geometry": {"cut_deg": 22.0, "cue_to_ball_m": 0.71, "ball_to_pocket_m": 0.44,
               "required_precision_deg": 1.8, "geometry_score": 0.62},
  "preview": {"pots": true, "scratch": false, "foul": null, "cue_end": [0.52, 1.02], "next_viable_pots": 3},
  "rollouts": {"n": 8, "p_pot": 0.81, "p_scratch": 0.02, "p_foul": 0.02, "p_keep_turn": 0.79,
               "cue_end_mean": [0.51, 1.00], "cue_end_sd_m": 0.06,
               "next_best_p_pot_geom": 0.74, "opp_threat_if_turn_passes": 0.66, "cluster_contact": false},
  "description": "Pot 3 into top-right corner, medium draw; cue ends centre table leaving 5 to the side pocket."
}
```

### A.4 Decision log record (one JSON line per shot in `games.jsonl` / `decisions.jsonl`)

```json
{
  "exp_id": "e2", "game_id": "e2_seed17_A", "position_id": null, "seat": "P1", "system": "MA",
  "model": "claude-haiku-5-5", "effort": "medium", "prompt_tag": "prompts-v1", "git_commit": "abc1234",
  "state_hash": "sha256:…", "candidates_hash": "sha256:…", "access_level": "L2", "noise_preset": "medium",
  "agents": [
    {"agent": "offense", "request_hash": "…", "parsed": {"…": "SpecialistReport"}, "raw_text": "…",
     "usage": {"input": 2310, "output": 420, "cache_read": 1800, "cache_write": 0},
     "latency_s": 3.1, "cached": false, "retries": 0, "error": null}
  ],
  "decision": {"shot_id": "S04", "intent": "position", "rationale": "…", "confidence": 0.7, "overruled": ["offense"]},
  "fallback": false,
  "executed": {"phi_deg": 63.61, "V0": 2.41, "a": 0.01, "b": -0.31},
  "outcome": {"potted": ["3"], "legal": true, "reason": null, "turn_over": false, "game_over": false},
  "cost_usd": 0.0041, "decision_latency_s": 6.8
}
```

---

## Appendix B: prompt templates (starting point; iterate on dev only, then freeze)

**Shared rules block** (all agents and SA): 8-ball rules in 8 lines (groups, open table, legal hit, fouls, ball in hand, the 8 last, losing conditions); an explanation of every candidate field (`p_pot`, `p_scratch`, `p_keep_turn`, `next_best_p_pot_geom`, `opp_threat_if_turn_passes`), and the coordinate/zone system; "choose only from the listed shot ids; never invent shots".

**Strategy knowledge blocks** (each specialist gets its own; SA gets all three, verbatim):
- *Offense:* prefer high-probability pots; account for the called pocket; with ball in hand, take the easiest high-percentage shot; count how many balls can realistically be run.
- *Position:* the best shot is often the one that leaves the next shot easy; avoid leaving the cue ball on a rail or behind balls; break clusters early when the risk is low; prefer stun/natural roll over extreme spin unless the leave clearly improves.
- *Safety:* if the best pot is below ~40–50 % and a safety leaves the opponent little, play safe; avoid scratches; near the end (opponent on the 8) be more defensive; never risk pocketing the 8 early.

**Specialist user message:** state summary, candidate table, last 3 shots, then: "From the **{offense|position|safety}** perspective only, score every candidate 0–10, give a ≤ 20-word note each, name your top choice, and summarise in ≤ 60 words."

**Coordinator user message:** state summary, candidate table, the three specialist reports (JSON), then: "Weigh the specialists' views against the game situation. Choose one shot id. Say which views you followed and which you overruled, and why, in ≤ 80 words."

**SA user message:** state summary, candidate table, last 3 shots, then: "Consider offense, position, and safety. Choose one shot id and explain in ≤ 80 words."

Keep instructions short and concrete; do not add "think step by step" boilerplate (thinking is adaptive). Version prompts as files in `src/cuemind/llm/prompts/` and log their hash with every decision.

---

## Appendix C: cost and runtime budget (estimates; replace with measured numbers)

Assumptions: ~2.5k input tokens per call (much of it cacheable system prompt), ~0.5–1.5k output tokens incl. adaptive thinking; SA = 1 call/decision, SA-vote = 4, MA = 4 (5 with critic); ~20–30 LLM decisions per player per full game. Prices: fill `prices.yaml` from the current **Vertex AI** Claude pricing page (Vertex pricing is set separately from Anthropic's first-party list; first-party list price for Haiku 5.5 is $0.10 / $0.50 per MTok in/out and Sonnet 5.5 is $2 / $10).

| Experiment | LLM decisions (approx.) | Calls (approx.) | Haiku est. | Wall-clock est. |
|---|---|---|---|---|
| Dev prompt iteration (3 rounds × 60 pos × SA+MA) | 360 | 900 | < $1 | 1 h |
| E1 (205 positions × SA, MA, SA-vote) | 615 | 1,850 | ~$2 | 1–2 h |
| E2 (150 games, LLM sides) | ~6,000 | ~15,000 | ~$10–15 | 6–10 h with 6 workers |
| E4 (L0, L1 × SA, MA × 180) | 720 | 1,800 | ~$2 | 1–2 h |
| E5–E7, E10 | ~2,500 | ~7,000 | ~$5 | 3–5 h |
| E8 Sonnet (90 pos + 10 games) | ~700 | ~2,000 | ~$40–80 (Sonnet) | 2–3 h |

Simulation: L2 annotation ≈ 2–4 s per decision with parallel rollouts; oracle ≈ 30–60 min on 8 cores for 180 positions; non-LLM games ≈ 30–90 s each. **The agent must measure actual cost on the first 20 decisions of each experiment and update this table before continuing.**

---

## Appendix D: command cheat-sheet (target interface; implement these)

```bash
# setup
uv venv --python 3.12 .venv && uv pip install -r requirements.txt --extra-index-url https://archive.panda3d.org/simple/ --prerelease=allow --index-strategy unsafe-best-match
cp .env.example .env            # then edit paths/IDs
python scripts/check_llm.py     # one real call, prints usage

# tests
pytest -q                       # unit + integration + smoke (FakeLLM only)
pytest -q -m live               # real API, manual

# baselines and dataset
python experiments/run_baselines.py --config experiments/configs/e0.yaml
python experiments/build_dataset.py --config experiments/configs/dataset.yaml   # + oracle, overnight

# LLM experiments (always with a budget)
python experiments/run_positions.py  --config experiments/configs/e1.yaml --max-usd 5
python experiments/run_tournament.py --config experiments/configs/e2.yaml --max-usd 20 --workers 6
python experiments/run_positions.py  --config experiments/configs/e4.yaml --max-usd 5

# analysis and demo
python experiments/analyze.py               # all tables/figures -> results/
python scripts/export_demo_assets.py        # PNG/GIF for slides
streamlit run demo/app.py                   # demo (use "Cached replay" mode offline)
```

---

*End of plan. When in doubt: keep the comparison fair, keep the demo reliable, and make every number traceable to a file.*
