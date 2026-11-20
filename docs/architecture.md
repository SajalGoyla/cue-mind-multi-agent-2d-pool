# Architecture

This document explains how CueMind is put together and how information flows through one decision. It is kept up to date as components land; sections marked *(planned)* are not built yet. Design rationale lives in [`decisions.md`](decisions.md); the overall plan in [`EXECUTION_PLAN.md`](EXECUTION_PLAN.md).

## 1. One decision, end to end

```
TableState (positions, groups, shooter, ball-in-hand)
   │
   ├─ ball in hand? ─► ball_in_hand.place_cue_ball()   shared, non-LLM (D-005)
   │
   ▼
candidates.pipeline.prepare(state, PrepConfig)        identical for every policy
   ├─ generate.generate()     pots × power × spin, safeties, contact fallback (D-006)
   ├─ annotate (L0/L1/L2)     geometry / one preview / 8 noisy rollouts (D-010)
   ├─ prune                   ≤ 12 pots + 3 safeties, ids S01..S15 (D-007, D-010)
   └─ describe                plain-language rows for prompts and UI (D-012)
   │
   ▼
Policy.decide(DecisionContext) -> Decision(shot_id, intent, rationale, trace, usage)
   B0 random │ B1 geometric │ B2 search (D-011) │ SA │ SA-vote │ MA │ MA+critic (D-016, D-017)
   │                                    invalid id / exception → shared fallback (D-009)
   ▼
NoiseModel.apply(stroke, rng_for(seed, shot_index, "execute"))   common random numbers (D-004)
   │
   ▼
sim.execute.simulate() ─► extract_events() ─► rules.judge() ─► rules.advance()
   │                                                               │
   └──────────── one JSON record per shot (game log) ◄─────────────┘
```

## 2. Modules

| Module | Responsibility | Key types / functions |
|---|---|---|
| `sim/state.py` | Position representation, validation, hashing, conversion to/from pooltool | `TableState`, `validate`, `state_hash`, `to_system`, `positions_from_system`, `rack_state` |
| `sim/execute.py` | Run a stroke in pooltool (on a copy, with retry, D-014); summarise rule-relevant events | `Stroke`, `simulate`, `SimulationError`, `extract_events`, `ShotEvents`, `warm_up` |
| `sim/noise.py` | Execution error model and deterministic RNG streams | `NoiseModel`, `NOISE_PRESETS`, `rng_for` |
| `game/rules.py` | Simplified 8-ball rules | `Call`, `Judgement`, `judge`, `advance` |
| `game/break_shot.py` | Seeded break stroke | `break_stroke` |
| `game/ball_in_hand.py` | Shared cue-ball placement | `place_cue_ball`, `resolve_ball_in_hand` |
| `game/runner.py` | Game loop, fallback handling, JSONL logging, CLI | `play_game`, `GameResult` |
| `candidates/direct.py` | Ghost-ball pot geometry (pooltool `ai.pot`) | `direct_pots`, `PotGeometry.ease` |
| `candidates/model.py` | Candidate data model | `Candidate`, `Geometry`, `Preview`, `Rollouts`, `candidates_hash` |
| `candidates/generate.py` | Candidate generation (L0) | `generate`, `pot_candidates`, `safety_candidates`, `contact_fallback` |
| `candidates/annotate.py` | L1 preview and L2 Monte-Carlo statistics | `preview`, `rollouts`, `ease_from` |
| `candidates/describe.py` | Diamond coordinates, zones, candidate and state text | `describe`, `state_summary`, `where` |
| `candidates/pipeline.py` | Prepare the menu (generate → annotate → prune → describe) | `PrepConfig`, `prepare`, `prune_l0`, `prune_annotated` |
| `policies/base.py` | Policy protocol, fallback rule | `DecisionContext`, `Decision`, `Policy`, `fallback_choice` |
| `policies/simple_bots.py` | B0 random, B1 geometric | `RandomBot`, `GeometricBot` |
| `policies/search_bot.py` | B2: max expected value over the annotated menu | `SearchBot`, `SearchWeights`, `expected_value` |
| `policies/registry.py` | Policy names for CLIs/configs; builds LLM clients from config | `make_policy`, `make_llm_client`, `known_policies` |
| `policies/single_agent.py` | SA and SA-vote | `SingleAgent` |
| `policies/multi_agent.py` | MA, MA+critic, MA ablations | `MultiAgent` |
| `agents/graph_multi.py` | LangGraph fan-out/fan-in graph | `build_graph`, `MAState` |
| `agents/common.py` | id validation with one corrective retry; usage totals | `call_with_id_check`, `sum_usage` |
| `llm/client.py` | Claude on Vertex, FakeLLM, cost accounting | `ClaudeVertexClient`, `FakeLLM`, `LLMResult`, `cost_usd` |
| `llm/cache.py` | SQLite response cache and spending ledger | `ResponseCache`, `Budget`, `BudgetExceeded` |
| `llm/schemas.py` | Structured-output schemas | `SingleDecision`, `SpecialistReport`, `CoordinatorDecision`, `CriticVerdict` |
| `llm/promptlib.py` + `llm/prompts/*.md` | Prompt text and assembly | `system_single`, `system_specialist`, `decision_context_text`, `prompt_version` |
| `eval/tournament.py` | Paired, resumable, parallel tournaments | `build_jobs`, `run_tournament` |
| `eval/metrics.py` | Per-seat game metrics, pairing summaries | `seat_metrics`, `pairing_summary` |
| `eval/stats.py` | Wilson, bootstrap, sign test, Wilcoxon, Holm | |

## 3. Coordinate frame and table

pooltool frame, metres. Origin at the bottom-left corner of the cushion-nose rectangle; +x right (0 to W = 0.9906), +y up the table (0 to L = 1.9812). Ball radius R = 0.028575. The break is played from the bottom (head) end at the head spot (W/2, L/4); the rack's apex points down-table from the foot spot (W/2, 3L/4). Pockets: `lb, lc, lt, rb, rc, rt` (left/right × bottom/centre/top); pocket centres lie slightly outside the playing surface. Aim `phi` is in degrees counter-clockwise from +x.

## 4. Game rules (summary of `game/rules.py`)

- **Break:** legal if a ball drops or ≥ 4 object balls reach a cushion; scratch/illegal break = foul. Table stays open. An 8 on the break is re-spotted. Breaker continues only if an object ball dropped.
- **Open table:** any ball except the 8 may be hit first; the group is assigned when the called ball drops in the called pocket.
- **Fouls:** scratch, no contact, wrong first contact, no cushion after contact (unless a ball dropped). Foul → opponent has ball in hand anywhere.
- **Turn:** continues only if the shot is legal, not a declared safety, and the called ball dropped in the called pocket.
- **8-ball:** legal pot (on the 8, called pocket, no foul) wins; any other 8 pot loses.
- **Shot cap:** 150 shots → draw (scored 0.5).

## 5. Game log record

One JSON line per shot (`game/runner.py`). Fields: `exp_id, game_id, seed, shot_index, seat, system, kind (break|decision), state_hash, state, ball_in_hand, prep, candidates_hash, candidates[], decision{shot_id,intent,rationale,confidence}, fallback, trace, usage, planned, executed, outcome{potted, legal, foul, turn_over, game_over, winner, called_ball_made, group_assigned, note, first_ball_hit, truncated}, decision_latency_s, step_s`. These records drive the metrics, the failure analysis, and the demo's replay view.

## 6. Testing

`tests/unit` covers state validation and hashing, simulation determinism and copy isolation, the cue `b = 0.25` default regression, noise determinism and clipping, every rules branch with synthetic events, candidate determinism and safety legality, pruning diversity, and ball-in-hand placement. `tests/integration` plays full games, checks logs, policy-failure fallback, mid-game starts, and reproducibility. Run `pytest -q` (FakeLLM only; live API tests are opt-in with `-m live`).

## 7. Reproducibility

A game is fully determined by (seed, policies, `PrepConfig`, noise preset): the rack (D-008, D-013), break jitter, rollout streams (D-010) and execution noise (D-004) all derive from the seed through `rng_for` (SHA-256 of the keys), never from global or hash-order-dependent state. Verified by running the same L2 game under different `PYTHONHASHSEED` values. LLM policies are the exception (no temperature control); their responses are cached for exact replay (planned, Phase 5).

## 8. LLM decision flow (SA and MA)

```
DecisionContext ──► promptlib.decision_context_text()   position summary + last 3 shots + candidate table
                     │
     SA:  system_single (rules + offense + position + safety knowledge + role) ──► SingleDecision
     MA:  ┌ system_specialist(offense)  ─► SpecialistReport ┐
          ├ system_specialist(position) ─► SpecialistReport ┼─► system_coordinator ─► CoordinatorDecision ─► (critic)
          └ system_specialist(safety)   ─► SpecialistReport ┘
                     │
   every call: cache lookup → budget check → messages.parse (structured output, cached system prompt)
               → id check (one corrective retry) → shared fallback if still invalid
                     │
   Decision.trace: prompt version, every attempt (parsed output, tokens, latency, cache hit), reports,
                   followed/overruled specialists, critic verdict  → logged per shot, shown in the demo
```
