# Architecture (v2)

How CueMind is put together and how information flows through one decision. Design rationale lives in [`decisions.md`](decisions.md); D-029 explains why v2 replaced the positions benchmark with game play.

## 1. One shot, end to end

```
TableState (positions, groups, shooter, ball-in-hand)
   │
   ├─ break? ─► break_shot.break_stroke()             shared, seeded, no policy (D-005)
   ├─ ball in hand? ─► ball_in_hand.place_cue_ball()  shared, non-LLM (D-005)
   │
   ▼
candidates.pipeline.build_menu(state, MenuConfig)    identical for every policy
   ├─ generate.generate()   pots × speed × spin, safeties (legality filter), contact fallback
   ├─ menu rule             4 easiest target/pocket pairs × medium × draw/stun/follow + ≤ 3 safeties
   ├─ seeded shuffle        list order carries no information
   └─ describe              one line per shot: ball, pocket, speed, spin, cut, difficulty word
   │
   ▼
Policy.decide(DecisionContext) -> Decision(shot_id, intent, rationale, trace, usage)
   B   simulates every candidate privately (annotate.rollouts), max expected value (D-011 weights)
   SA  one LLM call
   MA  3 specialists shortlist ≤ 3 ids in parallel → coordinator picks a shortlisted id
   B1  easiest pot (pilot only)
   │                                    exception / unknown id → shared fallback (easiest pot)
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
| `game/break_shot.py`, `game/ball_in_hand.py` | Seeded break; shared cue-ball placement | `break_stroke`, `place_cue_ball`, `resolve_ball_in_hand` |
| `game/runner.py` | Game loop, fallback handling, JSONL logging, CLI | `play_game`, `GameResult` |
| `candidates/direct.py` | Ghost-ball pot geometry (pooltool `ai.pot`) | `direct_pots`, `PotGeometry.ease` |
| `candidates/model.py` | Candidate data model | `Candidate`, `Geometry`, `Rollouts`, `candidates_hash` |
| `candidates/generate.py` | All geometric candidates | `generate`, `pot_candidates`, `safety_candidates`, `contact_fallback` |
| `candidates/pipeline.py` | The raw menu every policy sees | `MenuConfig`, `build_menu` |
| `candidates/describe.py` | Diamond coordinates, zones, candidate and position text | `describe`, `state_summary`, `where` |
| `candidates/annotate.py` | B's private simulation statistics | `rollouts` |
| `policies/base.py` | Policy protocol, context, fallback rule | `DecisionContext`, `Decision`, `Policy`, `fallback_choice` |
| `policies/search_bot.py` | B | `SearchBot`, `SearchWeights`, `expected_value` |
| `policies/simple_bots.py` | B1 | `GeometricBot` |
| `policies/single_agent.py`, `policies/multi_agent.py` | SA, MA | `SingleAgent`, `MultiAgent` |
| `policies/registry.py` | Policy names for CLIs/configs (`b`, `b1`, `sa`, `ma`) | `make_policy`, `make_llm_client` |
| `agents/graph_multi.py` | LangGraph graph for MA | `build_graph`, `MAState` |
| `agents/common.py` | Id validation with one corrective retry; usage totals | `call_with_id_check`, `sum_usage` |
| `llm/client.py` | Ollama client, offline fake LLM | `OllamaClient`, `FakeLLM`, `LLMResult` |
| `llm/cache.py` | SQLite response cache | `ResponseCache`, `request_key` |
| `llm/schemas.py` | Structured-output schemas | `SingleDecision`, `SpecialistShortlist`, `CoordinatorDecision` |
| `llm/promptlib.py` + `llm/prompts/*.md` | Prompt text and assembly | `system_single`, `system_specialist`, `system_coordinator`, `prompt_version` |
| `eval/tournament.py` | Paired, resumable, parallel tournaments | `build_jobs`, `run_tournament` |
| `eval/metrics.py`, `eval/stats.py` | Per-seat game metrics, head-to-head summaries; Wilson, bootstrap, sign test | `seat_metrics`, `pairing_summary` |

## 3. Coordinate frame and table

pooltool frame, metres. Origin at the bottom-left corner of the cushion-nose rectangle; +x right (0 to W = 0.9906), +y up the table (0 to L = 1.9812). Ball radius R = 0.028575. The break is played from the bottom (head) end at the head spot (W/2, L/4); the rack's apex points down-table from the foot spot (W/2, 3L/4). Pockets: `lb, lc, lt, rb, rc, rt` (left/right × bottom/centre/top). Aim `phi` is in degrees counter-clockwise from +x. Prompts use diamonds: x 0–4, y 0–8.

## 4. Game rules (summary of `game/rules.py`)

- **Break:** legal if a ball drops or ≥ 4 object balls reach a cushion; scratch/illegal break = foul. Table stays open. An 8 on the break is re-spotted. Breaker continues only if an object ball dropped.
- **Open table:** any ball except the 8 may be hit first; the group is assigned when the called ball drops in the called pocket.
- **Fouls:** scratch, no contact, wrong first contact, no cushion after contact (unless a ball dropped). Foul → opponent has ball in hand anywhere.
- **Turn:** continues only if the shot is legal, not a declared safety, and the called ball dropped in the called pocket.
- **8-ball:** legal pot (on the 8, called pocket, no foul) wins; any other 8 pot loses. A foul that does not pot the 8 only gives ball in hand.
- **Shot cap:** 150 shots → draw (scored 0.5).

## 5. Fairness contract

For SA vs MA: the same menu (`candidates_hash` logged per shot), the same model, temperature and seeds, the same strategy knowledge (SA's system prompt contains every specialist's block; tested), the same opponent B, the same game seeds and noise draws, the same fallback rule. B has more information than SA and MA (its private simulation); this is intended and stated: B is the common yardstick.

## 6. LLM decision flow

```
DecisionContext ──► promptlib.decision_context_text()   position + last 3 shots + raw candidate list
                     │
     SA:  system_single (rules + offense + position + defense + role) ──► SingleDecision
     MA:  ┌ system_specialist(offense)  ─► SpecialistShortlist ┐
          ├ system_specialist(position) ─► SpecialistShortlist ┼─► merge ─► same #1 for all? ─► play it
          └ system_specialist(defense)  ─► SpecialistShortlist ┘          └─► coordinator (sees only the
                                                                               shortlisted shots + reasons)
                     │
   every call: cache lookup → Ollama /api/chat with the JSON schema as `format`
               → id check (one corrective retry) → shared fallback if still invalid
                     │
   Decision.trace: prompt version, every attempt (parsed output, tokens, latency, cache hit),
                   shortlists, union, unanimous flag, followed  → logged per shot, shown in the demo
```

## 7. Game log record

One JSON line per shot (`game/runner.py`): `exp_id, game_id, seed, shot_index, seat, system, kind (break|decision), state_hash, state, ball_in_hand, menu, noise, candidates_hash, candidates[], decision{shot_id,intent,rationale,confidence}, fallback, trace, usage, planned, executed, outcome{potted, legal, foul, turn_over, game_over, winner, called_ball_made, group_assigned, note, first_ball_hit, truncated}, decision_latency_s, step_s`. Per-game summaries in `games.jsonl` add `seat_metrics` and `ending` (legal 8, 8 with foul, 8 early or wrong pocket).

## 8. Reproducibility and testing

A game is fully determined by (seed, policies, `MenuConfig`, noise level): the rack (D-008, D-013), break jitter, menu shuffle, B's simulation streams and execution noise all derive from the seed through `rng_for`. LLM answers are seeded (Ollama `seed = 1000 + repeat_idx`) and cached for exact replay. `pytest -q` runs unit, integration and demo smoke tests offline with `FakeLLM`.
