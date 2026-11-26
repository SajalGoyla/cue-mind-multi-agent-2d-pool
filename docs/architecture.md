# Architecture (v2)

How CueMind v2 is put together and how one shot is decided. Design rationale: [`decisions.md`](decisions.md) (v2: D-029); plan: [`PLAN_V2.md`](PLAN_V2.md). The v1 architecture (positions benchmark, oracle, forecast menus for LLMs) is preserved under the tag `v1-positions-benchmark`.

## 1. One shot, end to end

```
TableState (positions, groups, shooter, ball-in-hand)
   │
   ├─ ball in hand? ─► ball_in_hand.place_cue_ball()        shared, non-LLM
   │
   ▼
candidates.pipeline.prepare() → menu_v2.build_menu_v2()      identical for every player
   ├─ generate: every direct pot × {soft, medium, firm} × {draw, stun, follow} + legal safeties
   ├─ forecast: 3 noisy simulations per variant (low noise), averaged → attached to candidates
   ├─ select:   best variant per legal ball, then best remaining (≤ 12 pots) + ≤ 3 safeties
   ├─ shuffle:  rng_for(seed, shot_index, "menu") → ids S01..S15
   └─ describe: geometry-only line (ball, pocket, speed, spin, cut, ease word)
   │
   ▼
Policy.decide(DecisionContext) → Decision(shot_id, rationale, trace, usage)
   Stat: argmax of the D-011 objective over the attached forecasts
   SA:   one LLM call on the geometry-only menu
   MA:   offense · position · safety shortlists (k = 3) ─► unanimous #1? play it : coordinator
   (B1: easiest pot, pilot only)            invalid answer → one retry → geometry-only fallback
   │
   ▼
NoiseModel.apply(stroke, rng_for(seed, shot_index, "execute"))   playing noise: none | low
   │
   ▼
sim.execute.simulate() ─► extract_events() ─► rules.judge() ─► rules.advance()
   │
   └── one JSON record per shot (game log: state, menu with forecasts, decision, trace, outcome)
```

The LLM systems never see forecast numbers; only the menu's *selection* uses simulation.

## 2. Modules

| Module | Responsibility |
|---|---|
| `sim/state.py`, `sim/execute.py`, `sim/noise.py`, `sim/trajectories.py` | Position model, stroke execution (retry on pooltool failures), noise with common random numbers, trajectories for the UI |
| `game/rules.py`, `break_shot.py`, `ball_in_hand.py`, `runner.py` | Simplified called-shot 8-ball rules, seeded break, shared ball-in-hand placement, game loop with JSONL logs |
| `candidates/direct.py`, `generate.py` | Ghost-ball pot geometry; pots × 9 speed/spin variants, safeties, contact fallback |
| `candidates/annotate.py` | Noisy forecasts (`rollouts`), used only to build the menu and by Stat |
| `candidates/menu_v2.py`, `pipeline.py` | The v2 menu rule; `PrepConfig(n_forecast=3, forecast_noise="low")` |
| `candidates/describe.py` | Geometry-only candidate lines and the position summary |
| `policies/search_bot.py` | Stat (reference player) |
| `policies/simple_bots.py` | B1 (pilot) |
| `policies/single_agent.py`, `multi_agent.py` | SA and MA v2 |
| `agents/graph_multi.py`, `agents/common.py` | LangGraph team graph; id validation with one corrective retry |
| `llm/client.py`, `cache.py`, `schemas.py`, `promptlib.py`, `prompts/*.md` | Ollama client (retries), response cache, output schemas, prompt text and assembly |
| `eval/tournament.py`, `metrics.py`, `stats.py` | Paired resumable tournaments, per-seat metrics, Wilson / bootstrap / sign test |
| `viz/` | Plotly and matplotlib table drawings |

## 3. MA v2 team graph

```
START ─► offense ──┐                    (each: rules + own strategy block + menu → top-3 ids + reasons)
     ├─► position ─┼─► join ─► all three #1 equal? ─yes─► play it (no coordinator call)
     └─► safety ───┘                └─no─► coordinator (rules + all blocks + menu + nominations,
                                              each nominated id shown with its shot line)
                                              → shot from the union of nominations
```

Fairness: SA's prompt contains every strategy block (tested); same model (Qwen 2.5 3B Instruct), temperature 0.7, seeds, and fallback rule.

## 4. Coordinate frame and rules

pooltool frame in metres; prompts use diamonds (0–4 across, 0–8 along) and named zones. Rules summary: legal break or foul; open table until a called pot assigns groups; fouls (scratch, no contact, wrong first contact, no cushion after contact) give ball in hand; the turn continues only when the called ball drops in the called pocket; pocketing the 8 legally wins, pocketing it early, in the wrong pocket or with a foul loses; 150-shot cap = draw.

## 5. Reproducibility and robustness

A game is determined by (seed, policies, `PrepConfig`, playing noise), except for LLM sampling (temperature 0.7, seeded per sample; responses cached, so replays are exact). Long runs use `scripts/run_queue.py`: detached-friendly, resumable, keeps Windows awake, and restarts Ollama when its runner's memory grows (D-027, D-028).

## 6. Tests

`pytest -q`: state, simulation, noise, rules, candidates and ball in hand, robustness regressions, the v2 menu rule, LLM agents with fake clients (prompt fairness, no forecast numbers in LLM text, retries and fallbacks, unanimity shortcut, coordinator restricted to nominations), Ollama client (mocked), stats, game runner, tournaments, demo smoke test.
