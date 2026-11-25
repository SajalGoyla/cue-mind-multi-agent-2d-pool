# Presentation and demo outline (Dec 7)

Mapped to the rubric (problem 10 %, contributions 10 %, evaluation 10 %; demo 30 % graded separately). Every number on a slide must come from `results/summary.md`.

## Slides (~12)

**Problem statement**
1. *Title:* CueMind: does a team of LLM agents play better pool than one LLM? Team, course.
2. *Setting:* 8-ball in 60 seconds. Why it is a good testbed: sequential decisions with long-term consequences and an unambiguous outcome.
3. *State of the art and its limits:* PoolBench (LLMs that output raw angle/power pocketed nothing in 300 shots); CueTip (a neural policy chooses, the LLM explains); Smit et al. (multi-agent debate often fails to beat a strong single agent).
4. *Question and scope:* SA vs MA against the same reference player; simulated play only, one ruleset, one table, no user study.

**Major contributions**
5. *Architecture:* code lists raw candidates from geometry; B simulates privately; SA and MA judge from geometry; MA = 3 specialist shortlists + coordinator (LangGraph).
6. *Fair comparison:* same menu, same knowledge (tested), same seeds and noise draws, same opponent; what B knows that the LLMs do not.
7. *What we changed and why:* the v1 benchmark was circular (simulated statistics in, simulated oracle out); v2 measures real games. A strong "critical analysis" slide.

**Evaluation**
8. *Pilot:* at zero noise B beats an easiest-pot bot 90 % of the time, so decisions matter in games; at low noise they barely do (62 %, not significant).
9. *Main result:* win rate against B for SA and MA with CIs; SA vs MA seed by seed.
10. *How the systems play:* pot success, balls per inning, fouls, how games ended; cost and latency per decision.
11. *Case studies:* one game moment where the team helped, one where it hurt (from real logs).
12. *Limitations and future work:* fine-tuning both systems on B's choices, stronger models, richer menus.

## Demo script (4–5 minutes, rehearse twice)

1. (20 s) What it does: code proposes shots; B simulates; LLM agents judge from geometry alone.
2. (90 s) *Decide a shot*: a demo position; B, SA and MA side by side; the three specialist shortlists and the coordinator's pick; execute the shot.
3. (90 s) *Replay a game*: the last shots of a real B vs MA game with the shortlists.
4. (30 s) *Results*: the win-rate chart; latency per decision from measurements.

**Pre-flight (1 hour before):** `python scripts/check_llm.py --quick`; `pytest -q tests/smoke`; run every demo position once in Ollama mode so answers are cached; copy two real game logs into `demo/games/`; laptop on power; app already launched; backup recording ready, labelled as recorded.
