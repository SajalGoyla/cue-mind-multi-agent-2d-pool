# Presentation and demo outline (Dec 7)

Mapped to the rubric (problem 10 %, contributions 10 %, evaluation 10 %; demo 30 % graded separately). Every number on a slide must come from `results/summary.md` or a file it names; fill the bracketed slots only from there.

## Slides (~12–14)

**Problem statement**
1. *Title:* CueMind: do specialised LLM agents choose better pool shots than one LLM? Team, course.
2. *Setting:* 8-ball in 60 seconds (groups, fouls, ball in hand, the 8). Why it is a good testbed: sequential decisions with long-term consequences and an unambiguous outcome.
3. *State of the art and its limits:* PoolBench (LLMs that output raw angle/power pocketed nothing in 300 shots); CueTip (a neural policy chooses, the LLM explains; LLM-chosen play wins 26–32 %); Smit et al. (multi-agent debate often fails to beat a strong single agent).
4. *Our question, hypotheses, scope:* RQ1–RQ3; H1–H4 (plan §2.2); simulated play only, one ruleset, one table, no user study.

**Major contributions**
5. *Architecture diagram* (from `docs/architecture.md` §1 and §8): code generates and simulates shots; agents choose by id; why this split.
6. *Fairness contract:* same menu, same information, same strategy knowledge (tested), same noise draws (common random numbers), compute-matched SA-vote, same fallback rule.
7. *Benchmark:* positions with frozen menus and oracle values, difficulty and situation tags, hand-made hard cases; paired-seed tournaments.
8. *Engineering:* LangGraph fan-out, structured outputs, response cache and budget guard, failure handling; measured latency and cost per decision: [from summary].

**Evaluation**
9. *Baselines (E0):* bot tournaments; what they say about luck vs skill at each noise level: [E0, E0b].
10. *Main result (E1):* regret with CIs for B0/B1/B2/SA/SA-vote/MA; paired tests: [E1].
11. *Where collaboration helps (RQ2):* regret by situation; cost–quality plot: [E1 figures].
12. *Information and noise (RQ3):* L0/L1/L2 ablation [E4]; noise [E0b/E5].
13. *Case studies:* one success, one failure, with agent traces (from the demo replay); error categories.
14. *Limitations and future work* (plan §17), prioritised by what we observed (e.g. no kicks/banks; luck-dominated games at medium noise).

## Demo script (4–5 minutes, rehearse twice)

1. (20 s) Who it is for and what it does: an AI shot selector where physics is done by code and judgement by LLMs.
2. (60 s) *Decide a shot* on an easy position: candidate menu on the table, single agent vs team cards, execute with noise.
3. (90 s) A position where the specialists disagree (pick from real E1 logs): point at the specialist that changed the decision; open the oracle panel.
4. (60 s) *Replay a game*: the last shots of a real logged game with traces.
5. (30 s) *Results* tab: one headline chart; latency and cost from measurements.

**Pre-flight (1 hour before):** `python scripts/check_llm.py --quick`; `pytest -q tests/smoke`; response cache present for every demo position (run each once the day before); laptop on power; app already launched; backup recording on the desktop, labelled as recorded.

## Sample runs to prepare (from real logs only)

| Slot | Shows | Source |
|---|---|---|
| S1 | Agreement on an easy pot | E1 position where SA and MA pick the same top shot |
| S2 | Position play changes the choice | E1 position tagged `position` where MA and SA differ |
| S3 | Safety | E1 or hard case tagged `defense` |
| S4 | Failure case | High-regret MA decision (coordinator followed the wrong specialist) |
| S5 | Endgame on the 8 | Last shots of a real E2 game |
