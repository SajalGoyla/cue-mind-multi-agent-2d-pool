# Final report outline (due Dec 15, ACM Master Article template)

Write the report from these sources: `docs/architecture.md` (method), `docs/decisions.md` (design choices and their reasons), `docs/progress_log.md` (what happened, issues found), `data/README.md` (benchmark), `results/summary.md` (every number), `results/figures/` (every figure), `docs/ta_feedback.md` (response to feedback).

1. **Abstract.**
2. **Introduction.** Problem, why sequential decisions with consequences matter for agent systems, RQ1–RQ3, contributions (bulleted), headline finding with uncertainty.
3. **Related work** (compare critically, say how each informed our design): computational pool (Archibald et al.), pooltool; LLMs in pool (PoolBench: numeric outputs fail → we choose by id; CueTip: LLM explains a neural policy → we let the LLM choose among simulated options); multi-agent LLM collaboration and its critiques (Smit et al.: strong single-agent baselines → our fairness contract and SA-vote); tool-augmented agents.
4. **Method.** Environment and rules (D-002); candidate generation (D-006) and access levels (D-010); systems B0–B2 (D-011), SA, SA-vote, MA, critic (D-016, D-017); prompts (appendix); fairness contract.
5. **Benchmark and setup.** Positions (D-019, `data/README.md`), oracle definition and its limits; hard cases (D-022, human-reviewed labels); tournaments (D-018) with common random numbers (D-004); noise model and calibration (E0, E0b); models, versions, effort, prices, hardware; prompt-iteration procedure on the dev split only.
6. **Results.** E1 (RQ1); situation breakdown, cost, SA-vote, E6 ablations (RQ2); E4 and noise (RQ3); E2 games; E10 variance; E8 model confirmation.
7. **Analysis.** Failure categories with examples (`docs/failure_analysis.md`); why results occurred; when a single agent is preferable.
8. **Limitations.** Simulated noise is not human play; luck-dominated games at medium noise (E0); one model family; oracle is a short-horizon approximation; no kicks/banks; no user study (scenario-based evaluation only).
9. **Future work** (plan §17, prioritised by findings). 10. **Conclusion.** References. Appendix: prompts (`src/cuemind/llm/prompts/`), schemas, extra tables, response to TA feedback.
