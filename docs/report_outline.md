# Final report outline (due Dec 15, ACM Master Article template)

Write the report from these sources: `docs/architecture.md` (method), `docs/decisions.md` (design choices and their reasons; D-029 for v2), `docs/progress_log.md` (what happened), `results/summary.md` (every number), `results/figures/` (every figure), `docs/ta_feedback.md` (response to feedback).

1. **Abstract.**
2. **Introduction.** Problem, why sequential decisions with consequences matter for agent systems, research question (does a team of specialised LLM agents play better than one LLM, against the same reference player?), contributions, headline finding with uncertainty.
3. **Related work** (compare critically, say how each informed our design): computational pool (Archibald et al.), pooltool; LLMs in pool (PoolBench: numeric outputs fail → the LLM chooses among candidates by id; CueTip: a neural policy chooses and the LLM explains → here the LLM chooses); multi-agent LLM collaboration and its critiques (Smit et al.: strong single-agent baselines → matched knowledge and a common opponent); tool-augmented agents.
4. **Method.** Environment and rules (D-002); raw candidate menu (D-029); B (simulation-based reference, D-011); SA; MA v2 with shortlists (D-029); prompts (appendix); fairness contract.
5. **Setup.** Game protocol (paired seeds, both seat orders, common random numbers, D-004, D-018); noise pilot and the chosen level; prompt development on dev seeds only, then frozen; model, temperature, hardware.
6. **Results.** Win rate against B for SA and MA with CIs; SA vs MA seed by seed; how games ended; pot success, balls per inning, fouls; calls, tokens and latency.
7. **Analysis.** Annotated moments from real game logs (where the team helped, where it hurt); when a single agent is preferable.
8. **Why v1 was dropped (one paragraph).** The positions benchmark showed the LLMs simulated statistics and graded with a simulated oracle built from the same simulator and similar terms; evidence in D-029.
9. **Limitations.** Simulated play only; one small open model; menus limited to direct pots and simple safeties (no kicks/banks); limited number of games; no user study.
10. **Future work.** Fine-tuning (imitating B's choices) for both SA and MA; stronger models; richer menus. **Conclusion.** References. Appendix: prompts (`src/cuemind/llm/prompts/`), schemas, extra tables, response to TA feedback.
