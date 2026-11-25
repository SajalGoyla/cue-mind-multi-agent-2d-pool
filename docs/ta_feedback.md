# TA feedback and how we addressed it

The grading rubric checks whether important feedback from the October TA discussion was addressed. **A team member must paste the feedback here** (it is not recorded anywhere in the repository yet), then fill in the right-hand column as the work progresses. The report's "Response to TA feedback" paragraph and the slides should be written from this table.

| # | Feedback (verbatim or close paraphrase) | How we addressed it | Where (file / section / figure) |
|---|---|---|---|
| 1 | _to be filled in by the team_ | | |
| 2 | | | |
| 3 | | | |

Changes relative to the October 26 proposal that the report should mention:

- Rules: our own simplified called-shot 8-ball layer instead of pooltool's ruleset (D-002).
- Model: a local open-weight model (Qwen 2.5 7B Instruct via Ollama) for all LLM experiments, because Claude on Vertex AI never received quota (D-025). Fine-tuning becomes possible with a local model.
- Evaluation (v2, D-029): full games against a common simulation-based reference player (B vs B, B vs SA, B vs MA) instead of a decision-level benchmark with a simulated oracle, which turned out to be circular. The LLM systems see raw geometric candidates only.
- Banks and kicks are not in the candidate generator (D-006).
