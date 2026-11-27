# Model-size check: does the team help with a larger model? (written before any 14B game)

**Why.** The v2/v3 finding "a team of LLM agents does not beat a single agent" was measured with Qwen 2.5 3B. The obvious objection is that a 3B model is too weak to use specialist roles. This run tests that objection directly.

**What changes.** Only the model: Qwen 2.5 **14B** Instruct (Ollama, temperature 0.7, seeded) instead of 3B. Everything else is identical to `v2_main` at zero playing noise: code and frozen prompts (`prompts-v2`), menu, the reference player Stat, seeds 9000–9029, both seat orders, common random numbers. No prompt tuning for 14B. Config: `experiments/configs/v2_scale14b.yaml` (Stat vs SA and Stat vs MA, 60 games each; the Stat vs Stat control is reused from `v2_main`).

**Where.** One Google Cloud VM (`g2-standard-8`, 1 × NVIDIA L4), project `cuemind-516017`, deleted after the run. Spending cap $25.

**Prediction (stated before the run).**
- **Team effect at 14B:** MA − SA, paired per seed (wins in the two games against Stat), stays within ±0.16 wins per seed, the same bound as the pooled 3B result (+0.04 [−0.08, +0.16]). If MA − SA at 14B is positive with a 95 % interval above zero, the conclusion becomes "teaming starts to pay off with a larger model".
- **Model effect:** both SA and MA win more often against Stat at 14B than at 3B (3B, zero noise: SA 22 %, MA 25 %).

**Reported metrics (no others).** Win rate against Stat with 95 % Wilson CI for SA and MA at 3B and 14B; MA − SA paired per seed with bootstrap CI and sign test; LLM calls and seconds per decision.
