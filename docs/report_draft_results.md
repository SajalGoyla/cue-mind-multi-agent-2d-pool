# Report draft: Results and Analysis

*Draft text for sections 6–7. Every number comes from `results/summary.md` (regenerate with `python experiments/analyze.py`). Slots marked [Q] wait for experiments still in the queue (E4, E10, E2, E5, E6).*

## 6. Results

### 6.1 Baselines and calibration (E0, E0b)

Between non-LLM bots, game outcomes separate random play from informed play but not much more. At our main (medium) noise level the search bot B2 wins 79 % [70, 86] of 100 paired games against random choice (B0), and the geometric bot B1 wins 61 % [51, 70] against B0. B2 against B1 wins only 54 % [44, 63] (sign test p = 0.56), even though B2 pots more often (61 % vs 53 % of attempts). The same holds at low (53 % [41, 65]) and high noise (52 % [39, 64]). With execution noise, eight-ball outcomes are dominated by luck and by who inherits a good layout; a 50-seed tournament can only detect very large differences in decision quality. We therefore use decision-level regret on fixed positions as the primary measure and game win rates as secondary evidence (Section 5.3).

Decision-level regret, in contrast, cleanly orders the bots (E1, 180 test positions, mean regret in oracle units with 95 % bootstrap CIs; lower is better): B0 0.706 [0.597, 0.827], B1 0.343 [0.282, 0.406], B2 0.241 [0.196, 0.288]. B2 picks a shot within one standard error of the oracle's best in 62 % of positions, B1 in 50 %, B0 in 33 %.

### 6.2 Prompt development (dev split only)

On the 60 development positions, the initial prompts gave SA 0.412 and MA 0.423 mean regret against B2's 0.359. Both LLM systems chose safeties about a third of the time (B2: 13 %), and MA's coordinator followed the position specialist in 52 of 60 decisions, often preferring a better cue-ball leave over a clearly higher pot probability. One revision, applied identically to every LLM system's shared rules (pot probability first; the leave breaks ties; a safety only when no pot is reasonably likely), plus two role-specific fixes for MA (a specialist's top choice must be its highest-scored shot; the coordinator must not default to one specialist), reduced regret to SA 0.380 and MA 0.351 and the safety rate to 20 % and 12 %. Prompts were then frozen (`prompts-v1`) before any test-split run.

### 6.3 RQ1: do specialized agents choose better shots?

| System | Mean regret [95% CI] | Top-1 | LLM calls per decision | Latency per decision |
|---|---|---|---|---|
| B2 search | 0.241 [0.196, 0.288] | 0.62 | 0 | — |
| SA | 0.294 [0.242, 0.349] | 0.56 | 1.0 | 8.5 s |
| SA-vote (4 samples) | 0.296 [0.243, 0.352] | 0.55 | 4.0 | 27.0 s |
| MA | 0.307 [0.254, 0.363] | 0.53 | 4.1 | 30.1 s |
| B1 geometric | 0.343 [0.282, 0.406] | 0.50 | 0 | — |
| B0 random | 0.706 [0.597, 0.827] | 0.33 | 0 | — |

*Table: E1, 180 test positions, information level L2, Qwen 2.5 7B Instruct, prompts-v1. Latency measured with four decisions in flight on one RTX 4060 laptop GPU.*

With identical candidates, information, strategy knowledge and model, the multi-agent system does not choose better shots than a single agent: MA's mean regret is 0.013 higher than SA's (paired Wilcoxon p = 0.80, rank-biserial +0.04) and 0.011 higher than the compute-matched SA-vote (p = 0.86). Both LLM systems outperform the geometric bot but trail the search bot, which uses the same simulation statistics through a fixed hand-written objective (MA vs B2: +0.066, p = 0.017, Holm-adjusted 0.051). No LLM decision needed the fallback rule. Game-level results (E2): [Q].

### 6.4 RQ2: where does collaboration help, and is it worth the cost?

Broken down by situation, MA's regret relative to SA is similar in offense positions (0.341 vs 0.345), worse in position-play positions (0.246 vs 0.223) and clearly worse in defensive positions (0.304 vs 0.253); by difficulty MA is ahead only on medium positions (0.349 vs 0.377). The defensive result is the opposite of what the design intended. The traces explain it: the safety specialist recommends a safety in 96 % of positions, so it acts as a constant voice for caution rather than a selective one, and the coordinator occasionally follows it over a near-certain pot. A rule-based categorisation of all decisions with regret above 0.3 (`experiments/categorize_errors.py`) locates MA's extra errors: B2 makes 54 such errors, SA 65, SA-vote 64 and MA 69; the counts of wrong speed/spin variants are equal for the three LLM systems (26 each), while MA makes more errors in choosing the target and pocket (35 vs 28 for SA). In 21 of MA's 69 errors at least one specialist had named the oracle's best shot as its top choice, yet the coordinator followed another specialist: the information was present in the team but lost in aggregation. Removing each specialist (E6): [Q]. Since MA costs about four times the calls and 3.5 times the latency of SA without a quality gain, it is not worth its cost for this model. SA-vote, which spends the same compute on independent samples, also gives no gain, suggesting the limitation is the model's judgment rather than sampling noise. Decision variance across repeated samples (E10): [Q].

### 6.5 RQ3: simulation access and execution noise

The bots already show that *how* simulation is summarised matters. Given one noise-free preview (L1), B2 is worse than with geometry alone (0.387 vs 0.318), because a perfect-stroke preview is over-confident about pots that often miss with realistic strokes; with eight noisy rollouts (L2) it reaches 0.246 (189 positions including hard cases). The LLM systems follow the same pattern and depend on simulation even more (E4, same 180 positions and frozen menus): SA's regret is 0.435 with geometry only, 0.441 with one preview and 0.290 with noisy rollouts; MA's is 0.546, 0.496 and 0.307. Contrary to our hypothesis H4, the multi-agent disadvantage is *largest* with the least information (MA − SA: +0.112 at L0, p = 0.046 unadjusted, Holm 0.14; +0.055 at L1; +0.017 at L2, p = 0.67): with only geometry the specialists have little to anchor their judgements on, and the coordinator compounds their errors rather than averaging them out. A single noise-free preview gives the single agent nothing, mirroring its harmful effect on B2. No LLM decision needed the fallback rule (0 of 1,080). Effect of execution noise on regret (E5, low/high vs E1 medium): [Q].

## 7. Analysis

**Failure modes.** The largest regrets for both LLM systems come from choosing between variants of the same pot (speed and spin) where the oracle values the leave or a small scratch risk much more than the systems do; for example, potting the 8 with medium follow (12 % scratch risk) instead of soft draw (no scratch) costs 1.3 oracle units. Ordering by pot probability puts the highest-percentage pot first, and S01 is the oracle's best shot in only 38 % of positions; B2 picks it in 72 %, SA in 62 % and MA in 56 %. Over-weighting safety (MA's safety specialist) is the second recurring failure. Categorised failure counts: [team, from `docs/failure_analysis_e1.md`].

**Why specialization did not help here.** Each specialist sees the same candidate table and statistics; the decision mostly requires weighing a few numbers correctly, which a single prompt already contains. Splitting the perspectives adds a summarisation step in which nuance is lost and a biased voice (safety) gains weight, while the coordinator, the same 7B model, must still integrate everything. This is consistent with reports that multi-agent debate rarely beats a strong single-agent baseline at matched compute [Smit et al. 2024]. Whether a stronger model changes the picture is open until the Claude confirmation run (E8).

**Limits of the evidence.** One small open model; a short-horizon oracle with its own approximations; menus limited to direct pots and simple safeties (no kicks or banks); simulated noise rather than human play; no user study.
