# Report draft: Method and Experimental Setup

*Draft text for sections 4–5 of the report, written from the implemented system (see `docs/architecture.md`, `docs/decisions.md`). Numbers marked [R] must be filled in from `results/summary.md`.*

## 4. Method

### 4.1 Environment

We use the pooltool simulator [Kiefl 2024] for event-based billiards physics on a 7-foot table (0.99 m × 1.98 m, ball radius 28.6 mm). pooltool provides deterministic simulation of sliding, rolling, spinning, ball–ball and ball–cushion collisions and pocketing, and returns a typed event log for every shot. We implement our own simplified called-shot eight-ball rules on top of these events instead of pooltool's built-in ruleset, because our benchmark starts games from arbitrary mid-game positions with groups already assigned, and because the built-in ruleset cannot be copied to evaluate hypothetical shots (D-002). The rules follow WPA eight-ball with documented simplifications: the table stays open after the break, an 8 pocketed on the break is re-spotted, the turn continues only when the called ball drops in the called pocket, a declared safety always ends the turn, and every foul (scratch, no contact, wrong first contact, no cushion after contact) gives the opponent ball in hand anywhere on the table. The break and the ball-in-hand placement are shared, deterministic, non-LLM procedures, so that the comparison isolates shot selection (D-005).

**Execution noise.** A plan is not executed perfectly. Each executed stroke receives Gaussian errors in aim (σ = 0.5°), cue speed (σ = 5 % relative) and tip offset (σ = 0.02 ball radii) at our main "medium" setting (low and high settings halve and double these). Noise is drawn from random streams keyed by (game seed, shot index), so two systems that choose the same shot at the same point of the same seeded game receive the same execution error (common random numbers, D-004). Without noise, choosing the best simulated shot is an oracle and games are decided by who breaks; with noise, risk and safety play matter.

### 4.2 Candidate generation and simulation access

LLMs are poor at producing the continuous parameters of a pool stroke [PoolBench]. Following the common agent pattern in which tools compute and the model judges, a shared generator proposes candidate shots and every decision maker chooses one by its identifier (D-003):

- **Pots:** every direct pot whose ghost-ball geometry is feasible (pooltool's `ai.pot` helpers: clear cue and pocket paths, room for the ghost ball, cut ≤ 80°), crossed with three speeds (1.5, 2.5, 4.0 m/s) and three spins (draw, stun, follow).
- **Safeties:** contact shots at three cut angles and two speeds on every legal ball, kept only if a noise-free simulation is legal.
- **Fallback:** if neither exists, aim at each legal ball's centre.

Each candidate is annotated at one of three **simulation-access levels**, the experimental variable of RQ3 (D-010): **L0** geometry only (cut angle, distances, a geometric difficulty score); **L1** one noise-free preview simulation (pots or not, foul, cue-ball end position, number of follow-up pots); **L2** eight simulations with execution noise, summarised as probabilities of potting, scratching, fouling and keeping the turn, the mean cue-ball end position, the ease of the shooter's best next shot when the turn continues, and the ease of the opponent's best shot when it passes. The menu is pruned to at most 12 pots (best variant per target, pocket and spin, ranked by pot probability) and 3 safeties (lowest opponent threat). Pruning never uses the objective of the search baseline, so the menu does not pre-select its answer.

For the language models each candidate is rendered as one line of plain text, and positions are given in *diamonds* (the rail sights; 0–4 across, 0–8 along the table) with named zones, because models reason poorly about raw coordinates (D-012).

### 4.3 Decision makers

- **B0 random** and **B1 geometric** (easiest pot by geometry, medium speed, centre-ball) are reference baselines.
- **B2 search** chooses the candidate maximising a transparent expected value over the same annotated menu: EV = p_pot·(1 + 0.5·next_ease) − 1.5·p_scratch − 1.0·p_other_foul − 0.6·(1 − p_keep_turn)·opp_threat (D-011).
- **SA (single agent)** makes one structured-output call per decision. Its system prompt contains the game rules, an explanation of every candidate field, and three strategy blocks (offense, position, safety).
- **MA (multi-agent)** is a LangGraph graph (D-017): three specialists (offense, position, safety) run concurrently on the same position text, each with the rules and only its own strategy block; each scores every candidate 0–10, names a top choice and writes a short summary. A coordinator, whose prompt holds all three strategy blocks, reads the reports and chooses the shot, stating which specialists it followed or overruled. An optional critic may veto a clear blunder once.
- **SA-vote** draws four independent single-agent samples and takes a majority vote, matching MA's four calls per decision; it controls for the possibility that a multi-agent advantage comes only from extra inference compute [Smit et al. 2024].

All LLM outputs are constrained by JSON schemas (structured outputs). If a returned identifier is not in the menu, the model is asked once more with the list of valid identifiers; if it still fails, a shared fallback rule picks the highest pot-probability shot, and the fallback is logged and reported (D-009, D-016).

### 4.4 Fairness contract

For every comparison between LLM systems we hold fixed: the candidate list (verified by hashing), the information level, the rollout and execution random streams, the model and its reasoning-effort setting, the strategy knowledge in the prompts (the single agent's prompt contains every specialist's strategy block verbatim; tested), the non-shot policies (break, ball in hand) and the fallback rule. Prompts are tuned only on a development split and frozen before test runs.

## 5. Benchmark and experimental setup

### 5.1 Positions benchmark

No dataset exists for this task, so we built one (D-019, `data/README.md`). Positions are sampled from games between B2 and B1 at medium noise (two per game) plus 20 % random valid layouts, de-duplicated, and split into a development set (60) and a test set (180, stratified into easy / medium / hard by the best pot probability) [R: actual counts from `data/positions/stats.json`]. Each position stores one frozen, fully annotated menu; for L0 and L1 runs the same menu is re-described with information hidden, so access levels differ only in information.

**Ground truth.** For every candidate we estimate an *expected inning value* (EIV) from 24 noisy executions: +1 per own ball legally pocketed, the inning continued greedily by the B1 rule for up to two further shots, −1 per foul, −0.75 × the ease of the opponent's best shot whenever the turn passes, and ±5 for winning or losing on the 8. The continuation uses B1 rather than B2 so that the oracle does not agree with the search baseline by construction. A decision's **regret** is the best EIV in the menu minus the chosen shot's EIV; we also report top-1 accuracy (within one standard error of the best) and a normalised score (0 = uniformly random choice, 1 = best). Each position is tagged with situations: *offense*, *position* (several good pots whose leaves differ) and *defense* (no good pot, or the opponent on the 8). Nine hand-made hard cases (safety-only, shape traps, endgame, cluster, scratch risk, ball in hand) complement the sampled positions; their labels are drafted from the oracle and reviewed by a team member (D-022).

### 5.2 Games

Tournaments use paired seeds: for each seed, each system breaks once, and both games share the rack, break, rollout and noise streams (D-018). Win rates are reported with Wilson 95 % intervals; paired seed scores with bootstrap intervals and a two-sided sign test. Decision-level comparisons use the Wilcoxon signed-rank test on per-position regret with Holm correction across the primary comparisons (MA vs SA, MA vs SA-vote, MA vs B2) and rank-biserial effect sizes.

### 5.3 Calibration finding

In bot tournaments (E0, E0b), B2 beats random play clearly (79 % [70, 86] at medium noise) but is not distinguishable from B1 at any noise level (52–54 %, 60–100 games each), although it pots more often. Game outcomes are therefore dominated by execution luck and table layout, which limits the power of game-level comparisons; the decision-level benchmark carries the main comparison, and game results are reported as secondary evidence (D-024).

### 5.4 Models and cost

LLM systems use Claude Haiku 5.5 through Google Cloud Vertex AI for the main experiments and Claude Sonnet 5.5 to confirm key results [R], with adaptive thinking at reasoning effort "medium" and the static system prompt cached. Temperature cannot be set on these models, so live outputs are not bit-reproducible; all responses are cached for exact replay, and decision variance is measured directly (E10). Token counts, estimated cost and latency are logged for every call [R: per-decision cost and latency].
