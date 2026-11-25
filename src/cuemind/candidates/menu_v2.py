"""v2 shot menu (D-029): forecast every variant, keep the top 15, show geometry only.

1. Generate every viable direct pot for the shooter's legal balls x 3 speeds (soft, medium,
   firm) x 3 spins (draw, stun, follow) = 9 variants per pot, plus legal safety shots.
2. Forecast every variant with `n_forecast` (default 3) noisy simulations at the forecast
   noise level (default low) and average them (`candidates.annotate.rollouts`).
3. Score each variant with the reference player's objective (`policies.search_bot`).
4. Keep at most 15: first the best-scoring pot variant of every legal ball that has a pot (so
   every ball of the shooter's group is represented), then the best remaining pot variants up
   to `max_pots`, then up to `max_safeties` safeties (best first). If nothing exists, the
   contact-only fallback shots are used.
5. Shuffle with a per-decision seed so that the order carries no information, then number
   S01..Snn.

The forecasts stay attached to the candidates (the reference player Stat uses them), but the
LLM-facing description is the geometry-only line (ball, pocket, speed, spin, cut, ease word).
The menu *selection* uses simulation; this is stated in the report. The agents never see any
forecast number.
"""

from __future__ import annotations

from cuemind.candidates.annotate import rollouts
from cuemind.candidates.describe import describe_all
from cuemind.candidates.generate import assign_ids, contact_fallback, generate
from cuemind.candidates.model import Candidate
from cuemind.policies.search_bot import SearchWeights, expected_value
from cuemind.sim.noise import rng_for
from cuemind.sim.state import TableState

MAX_TOTAL = 15


def build_menu_v2(state: TableState, *, n_forecast: int = 3, noise: str = "low",
                  max_pots: int = 12, max_safeties: int = 3,
                  key: tuple = ()) -> list[Candidate]:
    cands = generate(state)
    if not cands or all(c.shot_type == "contact_only" for c in cands):
        menu = cands or contact_fallback(state)
        menu = rollouts(state, menu, n_forecast, noise, key)
    else:
        cands = rollouts(state, cands, n_forecast, noise, key)
        w = SearchWeights()
        score = {c.key: expected_value(c, w) for c in cands}
        order = lambda c: (-score[c.key], c.key)  # noqa: E731 - deterministic tie-break
        pots = sorted((c for c in cands if c.shot_type == "pot"), key=order)
        safeties = sorted((c for c in cands if c.shot_type == "safety"), key=order)

        chosen: list[Candidate] = []
        seen_balls: set[str] = set()
        for c in pots:  # best variant of every legal ball first (coverage)
            if c.target_ball not in seen_balls:
                chosen.append(c)
                seen_balls.add(c.target_ball)
        chosen = chosen[:max_pots]
        taken = {c.key for c in chosen}
        for c in pots:  # then the best remaining variants
            if len(chosen) >= max_pots:
                break
            if c.key not in taken:
                chosen.append(c)
                taken.add(c.key)
        n_saf = min(max_safeties, MAX_TOTAL - len(chosen))
        menu = chosen + safeties[:n_saf]
        if len(menu) < 2:  # always offer a real choice
            menu += [c for c in contact_fallback(state) if c.key not in {m.key for m in menu}][: 2 - len(menu)]
            menu = [m if m.rollouts else rollouts(state, [m], n_forecast, noise, key)[0] for m in menu]

    rng = rng_for(*key, "menu")
    menu = [menu[i] for i in rng.permutation(len(menu))]
    return describe_all(state, assign_ids(menu), "L0")
