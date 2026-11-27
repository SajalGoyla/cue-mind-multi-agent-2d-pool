"""v3 condition A menu (D-031): selected by geometry and physics hints only, no forecasts.

1. Every viable direct pot for the shooter's legal balls x 3 speeds x 3 spins (9 variants)
   plus legal safeties (the noise-free legality check stays a validity filter only).
2. A simulation-free physics hint for every pot variant (`candidates.hints`).
3. Keep at most 15: the best variant of every legal ball with a pot (by the rule score:
   ease x (1 + 0.5 x hinted next-shot ease) - scratch warning), then the best remaining
   variants up to `max_pots`, then up to `max_safeties` safeties (one per target ball first,
   medium speed preferred because it reaches a cushion more reliably).
4. Seeded shuffle; numbered S01..Snn. No simulation result is used anywhere.
"""

from __future__ import annotations

from dataclasses import asdict

from cuemind.candidates.generate import assign_ids, contact_fallback, generate
from cuemind.candidates.hints import hints_for, rule_score
from cuemind.candidates.model import Candidate
from cuemind.sim.noise import rng_for
from cuemind.sim.state import TableState

MAX_TOTAL = 15


def build_menu_geo(state: TableState, *, max_pots: int = 12, max_safeties: int = 3,
                   key: tuple = ()) -> list[Candidate]:
    cands = generate(state)
    hints = hints_for(state, cands)
    cands = [c.model_copy(update={"hint": asdict(hints[c.key])}) if c.key in hints else c
             for c in cands]
    score = {c.key: rule_score(c, hints.get(c.key)) for c in cands}
    pots = sorted((c for c in cands if c.shot_type == "pot" and c.key in hints),
                  key=lambda c: (-score[c.key], c.key))
    chosen: list[Candidate] = []
    seen: set[str] = set()
    for c in pots:
        if c.target_ball not in seen:
            chosen.append(c)
            seen.add(c.target_ball)
    chosen = chosen[:max_pots]
    taken = {c.key for c in chosen}
    for c in pots:
        if len(chosen) >= max_pots:
            break
        if c.key not in taken:
            chosen.append(c)
            taken.add(c.key)
    safeties = [c for c in cands if c.shot_type == "safety"]
    safeties.sort(key=lambda c: (c.power_label != "medium", c.geometry.cut_deg, c.key))
    picked, balls = [], set()
    for c in safeties:  # one per target ball first
        if c.target_ball not in balls:
            picked.append(c)
            balls.add(c.target_ball)
    picked += [c for c in safeties if c not in picked]
    menu = chosen + picked[: min(max_safeties, MAX_TOTAL - len(chosen))]
    if len(menu) < 2:
        extra = [c for c in (cands + contact_fallback(state)) if c.key not in {m.key for m in menu}]
        menu += extra[: 2 - len(menu)]
    rng = rng_for(*key, "menu")
    menu = [menu[i] for i in rng.permutation(len(menu))]
    return assign_ids(menu)
