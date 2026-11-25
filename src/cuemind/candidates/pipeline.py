"""The raw decision menu (v2, D-029): geometry only, no simulated forecasts.

Every policy in a game (B, SA, MA) receives the output of the same `build_menu` call.

Menu rule:
    pots      the `max_pot_targets` geometrically easiest target/pocket pairs, each with every
              speed in `speeds` and every spin in `spins` (default: 4 pairs x medium x
              draw/stun/follow = up to 12 pots)
    safeties  up to `max_safeties`, taken round-robin over the legal target balls. A safety is
              listed only if a noise-free simulation of it is legal; this is a validity filter,
              its result is never shown to anyone.
    fallback  contact-only shots when nothing else exists.

The order is shuffled with a seeded generator so the position of a shot in the list carries
no information (in v1, S01 was always the highest simulated pot probability).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from cuemind.candidates.describe import describe_all
from cuemind.candidates.generate import assign_ids, generate
from cuemind.candidates.model import Candidate
from cuemind.sim.noise import rng_for
from cuemind.sim.state import TableState


@dataclass(frozen=True)
class MenuConfig:
    max_pot_targets: int = 4
    speeds: tuple[str, ...] = ("medium",)
    spins: tuple[str, ...] = ("draw", "stun", "follow")
    max_safeties: int = 3
    shuffle: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "MenuConfig":
        d = dict(d or {})
        for k in ("speeds", "spins"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)


def _pick_pots(cands: list[Candidate], cfg: MenuConfig) -> list[Candidate]:
    pots = [c for c in cands if c.shot_type == "pot"]  # generate() orders them by geometric ease
    targets: list[tuple[str, str]] = []
    for c in pots:
        t = (c.target_ball, c.pocket)
        if t not in targets:
            targets.append(t)
    keep = set(targets[: cfg.max_pot_targets])
    return [c for c in pots if (c.target_ball, c.pocket) in keep
            and c.power_label in cfg.speeds and c.spin_label in cfg.spins]


def _pick_safeties(cands: list[Candidate], k: int) -> list[Candidate]:
    by_ball: dict[str, list[Candidate]] = {}
    for c in cands:
        if c.shot_type == "safety":
            by_ball.setdefault(c.target_ball, []).append(c)
    out: list[Candidate] = []
    rank = 0
    while len(out) < k and any(rank < len(v) for v in by_ball.values()):
        for ball in by_ball:
            if rank < len(by_ball[ball]) and len(out) < k:
                out.append(by_ball[ball][rank])
        rank += 1
    return out


def build_menu(state: TableState, cfg: MenuConfig | None = None,
               menu_key: tuple = ()) -> list[Candidate]:
    """Raw menu for the shooter in `state` (cue ball already placed), ids S01.. in shuffled order."""
    cfg = cfg or MenuConfig()
    cands = generate(state)
    menu = _pick_pots(cands, cfg) + _pick_safeties(cands, cfg.max_safeties)
    if not menu:
        menu = [c for c in cands if c.shot_type == "contact_only"]
    if cfg.shuffle and len(menu) > 1:
        order = rng_for(*menu_key, "menu").permutation(len(menu))
        menu = [menu[i] for i in order]
    return describe_all(assign_ids(menu))
