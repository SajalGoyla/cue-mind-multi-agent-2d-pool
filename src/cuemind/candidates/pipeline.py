"""Decision preparation: generate -> annotate (access level) -> prune -> describe.

Every policy in a comparison receives the output of the *same* `prepare` call, which is
what the fairness contract requires (identical menus and information).

Access levels (the RQ3 variable):
    L0  geometry only (no simulation statistics shown)
    L1  one noise-free preview simulation per candidate
    L2  n noisy rollouts per candidate (main setting)

Pruning (D-007, D-010):
    L0      round-robin over pot geometries by geometric ease, then the first safeties.
    L1/L2   a geometry-ordered working set of `working_pots` pots and `working_safeties`
            safeties is simulated; the final menu keeps pots with p_pot >= min_p_pot, the
            best variant per (target, pocket, spin), the top `max_pots` by p_pot, and the
            `max_safeties` safeties with the lowest opponent threat. Pruning never uses the
            B2 objective, so the menu does not pre-select B2's answer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from cuemind.candidates.annotate import preview, rollouts
from cuemind.candidates.describe import describe_all
from cuemind.candidates.generate import assign_ids, contact_fallback, generate
from cuemind.candidates.model import Candidate
from cuemind.sim.state import TableState

ACCESS_LEVELS = ("L0", "L1", "L2")


@dataclass(frozen=True)
class PrepConfig:
    access_level: str = "L2"
    n_rollouts: int = 8
    noise: str = "medium"        # noise preset used for L2 rollouts (= execution noise)
    max_pots: int = 12
    max_safeties: int = 3
    working_pots: int = 18
    working_safeties: int = 4
    min_p_pot: float = 0.10

    def __post_init__(self):
        if self.access_level not in ACCESS_LEVELS:
            raise ValueError(f"access_level must be one of {ACCESS_LEVELS}")

    def to_dict(self) -> dict:
        return asdict(self)


# Without simulation statistics, variants are taken round-robin over pot geometries (in
# order of geometric ease) so the menu covers many target/pocket choices, not just one.
VARIANT_PRIORITY = [("medium", "stun"), ("soft", "stun"), ("medium", "draw"),
                    ("medium", "follow"), ("firm", "stun"), ("soft", "follow"),
                    ("soft", "draw"), ("firm", "draw"), ("firm", "follow")]


def round_robin_pots(cands: list[Candidate], k: int) -> list[Candidate]:
    pots = [c for c in cands if c.shot_type == "pot"]
    geo_order: dict[tuple, int] = {}
    for c in pots:
        geo_order.setdefault((c.target_ball, c.pocket), len(geo_order))
    vp = {v: i for i, v in enumerate(VARIANT_PRIORITY)}
    chosen = sorted(pots, key=lambda c: (vp[(c.power_label, c.spin_label)],
                                         geo_order[(c.target_ball, c.pocket)]))[:k]
    chosen.sort(key=lambda c: (geo_order[(c.target_ball, c.pocket)], vp[(c.power_label, c.spin_label)]))
    return chosen


def prune_l0(cands: list[Candidate], cfg: PrepConfig) -> list[Candidate]:
    """Geometry-only pruning (used at access level L0)."""
    pots = round_robin_pots(cands, cfg.max_pots)
    safeties = [c for c in cands if c.shot_type == "safety"][: cfg.max_safeties]
    kept = pots + safeties
    return kept if kept else [c for c in cands if c.shot_type == "contact_only"]


def _p_pot(c: Candidate) -> float:
    if c.rollouts is not None:
        return c.rollouts.p_pot
    return 1.0 if (c.preview and c.preview.pots_called) else 0.0


def _threat(c: Candidate) -> float:
    if c.rollouts is not None:
        return c.rollouts.opp_threat + c.rollouts.p_foul
    return 1.0 if (c.preview and not c.preview.legal) else 0.5


def prune_annotated(cands: list[Candidate], cfg: PrepConfig) -> list[Candidate]:
    pots = [c for c in cands if c.shot_type == "pot" and _p_pot(c) >= cfg.min_p_pot]
    best: dict[tuple, Candidate] = {}
    for c in pots:  # keep the best variant per (target, pocket, spin)
        k = (c.target_ball, c.pocket, c.spin_label)
        if k not in best or _p_pot(c) > _p_pot(best[k]):
            best[k] = c
    pots = sorted(best.values(), key=lambda c: (-_p_pot(c), -c.geometry.ease, c.key))[: cfg.max_pots]
    safeties = sorted((c for c in cands if c.shot_type == "safety"),
                      key=lambda c: (_threat(c), c.key))[: cfg.max_safeties]
    return pots + safeties


def prepare(state: TableState, cfg: PrepConfig, rollout_key: tuple = ()) -> list[Candidate]:
    """Build the decision menu for the shooter in `state` (cue ball already placed)."""
    cands = generate(state)
    if cfg.access_level == "L0":
        menu = prune_l0(cands, cfg)
    else:
        working = round_robin_pots(cands, cfg.working_pots) + \
            [c for c in cands if c.shot_type == "safety"][: cfg.working_safeties]
        if not working:
            working = [c for c in cands if c.shot_type == "contact_only"] or contact_fallback(state)
        working = preview(state, working)
        if cfg.access_level == "L2":
            working = rollouts(state, working, cfg.n_rollouts, cfg.noise, rollout_key)
        menu = prune_annotated(working, cfg)
        if len(menu) < 2:  # make sure there is always a real choice
            extra = [c for c in working if c.key not in {m.key for m in menu}]
            extra.sort(key=lambda c: (-_p_pot(c), c.key))
            menu += extra[: 3 - len(menu)]
    return describe_all(state, assign_ids(menu), cfg.access_level)
