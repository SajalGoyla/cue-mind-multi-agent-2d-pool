"""Deterministic candidate generation (access level L0: geometry only).

* Pots: every viable direct pot x power {soft, medium, firm} x spin {draw, stun, follow}.
* Safeties: contact shots on each legal target at a few cut angles and soft/medium speed,
  kept only if a noise-free simulation of the shot is legal (no foul, no game loss). This
  validity filter uses the simulator, but no statistics from it are shown at level L0.
* Contact-only fallback: aim at the centre of each legal ball, used when nothing else exists.

The generator only proposes shots. Simulation statistics are added by `annotate.py`.
"""

from __future__ import annotations

import math

import pooltool as pt

from cuemind.candidates.direct import direct_pots
from cuemind.candidates.model import POWERS, SPINS, Candidate, Geometry
from cuemind.game.rules import Call, judge
from cuemind.sim.execute import SimulationError, extract_events, simulate
from cuemind.sim.state import TableState, to_system

SAFETY_CUTS = (0.0, -35.0, 35.0)
SAFETY_POWERS = ("soft", "medium")


def pot_candidates(state: TableState, system: pt.System) -> list[Candidate]:
    out = []
    for g in direct_pots(system, state.legal_targets):
        geom = Geometry(cut_deg=g.cut_deg, cue_to_ball_m=g.cue_to_ghost_m,
                        ball_to_pocket_m=g.ball_to_pocket_m,
                        required_precision_deg=g.required_precision_deg, ease=g.ease)
        for p_label, v0 in POWERS.items():
            for s_label, b in SPINS.items():
                out.append(Candidate(shot_type="pot", target_ball=g.target_ball, pocket=g.pocket,
                                     power_label=p_label, spin_label=s_label, phi=g.phi, V0=v0,
                                     b=b, geometry=geom))
    return out


def _contact_geometry(state: TableState, ball: str, cut: float) -> Geometry:
    d = math.dist(state.cue, state.balls[ball])
    return Geometry(cut_deg=abs(cut), cue_to_ball_m=round(d, 4), ease=0.0)


def safety_candidates(state: TableState, system: pt.System) -> list[Candidate]:
    out = []
    for ball in state.legal_targets:
        for cut in SAFETY_CUTS:
            try:
                phi = float(pt.aim.at_ball(system, ball, cut=cut)) % 360.0
            except Exception:  # noqa: BLE001
                continue
            for p_label in SAFETY_POWERS:
                cand = Candidate(shot_type="safety", target_ball=ball, pocket=None,
                                 power_label=p_label, spin_label="stun", phi=round(phi, 4),
                                 V0=POWERS[p_label], geometry=_contact_geometry(state, ball, cut))
                try:
                    shot = simulate(system, cand.stroke)
                except SimulationError:
                    continue
                ev = extract_events(shot)
                j = judge(state, ev, Call(ball=ball, pocket=None, safety=True))
                if j.foul is None and not j.game_over:
                    out.append(cand)
    return out


def contact_fallback(state: TableState) -> list[Candidate]:
    out = []
    for ball in state.legal_targets:
        bx, by = state.balls[ball]
        phi = math.degrees(math.atan2(by - state.cue[1], bx - state.cue[0])) % 360.0
        out.append(Candidate(shot_type="contact_only", target_ball=ball, pocket=None,
                             power_label="medium", phi=round(phi, 4), V0=POWERS["medium"],
                             geometry=_contact_geometry(state, ball, 0.0)))
    return out


def assign_ids(cands: list[Candidate]) -> list[Candidate]:
    return [c.model_copy(update={"shot_id": f"S{i + 1:02d}"}) for i, c in enumerate(cands)]


def generate(state: TableState, *, with_safeties: bool = True) -> list[Candidate]:
    """All candidate shots for the shooter (cue ball must already be placed).

    Order is deterministic: pots by geometric ease, then safeties, then fallbacks.
    IDs are provisional; pruning re-assigns them.
    """
    if state.cue is None:
        raise ValueError("place the cue ball before generating candidates")
    system = to_system(state)
    pots = pot_candidates(state, system)
    pots.sort(key=lambda c: (-c.geometry.ease, int(c.target_ball), c.pocket,
                             list(POWERS).index(c.power_label), list(SPINS).index(c.spin_label)))
    safeties = safety_candidates(state, system) if with_safeties else []
    cands = pots + safeties
    if not cands:
        cands = contact_fallback(state)
    return assign_ids(cands)
