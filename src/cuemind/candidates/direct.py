"""Direct (no-cushion) pot geometry using pooltool's ghost-ball helpers (pooltool.ai.pot,
Apache-2.0). In the feasibility spike these predicted pots at 98 % precision."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pooltool as pt
import pooltool.constants as const
from pooltool.ai.pot import core as pot

DEFAULT_MAX_CUT = 80.0


@dataclass(frozen=True)
class PotGeometry:
    target_ball: str
    pocket: str
    phi: float                  # aim direction (deg) for a centre-ball hit
    cut_deg: float
    cue_to_ghost_m: float
    ball_to_pocket_m: float
    required_precision_deg: float
    ghost: tuple[float, float]

    @property
    def ease(self) -> float:
        """Geometric ease in (0, 1]; higher is easier. Combines cut angle and path length
        (same functional form as the spike's difficulty factor, so scores are comparable)."""
        length_mm = (self.cue_to_ghost_m + self.ball_to_pocket_m) * 1000.0
        return round(1.0 / (1.0 + (self.cut_deg * 300.0 + length_mm) / 10_000.0), 4)


def _on_table(system: pt.System) -> list:
    # pooltool's occlusion helper checks the wrong attribute for pocketed balls
    # (ai/pot/core.py, 0.6.0), so only on-table balls are passed in.
    return [b for b in system.balls.values() if b.state.s != const.pocketed]


def direct_pots(system: pt.System, targets: list[str], max_cut: float = DEFAULT_MAX_CUT,
                with_reasons: bool = False):
    """Viable direct pots for the given targets. Never raises for geometric reasons.

    Returns a list of PotGeometry (and, with with_reasons=True, also a list of
    (target, pocket, reasons) for the rejected combinations).
    """
    table = system.table
    cue = system.balls["cue"]
    balls = _on_table(system)
    out: list[PotGeometry] = []
    rejected: list[tuple[str, str, list[str]]] = []
    for tid in targets:
        if tid not in system.balls or system.balls[tid].state.s == const.pocketed:
            continue
        ball = system.balls[tid]
        for pocket in table.pockets.values():
            reasons = []
            try:
                pp = pot.get_potting_point(ball, table, pocket)
                ghost = pot.calc_shadow_ball_center(ball, table, pocket)
            except Exception:  # noqa: BLE001 - degenerate geometry (ball in the jaws)
                rejected.append((tid, pocket.id, ["degenerate_geometry"]))
                continue
            aim_vec = ghost - cue.xyz[:2]
            pocket_vec = pp - ball.xyz[:2]
            if np.linalg.norm(aim_vec) < 1e-9:
                rejected.append((tid, pocket.id, ["degenerate_geometry"]))
                continue
            cut = abs(float(pot.angle_between_vectors(aim_vec, pocket_vec)))
            if pot.is_pocket_occluded(ball, table, pocket, balls):
                reasons.append("pocket_path_blocked")
            if not pot.is_room_for_cue_ball(ball, table, pocket, balls):
                reasons.append("no_room_for_ghost_ball")
            if pot.is_jaw_in_way(ball, table, pocket):
                reasons.append("jaw_in_way")
            if pot.is_object_ball_occluded(cue, ball, table, pocket, balls):
                reasons.append("cue_path_blocked")
            if cut > max_cut:
                reasons.append("cut_angle_too_large")
            if reasons:
                rejected.append((tid, pocket.id, reasons))
                continue
            phi = float(pot.calc_potting_angle(cue, ball, table, pocket)) % 360.0
            try:
                prec = float(pot.required_precision(cue.state, ball.state, table, pocket))
            except Exception:  # noqa: BLE001
                prec = float("nan")
            out.append(PotGeometry(
                target_ball=tid, pocket=pocket.id, phi=round(phi, 4), cut_deg=round(cut, 2),
                cue_to_ghost_m=round(float(np.linalg.norm(aim_vec)), 4),
                ball_to_pocket_m=round(float(np.linalg.norm(pocket_vec)), 4),
                required_precision_deg=round(prec, 3),
                ghost=(round(float(ghost[0]), 5), round(float(ghost[1]), 5)),
            ))
    out.sort(key=lambda g: (-g.ease, int(g.target_ball), g.pocket))
    return (out, rejected) if with_reasons else out
