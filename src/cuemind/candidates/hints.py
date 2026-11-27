"""Simulation-free physics hints for each speed/spin variant (v3 condition A, D-031).

Pure geometry plus standard pool rules of thumb, no physics simulation:

* Object ball leaves along the line from the ghost ball to the pocket (ghost-ball model).
* Cue ball after contact:
    stun   -> along the tangent line (90 degrees to the object ball's path); stops on a
              straight shot;
    follow -> bends forward from the tangent line towards the object ball's path;
    draw   -> bends back from the tangent line towards the shooter.
* Travel distance after contact from the speed word and how much energy the cue ball keeps
  (a full hit with stun gives almost everything to the object ball), with mirror reflections
  off the cushions.
* Scratch warning when the predicted path passes close to a pocket.
* Leave: the predicted stopping point, and the geometric ease of the shooter's best next pot
  from there (same ghost-ball geometry as the menu, with the target ball removed).

These are approximations; `experiments/validate_hints.py` measures how often they agree with
pooltool.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pooltool as pt
from pooltool.ai.pot import core as pot

from cuemind.candidates.direct import direct_pots
from cuemind.candidates.model import Candidate
from cuemind.sim.state import BALL_R, TABLE, TABLE_L, TABLE_W, TableState, to_system

# Cue-ball path length after contact (m) = a + b * sin(cut), per (spin, speed). Rules of thumb
# fitted once on dev/floor states (never the main-run seeds), like a player learning from
# practice; no simulation is run at decision time (D-031).
TRAVEL = {
    ("stun", "soft"): (0.48, 1.39), ("stun", "medium"): (0.39, 2.31), ("stun", "firm"): (0.63, 2.96),
    ("follow", "soft"): (0.78, 1.19), ("follow", "medium"): (1.18, 1.27), ("follow", "firm"): (1.85, 1.31),
    ("draw", "soft"): (0.14, 1.16), ("draw", "medium"): (0.29, 2.09), ("draw", "firm"): (1.30, 1.84),
}
# Angle (deg) of the cue ball's first leg away from the tangent line, towards the object
# ball's path (+) or back towards the shooter (-): K * cos(cut)^3, K per (spin, speed);
# fitted on dev/floor states like TRAVEL.
BEND_K = {
    ("stun", "soft"): 22.0, ("stun", "medium"): 11.0, ("stun", "firm"): 7.0,
    ("follow", "soft"): 36.0, ("follow", "medium"): 31.0, ("follow", "firm"): 19.0,
    ("draw", "soft"): -11.0, ("draw", "medium"): -24.0, ("draw", "firm"): -8.0,
}
STRAIGHT_SIN = 0.15          # below this sin(cut) the shot counts as straight
SCRATCH_LEGS = 2             # cushion legs checked for a scratch (later legs are unreliable)
SCRATCH_MARGIN = 0.035       # path within pocket radius + ball radius + this -> scratch warning
CUSHION = BALL_R             # the cue ball centre reflects one radius inside the rails


@dataclass
class Hint:
    direction: tuple[float, float] | None   # unit vector after contact (None: cue ball stops)
    travel_m: float
    end: tuple[float, float]
    scratch_pocket: str | None
    next_ease: float
    next_ball: str | None
    cushions: int
    hits_ball: str | None = None   # first other ball on the predicted cue-ball path


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v * 0.0


def _first_ball(a: np.ndarray, b: np.ndarray, obstacles: dict) -> tuple[str, float] | None:
    """First obstacle ball whose centre comes within 2R of segment a->b: (id, fraction)."""
    ab = b - a
    L2 = float(ab @ ab)
    best = None
    for bid, xy in obstacles.items():
        c = np.asarray(xy, float)
        t = 0.0 if L2 < 1e-12 else float(np.clip((c - a) @ ab / L2, 0, 1))
        if np.linalg.norm(a + t * ab - c) < 2 * BALL_R and t > 1e-6 and (best is None or t < best[1]):
            best = (bid, t)
    return best


def _trace(start: np.ndarray, direction: np.ndarray, length: float, obstacles: dict | None = None):
    """Straight path with mirror reflections inside the cushion rectangle, stopping at the first
    obstacle ball. Returns (end point, segment points, cushions hit, ball hit or None)."""
    lo = np.array([CUSHION, CUSHION])
    hi = np.array([TABLE_W - CUSHION, TABLE_L - CUSHION])
    p, d, left = start.astype(float).copy(), direction.astype(float).copy(), length
    pts, cushions = [p.copy()], 0
    p = np.clip(p, lo, hi)
    while left > 1e-6 and cushions < 6:
        ts = []
        for k in range(2):
            if d[k] > 1e-12:
                ts.append(((hi[k] - p[k]) / d[k], k))
            elif d[k] < -1e-12:
                ts.append(((lo[k] - p[k]) / d[k], k))
        t, k = min(ts) if ts else (left, None)
        step = left if (t >= left or k is None) else t
        hit = _first_ball(p, p + d * step, obstacles or {})
        if hit is not None:
            p = p + d * step * hit[1]
            pts.append(p.copy())
            return p, pts, cushions, hit[0]
        if t >= left or k is None:
            p = p + d * left
            pts.append(p.copy())
            break
        p = p + d * t
        pts.append(p.copy())
        left -= t
        d[k] = -d[k]
        cushions += 1
    return p, pts, cushions, None


def _near_pocket(pts: list[np.ndarray]) -> str | None:
    for a, b in zip(pts, pts[1:]):
        for pid, pk in TABLE.pockets.items():
            c = np.asarray(pk.center[:2], float)
            ab = b - a
            t = 0.0 if float(ab @ ab) < 1e-12 else float(np.clip((c - a) @ ab / (ab @ ab), 0, 1))
            if np.linalg.norm(a + t * ab - c) < float(pk.radius) + BALL_R + SCRATCH_MARGIN:
                return pid
    return None


def _next_shot(state: TableState, cue_end: np.ndarray, potted: str) -> tuple[float, str | None]:
    balls = {b: xy for b, xy in state.balls.items() if b != potted}
    if any(math.dist(tuple(cue_end), xy) < 2 * BALL_R for xy in balls.values()):
        return 0.0, None  # the predicted spot is occupied: no reliable next-shot estimate
    groups = dict(state.groups)
    if groups[state.shooter] == "open" and potted != "8":
        mine = "solids" if int(potted) < 8 else "stripes"
        groups[state.shooter] = mine
        groups["P2" if state.shooter == "P1" else "P1"] = "stripes" if mine == "solids" else "solids"
    nxt = TableState(cue=(float(cue_end[0]), float(cue_end[1])), balls=balls, shooter=state.shooter,
                     groups=groups)
    targets = nxt.legal_targets
    if not targets:
        return 0.0, None
    pots = direct_pots(to_system(nxt), targets)
    return (pots[0].ease, pots[0].target_ball) if pots else (0.0, None)


def hint_for_pot(state: TableState, system: pt.System, c: Candidate) -> Hint:
    ball, pocket = system.balls[c.target_ball], system.table.pockets[c.pocket]
    ghost = np.asarray(pot.calc_shadow_ball_center(ball, system.table, pocket), float)[:2]
    pp = np.asarray(pot.get_potting_point(ball, system.table, pocket), float)[:2]
    obj = np.asarray(ball.xyz[:2], float)
    cue = np.asarray(state.cue, float)
    u_in = _unit(ghost - cue)            # cue ball's approach direction
    u_ob = _unit(pp - obj)               # object ball's departure direction
    perp = u_in - (u_in @ u_ob) * u_ob    # tangent-line component of the approach
    sin_cut = float(np.linalg.norm(perp))
    tang = _unit(perp) if sin_cut > 1e-6 else np.zeros(2)
    if sin_cut < STRAIGHT_SIN:  # straight shot: follow runs through, draw comes back, stun stops
        d = {"follow": u_ob, "draw": -u_ob}.get(c.spin_label, np.zeros(2))
    else:
        cos_cut = math.sqrt(max(0.0, 1.0 - sin_cut ** 2))
        bend = math.radians(BEND_K[(c.spin_label, c.power_label)] * cos_cut ** 3)
        d = _unit(math.cos(bend) * tang + math.sin(bend) * u_ob)
    a, b = TRAVEL[(c.spin_label, c.power_label)]
    travel = a + b * sin_cut
    others = {b: xy for b, xy in state.balls.items() if b != c.target_ball}
    hit = None
    if float(np.linalg.norm(d)) < 1e-6 or travel < 0.03:
        end, pts, cushions, direction = ghost, [ghost, ghost], 0, None
    else:
        end, pts, cushions, hit = _trace(ghost, d, travel, others)
        direction = (float(d[0]), float(d[1]))
    scratch = _near_pocket(pts[: SCRATCH_LEGS + 1])
    ne, nb = (0.0, None) if (scratch or hit) else _next_shot(state, end, c.target_ball)
    return Hint(direction, round(travel, 3), (round(float(end[0]), 3), round(float(end[1]), 3)),
                scratch, round(ne, 3), nb, cushions, hit)


def hints_for(state: TableState, cands: list[Candidate]) -> dict[tuple, Hint]:
    """Hints for every pot candidate, keyed by Candidate.key."""
    system = to_system(state)
    out = {}
    for c in cands:
        if c.shot_type == "pot":
            try:
                out[c.key] = hint_for_pot(state, system, c)
            except Exception:  # noqa: BLE001 - degenerate geometry: no hint
                continue
    return out


def _zone(xy: tuple[float, float]) -> str:
    from cuemind.candidates.describe import zone

    return zone(xy)


def hint_text(h: Hint) -> str:
    if h.scratch_pocket:
        from cuemind.sim.state import POCKET_LABEL

        return f"WARNING: the cue ball's path runs towards the {POCKET_LABEL[h.scratch_pocket]} (scratch risk)"
    if h.direction is None:
        move = f"the cue ball stops near the contact point, {_zone(h.end)}"
    elif h.hits_ball:
        move = (f"the cue ball runs" + (f" off {h.cushions} cushion{'s' if h.cushions > 1 else ''}"
                                        if h.cushions else "")
                + f" into the {h.hits_ball} {_zone(h.end)} (its final position is uncertain)")
    else:
        move = (f"the cue ball travels about {h.travel_m:.1f} m"
                + (f" off {h.cushions} cushion{'s' if h.cushions > 1 else ''}" if h.cushions else "")
                + f" and should stop {_zone(h.end)}")
    word = "easy" if h.next_ease >= 0.7 else "moderate" if h.next_ease >= 0.5 else "hard" if h.next_ease > 0 else "no"
    nxt = f"{word} next shot" + (f" on the {h.next_ball}" if h.next_ball else "")
    return f"{move}; {nxt}"


def rule_score(c: Candidate, h: Hint | None) -> float:
    """The rule-based reference player's objective (fixed a priori, D-031): geometric ease,
    plus the hinted leave, minus a scratch warning."""
    if c.shot_type != "pot" or h is None:
        return -1.0
    return c.geometry.ease * (1.0 + 0.5 * h.next_ease) - (1.0 if h.scratch_pocket else 0.0)

