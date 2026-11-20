"""Shared ball-in-hand placement policy (identical for every system, not LLM-controlled).

After a foul the incoming player may place the cue ball anywhere. We evaluate a grid of
spots and pick the one whose best direct pot is geometrically easiest. Keeping this
non-LLM isolates the research question (shot selection) from placement skill.
"""

from __future__ import annotations

import math

import pooltool as pt

from cuemind.candidates.direct import direct_pots
from cuemind.sim.state import BALL_R, TABLE_L, TABLE_W, TableState, to_system

GRID_NX, GRID_NY = 5, 9
MARGIN = 2.2 * BALL_R
MIN_GAP = 2.2 * BALL_R


def grid_spots() -> list[tuple[float, float]]:
    xs = [MARGIN + i * (TABLE_W - 2 * MARGIN) / (GRID_NX - 1) for i in range(GRID_NX)]
    ys = [MARGIN + j * (TABLE_L - 2 * MARGIN) / (GRID_NY - 1) for j in range(GRID_NY)]
    return [(x, y) for y in ys for x in xs]


def _move_cue(system: pt.System, xy: tuple[float, float]) -> None:
    system.balls["cue"] = pt.Ball.create("cue", xy=xy)


def place_cue_ball(state: TableState) -> tuple[tuple[float, float], float]:
    """Return (spot, best ease at that spot). Deterministic for a given state."""
    occupied = list(state.balls.values())
    spots = [s for s in grid_spots() if all(math.dist(s, o) >= MIN_GAP for o in occupied)]
    if not spots:
        raise RuntimeError("no free spot for the cue ball")
    targets = state.legal_targets
    probe = to_system(state.model_copy(update={"cue": spots[0]}))
    best_spot, best_ease = spots[0], -1.0
    for s in spots:
        _move_cue(probe, s)
        pots = direct_pots(probe, targets)
        ease = pots[0].ease if pots else 0.0
        # Tie-break towards the table centre for robustness.
        if ease > best_ease + 1e-9:
            best_spot, best_ease = s, ease
    return best_spot, best_ease


def resolve_ball_in_hand(state: TableState) -> TableState:
    """If the shooter has ball in hand, return the state with the cue ball placed."""
    if not state.ball_in_hand and state.cue is not None:
        return state
    spot, _ = place_cue_ball(state)
    return state.model_copy(update={"cue": spot, "ball_in_hand": False})
