"""Plain-language rendering of positions and candidates for LLM prompts and the UI.

Positions are given in *diamonds* (the sights on the rails: 4 across the short rail, 8 along
the long rail) plus named zones. Every candidate gets a one-line description built only from
its geometry: ball, pocket, speed, spin, cut angle and a difficulty word. No simulated outcome
is ever shown (v2, D-029).
"""

from __future__ import annotations

from cuemind.candidates.model import Candidate
from cuemind.sim.state import POCKET_LABEL, TABLE_L, TABLE_W, TableState, other


def diamonds(xy: tuple[float, float]) -> tuple[float, float]:
    return round(xy[0] / TABLE_W * 4, 1), round(xy[1] / TABLE_L * 8, 1)


def zone(xy: tuple[float, float]) -> str:
    x, y = xy
    if y < TABLE_L * 0.2:
        ys = "near the bottom rail"
    elif y < TABLE_L * 0.45:
        ys = "lower half"
    elif y < TABLE_L * 0.55:
        ys = "middle (side-pocket line)"
    elif y < TABLE_L * 0.8:
        ys = "upper half"
    else:
        ys = "near the top rail"
    if x < TABLE_W * 0.18:
        xs = ", on the left rail"
    elif x > TABLE_W * 0.82:
        xs = ", on the right rail"
    else:
        xs = ""
    return ys + xs


def where(xy: tuple[float, float] | None) -> str:
    if xy is None:
        return "pocketed"
    dx, dy = diamonds(xy)
    return f"({dx}, {dy}) {zone(xy)}"


def ease_word(e: float) -> str:
    if e >= 0.7:
        return "easy"
    if e >= 0.5:
        return "moderate"
    if e > 0.0:
        return "hard"
    return "none"


def describe(c: Candidate) -> str:
    if c.shot_type == "pot":
        return (f"Pot {c.target_ball} into the {POCKET_LABEL[c.pocket]}, {c.power_label} speed, "
                f"{c.spin_label}; cut {c.geometry.cut_deg:.0f} deg, {ease_word(c.geometry.ease)} geometry.")
    if c.shot_type == "safety":
        return (f"Safety: play {c.power_label} into ball {c.target_ball} "
                f"(cut {c.geometry.cut_deg:.0f} deg), nothing called.")
    return f"Contact only: hit ball {c.target_ball} full at medium speed (last resort)."


def describe_all(cands: list[Candidate]) -> list[Candidate]:
    return [c.model_copy(update={"description": describe(c)}) for c in cands]


def state_summary(state: TableState) -> str:
    """Compact natural-language summary of the position for prompts."""
    me, opp = state.shooter, other(state.shooter)
    g = state.groups
    lines = []
    if g[me] == "open":
        lines.append("The table is OPEN (groups not yet assigned): any ball except the 8 may be "
                     "hit first; pocketing a called ball assigns you that group.")
    else:
        lines.append(f"You play {g[me]}; the opponent plays {g[opp]}.")
    mine, theirs = state.remaining(me), state.remaining(opp)
    if state.on_the_eight(me):
        lines.append("You are ON THE 8: pocket the 8 in the called pocket to win. Pocketing the 8 in "
                     "another pocket, or together with a foul, loses; any other foul gives the "
                     "opponent ball in hand.")
    else:
        lines.append(f"Your balls left ({len(mine)}): {', '.join(mine) or 'none'}.")
    if g[opp] != "open":
        lines.append(f"Opponent's balls left ({len(theirs)}): {', '.join(theirs) or 'none'}"
                     + (" -- opponent is on the 8." if state.on_the_eight(opp) else "."))
    lines.append(f"Cue ball at {where(state.cue)}.")
    lines.append("Object balls: " + "; ".join(
        f"{b} at {where(xy)}" for b, xy in sorted(state.balls.items(), key=lambda kv: int(kv[0]))) + ".")
    return "\n".join(lines)
