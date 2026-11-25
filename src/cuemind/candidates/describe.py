"""Plain-language rendering of positions and candidates for LLM prompts and the UI.

LLMs reason poorly over raw coordinates, so positions are given in *diamonds* (the sights
on the rails: 4 across the short rail, 8 along the long rail) plus named zones, and every
candidate gets a one-line description built only from the information allowed at the
current access level (L0 geometry, L1 + preview, L2 + rollout statistics).
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


def describe(c: Candidate, access_level: str) -> str:
    if c.shot_type == "pot":
        head = (f"Pot {c.target_ball} into the {POCKET_LABEL[c.pocket]}, {c.power_label} speed, "
                f"{c.spin_label}; cut {c.geometry.cut_deg:.0f} deg, {ease_word(c.geometry.ease)} geometry")
    elif c.shot_type == "safety":
        head = (f"Safety: play {c.power_label} into ball {c.target_ball} "
                f"(cut {c.geometry.cut_deg:.0f} deg), nothing called")
    else:
        head = f"Contact only: hit ball {c.target_ball} full at medium speed (last resort)"
    parts = [head]
    if access_level in ("L1", "L2") and c.preview is not None and c.rollouts is None:
        p = c.preview
        res = "pots it" if p.pots_called else ("FOUL: " + p.foul if p.foul else "no pot")
        parts.append(f"preview (perfect stroke): {res}; cue ends {where(p.cue_end)}; "
                     f"{p.next_viable_pots} pots available next")
    if access_level == "L2" and c.rollouts is not None:
        r = c.rollouts
        if c.shot_type == "pot":
            parts.append(
                f"over {r.n} realistic strokes: pots {r.p_pot:.0%}, scratch {r.p_scratch:.0%}, "
                f"foul {r.p_foul:.0%}; cue ends {where(r.cue_end_mean)}; "
                f"next shot if I continue: {ease_word(r.next_ease)} ({r.next_ease:.2f}); "
                f"opponent's chance if turn passes: {ease_word(r.opp_threat)} ({r.opp_threat:.2f})")
        else:
            parts.append(
                f"over {r.n} realistic strokes: foul {r.p_foul:.0%} (scratch {r.p_scratch:.0%}); "
                f"cue ends {where(r.cue_end_mean)}; opponent's best shot afterwards: "
                f"{ease_word(r.opp_threat)} ({r.opp_threat:.2f})")
    return ". ".join(parts) + "."


def describe_all(state: TableState, cands: list[Candidate], access_level: str) -> list[Candidate]:
    return [c.model_copy(update={"description": describe(c, access_level)}) for c in cands]


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
        lines.append("You are ON THE 8: pocket the 8 in the called pocket to win; pocketing the 8 "
                     "in the wrong pocket or on a shot with a foul (e.g. a scratch) loses.")
    else:
        lines.append(f"Your balls left ({len(mine)}): {', '.join(mine) or 'none'}.")
    if g[opp] != "open":
        lines.append(f"Opponent's balls left ({len(theirs)}): {', '.join(theirs) or 'none'}"
                     + (" -- opponent is on the 8." if state.on_the_eight(opp) else "."))
    lines.append(f"Cue ball at {where(state.cue)}.")
    lines.append("Object balls: " + "; ".join(
        f"{b} at {where(xy)}" for b, xy in sorted(state.balls.items(), key=lambda kv: int(kv[0]))) + ".")
    return "\n".join(lines)
