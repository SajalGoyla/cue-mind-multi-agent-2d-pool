"""Simplified, documented 8-ball rules on top of pooltool's physics events.

We use our own rules layer instead of pooltool's `_EightBall` ruleset because we need to
start games from arbitrary mid-game positions (group assignment, shooter, ball in hand),
and because `_EightBall` cannot be copied. pooltool still provides all physics and events.

Rules (a simplified WPA called-shot 8-ball; see docs/architecture.md):
* Break: legal if a ball is pocketed or >= 4 object balls reach a cushion; a scratch or
  illegal break is a foul. The table stays open after the break. An 8 pocketed on the
  break is re-spotted on the foot spot (no win, no loss). The breaker continues only if an
  object ball dropped and there was no foul.
* Open table: any object ball except the 8 may be hit first. The group is assigned when a
  player legally pockets the called ball in the called pocket.
* Fouls: cue ball pocketed (scratch); no ball contacted; first contact not a legal target;
  no cushion contacted after first contact and nothing pocketed. A foul gives the opponent
  ball in hand anywhere on the table. Pocketed object balls stay down.
* The turn continues only if the shot is legal, it was not declared a safety, and the
  called ball went into the called pocket.
* The 8: pocketing it legally (player on the 8, called pocket, no foul) wins; pocketing it
  any other way loses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from cuemind.sim.execute import ShotEvents
from cuemind.sim.state import BALL_R, TABLE_L, TABLE_W, TableState, ball_group, other

FOOT_SPOT = (TABLE_W / 2, TABLE_L * 3 / 4)


@dataclass(frozen=True)
class Call:
    """What the shooter declared: target ball and pocket, or a safety."""

    ball: str | None
    pocket: str | None
    safety: bool = False


@dataclass
class Judgement:
    legal: bool
    foul: str | None
    turn_over: bool
    game_over: bool
    winner: str | None
    potted: dict[str, str] = field(default_factory=dict)
    called_ball_made: bool = False
    group_for_shooter: str | None = None  # set when this shot assigns groups
    note: str = ""


def judge(state: TableState, ev: ShotEvents, call: Call) -> Judgement:
    shooter, opp = state.shooter, other(state.shooter)
    potted = ev.object_pocketed

    if state.break_shot:
        foul = None
        if ev.scratched:
            foul = "scratch on the break"
        elif not potted and ev.object_balls_hit_cushion < 4:
            foul = "illegal break (no ball pocketed and fewer than 4 balls reached a cushion)"
        dropped = [b for b in potted if b != "8"]
        turn_over = foul is not None or not dropped
        note = "8 pocketed on the break: re-spotted" if "8" in potted else ""
        return Judgement(legal=foul is None, foul=foul, turn_over=turn_over, game_over=False,
                         winner=None, potted=potted, note=note)

    legal_targets = state.legal_targets
    foul = None
    if ev.scratched:
        foul = "scratch (cue ball pocketed)"
    elif ev.first_ball_hit is None:
        foul = "no ball contacted"
    elif ev.first_ball_hit not in legal_targets:
        foul = f"wrong first contact (hit {ev.first_ball_hit}, legal: {','.join(legal_targets)})"
    elif not ev.cushion_after_contact and not potted:
        foul = "no cushion after contact"

    called_made = (not call.safety and call.ball is not None
                   and potted.get(call.ball) == call.pocket)

    if "8" in potted:
        won = (state.on_the_eight(shooter) and foul is None and call.ball == "8"
               and potted["8"] == call.pocket)
        winner = shooter if won else opp
        note = "8 pocketed legally" if won else "8 pocketed illegally"
        return Judgement(legal=foul is None and won, foul=foul, turn_over=True, game_over=True,
                         winner=winner, potted=potted, called_ball_made=called_made, note=note)

    assign = state.groups[shooter] == "open" and foul is None and called_made
    turn_over = foul is not None or not called_made
    return Judgement(legal=foul is None, foul=foul, turn_over=turn_over, game_over=False,
                     winner=None, potted=potted, called_ball_made=called_made,
                     group_for_shooter=ball_group(call.ball) if assign else None)


def free_spot(xy: tuple[float, float], occupied: list[tuple[float, float]]) -> tuple[float, float]:
    """Nearest free spot along the long axis (towards the foot rail first, then back)."""
    x, y = xy
    step = 0.5 * BALL_R
    for k in range(0, 200):
        for sign in (1, -1):
            cand = (x, y + sign * k * step)
            if BALL_R <= cand[1] <= TABLE_L - BALL_R and all(
                    math.dist(cand, o) >= 2.05 * BALL_R for o in occupied):
                return cand
    raise RuntimeError("no free spot found on the long axis")


def advance(state: TableState, j: Judgement, cue_after: tuple[float, float] | None,
            balls_after: dict[str, tuple[float, float]]) -> TableState:
    """Next state after a shot (the game-over case is handled by the caller)."""
    balls = dict(balls_after)
    if state.break_shot and "8" in j.potted:
        occupied = list(balls.values()) + ([cue_after] if cue_after else [])
        balls["8"] = free_spot(FOOT_SPOT, occupied)

    groups = dict(state.groups)
    if j.group_for_shooter is not None:
        g = j.group_for_shooter
        groups[state.shooter] = g
        groups[other(state.shooter)] = "stripes" if g == "solids" else "solids"

    if j.foul is not None:
        shooter, cue, bih = other(state.shooter), None, True
    elif j.turn_over:
        shooter, cue, bih = other(state.shooter), cue_after, False
    else:
        shooter, cue, bih = state.shooter, cue_after, False
    if cue is None:
        bih = True

    return TableState(cue=cue, balls=balls, shooter=shooter, groups=groups, ball_in_hand=bih,
                      shot_number=state.shot_number + 1, break_shot=False)
