"""Table state: the single serialisable description of a pool position.

Frame (pooltool, canonical for CueMind): metres, origin at the bottom-left corner of the
cushion-nose rectangle, +x to the right (0..W), +y up the table (0..L). The rack sits at
the top ("foot") end; the break is played from the bottom ("head") end.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Literal

import pooltool as pt
import pooltool.constants as const
from pydantic import BaseModel, Field

TABLE = pt.Table.from_game_type(pt.GameType.EIGHTBALL)
TABLE_W: float = float(TABLE.w)
TABLE_L: float = float(TABLE.l)
BALL_R: float = float(pt.BallParams.default().R)

SOLIDS = tuple(str(i) for i in range(1, 8))
STRIPES = tuple(str(i) for i in range(9, 16))
ALL_OBJECT_BALLS = SOLIDS + ("8",) + STRIPES

POCKET_LABEL = {
    "lb": "bottom-left corner", "rb": "bottom-right corner",
    "lc": "left side", "rc": "right side",
    "lt": "top-left corner", "rt": "top-right corner",
}

Group = Literal["open", "solids", "stripes"]
PlayerId = Literal["P1", "P2"]


def other(player: str) -> str:
    return "P2" if player == "P1" else "P1"


def group_balls(group: str) -> tuple[str, ...]:
    if group == "solids":
        return SOLIDS
    if group == "stripes":
        return STRIPES
    return SOLIDS + STRIPES


def ball_group(ball_id: str) -> str:
    if ball_id == "cue":
        return "cue"
    if ball_id == "8":
        return "eight"
    return "solids" if int(ball_id) < 8 else "stripes"


class TableState(BaseModel):
    """Everything a policy needs to know about the position before a shot."""

    schema_version: int = 1
    cue: tuple[float, float] | None = Field(
        description="Cue-ball position; None when the cue ball must be placed (ball in hand)")
    balls: dict[str, tuple[float, float]] = Field(description="Object balls still on the table")
    shooter: PlayerId = "P1"
    groups: dict[str, Group] = Field(default_factory=lambda: {"P1": "open", "P2": "open"})
    ball_in_hand: bool = False
    shot_number: int = 0
    break_shot: bool = False

    # ------------------------------------------------------------------ derived
    @property
    def shooter_group(self) -> Group:
        return self.groups[self.shooter]

    def remaining(self, player: str) -> list[str]:
        """Balls the player still has to pocket before the 8 (sorted numerically)."""
        g = self.groups[player]
        if g == "open":
            return sorted((b for b in self.balls if b != "8"), key=int)
        return sorted((b for b in self.balls if b in group_balls(g)), key=int)

    def on_the_eight(self, player: str) -> bool:
        """True when the player has no balls left before the 8 (also covers the rare open
        table where every other ball is gone)."""
        return not self.remaining(player) and "8" in self.balls

    @property
    def legal_targets(self) -> list[str]:
        """Balls the shooter may legally hit first."""
        if self.on_the_eight(self.shooter):
            return ["8"]
        return self.remaining(self.shooter)

    def to_json(self) -> str:
        return self.model_dump_json()


def state_hash(state: TableState) -> str:
    """Stable hash of the position (positions rounded to 0.1 mm)."""
    d = state.model_dump()
    d["cue"] = None if state.cue is None else [round(v, 4) for v in state.cue]
    d["balls"] = {k: [round(v, 4) for v in xy] for k, xy in sorted(state.balls.items())}
    blob = json.dumps(d, sort_keys=True)
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()[:16]


def _in_pocket_mouth(x: float, y: float) -> bool:
    return any(math.dist((x, y), p.center[:2]) < p.radius + 2 * BALL_R
               for p in TABLE.pockets.values())


def validate(state: TableState) -> list[str]:
    """Return a list of problems (empty == valid). Never raises."""
    problems: list[str] = []
    pts = dict(state.balls)
    if state.cue is not None:
        pts["cue"] = state.cue
    elif not state.ball_in_hand:
        problems.append("cue ball missing but ball_in_hand is False")
    for bid, (x, y) in pts.items():
        if bid != "cue" and bid not in ALL_OBJECT_BALLS:
            problems.append(f"unknown ball id {bid!r}")
        if not (math.isfinite(x) and math.isfinite(y)):
            problems.append(f"ball {bid}: non-finite position")
            continue
        on_surface = BALL_R - 1e-6 <= x <= TABLE_W - BALL_R + 1e-6 and \
            BALL_R - 1e-6 <= y <= TABLE_L - BALL_R + 1e-6
        if not on_surface and not _in_pocket_mouth(x, y):
            problems.append(f"ball {bid}: off the playing surface ({x:.3f}, {y:.3f})")
    items = list(pts.items())
    for i, (a, pa) in enumerate(items):
        for b, pb in items[i + 1:]:
            if math.dist(pa, pb) < 2 * BALL_R - 1e-6:
                problems.append(f"balls {a} and {b} overlap")
    for p, g in state.groups.items():
        if g not in ("open", "solids", "stripes"):
            problems.append(f"bad group {g!r} for {p}")
    if state.groups["P1"] != "open" and state.groups["P1"] == state.groups["P2"]:
        problems.append("both players have the same group")
    return problems


def to_system(state: TableState) -> pt.System:
    """Build a pooltool System. The cue ball must be placed (state.cue not None)."""
    if state.cue is None:
        raise ValueError("cue ball has no position; place it (ball in hand) first")
    balls = {"cue": pt.Ball.create("cue", xy=state.cue)}
    for bid, xy in state.balls.items():
        balls[bid] = pt.Ball.create(bid, xy=xy)
    return pt.System(cue=pt.Cue(cue_ball_id="cue"), table=TABLE.copy(), balls=balls)


def positions_from_system(system: pt.System) -> tuple[tuple[float, float] | None,
                                                      dict[str, tuple[float, float]]]:
    """(cue position or None if pocketed, on-table object balls) after a simulation."""
    cue = None
    balls: dict[str, tuple[float, float]] = {}
    for bid, b in system.balls.items():
        if b.state.s == const.pocketed:
            continue
        xy = (float(b.state.rvw[0][0]), float(b.state.rvw[0][1]))
        if bid == "cue":
            cue = xy
        else:
            balls[bid] = xy
    return cue, balls


def rack_state(seed: int = 0) -> TableState:
    """Standard 8-ball rack, cue ball on the head spot, P1 to break.

    Ball *positions* come from pooltool's rack (with its NumPy jitter seeded). Ball
    *numbers* are assigned here: pooltool picks them with `random.choice(list(set))`, whose
    order depends on Python's per-process hash seed, so the same seed gave different racks
    in different processes (D-013). Rack constraints: the 8 in the middle of the third row,
    one solid and one stripe in the two back corners, the rest in seeded random order.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    rack = pt.get_rack(pt.GameType.EIGHTBALL, TABLE)
    rack.pop("cue")
    spots = sorted(((float(b.state.rvw[0][0]), float(b.state.rvw[0][1])) for b in rack.values()),
                   key=lambda xy: (round(xy[1], 2), xy[0]))
    assert len(spots) == 15
    rng = random.Random(f"rack-{seed}")
    solids, stripes = list(SOLIDS), list(STRIPES)
    rng.shuffle(solids)
    rng.shuffle(stripes)
    corners = [solids.pop(), stripes.pop()]
    rng.shuffle(corners)
    rest = solids + stripes
    rng.shuffle(rest)
    ids: list[str | None] = [None] * 15
    ids[4] = "8"                         # rows: [0] [1,2] [3,4,5] [6..9] [10..14]
    ids[10], ids[14] = corners
    it = iter(rest)
    ids = [b if b is not None else next(it) for b in ids]
    return TableState(
        cue=(TABLE_W / 2, TABLE_L / 4),  # head spot
        balls={bid: xy for bid, xy in zip(ids, spots)},
        shooter="P1", break_shot=True,
    )
