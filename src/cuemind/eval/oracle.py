"""Ground-truth value of a candidate: expected inning value (EIV), by Monte Carlo.

For each of M noisy executions of the candidate:
* pocketing the 8 legally wins (+WIN); any game loss scores -LOSS;
* a foul scores -FOUL and the opponent gets ball in hand (threat 1.0);
* otherwise the shooter scores +1 per own ball legally pocketed, and if the turn
  continues the inning is played on greedily for up to H more shots with the B1 rule
  (easiest pot, medium speed, stun) under the same noise model;
* when the turn passes, -THREAT_W x (ease of the opponent's best pot from the resulting
  position) is added.

Using B1 (not B2) for the continuation avoids making the oracle agree with B2 by
construction. The oracle is an approximation of game value; its limits are discussed in
the report. Random streams are keyed by (position id, candidate key, rollout index).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cuemind.candidates.annotate import call_for
from cuemind.candidates.direct import direct_pots
from cuemind.candidates.model import Candidate
from cuemind.game.rules import Call, advance, judge
from cuemind.sim.execute import SimulationError, Stroke, extract_events, simulate
from cuemind.sim.noise import get_noise, rng_for
from cuemind.sim.state import TableState, group_balls, positions_from_system, to_system

WIN, LOSS, FOUL, THREAT_W = 5.0, 5.0, 1.0, 0.75
CONT_V0 = 2.5


@dataclass(frozen=True)
class OracleConfig:
    m: int = 24
    horizon: int = 2
    noise: str = "medium"


def _own_potted(state: TableState, potted: dict[str, str], group_after: str) -> int:
    if group_after == "open":
        return sum(1 for b in potted if b != "8")
    return sum(1 for b in potted if b in group_balls(group_after))


def _opp_ease(nxt: TableState) -> float:
    if nxt.cue is None or not nxt.legal_targets:
        return 1.0 if nxt.cue is None else 0.0
    pots = direct_pots(to_system(nxt), nxt.legal_targets)
    return pots[0].ease if pots else 0.0


def _shot_value(state: TableState, call: Call, stroke: Stroke) -> tuple[float, TableState | None, bool]:
    """Value of one executed shot, the next state, and whether the shooter continues."""
    me = state.shooter
    try:
        shot = simulate(to_system(state), stroke)
    except SimulationError:
        return -FOUL - THREAT_W, None, False
    j = judge(state, extract_events(shot), call)
    if j.game_over:
        return (WIN if j.winner == me else -LOSS), None, False
    cue, balls = positions_from_system(shot)
    nxt = advance(state, j, cue, balls)
    if j.foul is not None:
        return -FOUL - THREAT_W * 1.0, nxt, False
    v = float(_own_potted(state, j.potted, nxt.groups[me]))
    if j.turn_over:
        return v - THREAT_W * _opp_ease(nxt), nxt, False
    return v, nxt, True


def eiv(state: TableState, cand: Candidate, cfg: OracleConfig, key: tuple) -> tuple[float, float]:
    """(mean, standard error) of the expected inning value of `cand` in `state`."""
    nm = get_noise(cfg.noise)
    vals = []
    for i in range(cfg.m):
        stroke = nm.apply(cand.stroke, rng_for(*key, *cand.key, i, "oracle"))
        v, s, cont = _shot_value(state, call_for(cand), stroke)
        h = 0
        while cont and s is not None and h < cfg.horizon:
            pots = direct_pots(to_system(s), s.legal_targets)
            if not pots:
                break
            g = pots[0]
            st = nm.apply(Stroke(phi=g.phi, V0=CONT_V0), rng_for(*key, *cand.key, i, "cont", h))
            dv, s, cont = _shot_value(s, Call(g.target_ball, g.pocket), st)
            v += dv
            h += 1
        vals.append(v)
    arr = np.asarray(vals)
    return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0


def oracle_values(state: TableState, cands: list[Candidate], cfg: OracleConfig,
                  key: tuple) -> dict[str, dict[str, float]]:
    out = {}
    for c in cands:
        m, se = eiv(state, c, cfg, key)
        out[c.shot_id] = {"eiv": round(m, 4), "se": round(se, 4)}
    return out
