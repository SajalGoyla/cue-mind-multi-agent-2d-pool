"""Simulation statistics for candidates, used only inside the reference bot B (D-029).

B simulates every candidate of the raw menu `n` times with the execution noise model (the
player knows its own skill; with zero noise one run is exact). The LLM systems never see these
numbers. Statistics:

    p_pot         called ball in the called pocket on a legal shot (or a legal 8-ball win)
    p_scratch     cue ball pocketed
    p_foul        any foul (scratch included)
    p_keep_turn   shooter shoots again
    next_ease     mean geometric ease of my best next pot, over runs where I keep the turn
    opp_threat    mean ease of the opponent's best pot, over runs where the turn passes;
                  a foul (ball in hand) or a lost game counts as threat 1.0
    cue_end_*     where the cue ball stops (mean and spread)

Random streams are keyed by (sim_key, candidate key, run index), so statistics do not depend
on menu order or on which other candidates exist.
"""

from __future__ import annotations

import numpy as np
import pooltool as pt

from cuemind.candidates.direct import direct_pots
from cuemind.candidates.model import Candidate, Rollouts
from cuemind.game.rules import Call, Judgement, advance, judge
from cuemind.sim.execute import SimulationError, extract_events, simulate
from cuemind.sim.noise import get_noise, rng_for
from cuemind.sim.state import TableState, positions_from_system, to_system

FOUL_THREAT = 1.0


def call_for(c: Candidate) -> Call:
    if c.shot_type == "pot":
        return Call(ball=c.target_ball, pocket=c.pocket)
    return Call(ball=c.target_ball, pocket=None, safety=True)


def _best_ease(state: TableState) -> float:
    """Ease of the shooter's best direct pot in `state` (0 if none)."""
    if state.cue is None or not state.legal_targets:
        return 0.0
    pots = direct_pots(to_system(state), state.legal_targets)
    return pots[0].ease if pots else 0.0


def _outcome(state: TableState, system: pt.System, c: Candidate,
             stroke) -> tuple[Judgement, TableState | None, pt.System]:
    """Simulate and judge one stroke. If pooltool cannot simulate it (D-014), the outcome
    is scored conservatively as a foul that ends the turn, leaving the balls unmoved."""
    try:
        shot = simulate(system, stroke)
    except SimulationError:
        j = Judgement(legal=False, foul="simulation failed", turn_over=True, game_over=False,
                      winner=None)
        return j, advance(state, j, state.cue, dict(state.balls)), system
    j = judge(state, extract_events(shot), call_for(c))
    if j.game_over:
        return j, None, shot
    cue, balls = positions_from_system(shot)
    return j, advance(state, j, cue, balls), shot


def rollouts(state: TableState, cands: list[Candidate], n: int, noise: str,
             sim_key: tuple = ()) -> list[Candidate]:
    system = to_system(state)
    nm = get_noise(noise)
    me = state.shooter
    out = []
    for c in cands:
        pots = scratches = fouls = keeps = 0
        next_e: list[float] = []
        threat: list[float] = []
        ends: list[tuple[float, float]] = []
        for i in range(n):
            stroke = nm.apply(c.stroke, rng_for(*sim_key, *c.key, i, "rollout"))
            j, nxt, shot = _outcome(state, system, c, stroke)
            cue_end, _ = positions_from_system(shot)
            if cue_end is not None:
                ends.append(cue_end)
            won = j.game_over and j.winner == me
            pots += int((j.called_ball_made and j.legal) or won)
            scratches += int((j.foul or "").startswith("scratch"))
            fouls += int(j.foul is not None)
            if won:
                keeps += 1
                next_e.append(1.0)
            elif j.game_over:          # lost the game
                threat.append(FOUL_THREAT)
            elif not j.turn_over:
                keeps += 1
                next_e.append(_best_ease(nxt))
            elif j.foul is not None:   # opponent gets ball in hand
                threat.append(FOUL_THREAT)
            else:
                threat.append(_best_ease(nxt))
        mean = sd = None
        if ends:
            arr = np.array(ends)
            mean = (round(float(arr[:, 0].mean()), 3), round(float(arr[:, 1].mean()), 3))
            sd = round(float(np.sqrt(((arr - arr.mean(axis=0)) ** 2).sum(axis=1).mean())), 3)
        out.append(c.model_copy(update={"rollouts": Rollouts(
            n=n, p_pot=round(pots / n, 3), p_scratch=round(scratches / n, 3),
            p_foul=round(fouls / n, 3), p_keep_turn=round(keeps / n, 3),
            cue_end_mean=mean, cue_end_sd_m=sd,
            next_ease=round(float(np.mean(next_e)), 3) if next_e else 0.0,
            opp_threat=round(float(np.mean(threat)), 3) if threat else 0.0)}))
    return out
