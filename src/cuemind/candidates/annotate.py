"""Simulation annotations for candidates (access levels L1 and L2).

L1 (single preview): one noise-free simulation per candidate.
L2 (Monte Carlo): n noisy rollouts per candidate with the *same* noise model used for
execution (the player knows their own skill). Statistics:

    p_pot         called ball in the called pocket on a legal shot (or a legal 8-ball win)
    p_scratch     cue ball pocketed
    p_foul        any foul (scratch included)
    p_keep_turn   shooter shoots again
    next_ease     mean geometric ease of my best next pot, over rollouts where I keep the turn
    opp_threat    mean ease of the opponent's best pot, over rollouts where the turn passes;
                  a foul (ball in hand) or a lost game counts as threat 1.0
    cue_end_*     where the cue ball stops (mean and spread)

Rollout random streams are keyed by (rollout_key, candidate key, rollout index), so they do
not depend on menu order or on which other candidates exist.
"""

from __future__ import annotations

import numpy as np
import pooltool as pt

from cuemind.candidates.direct import direct_pots
from cuemind.candidates.model import Candidate, Preview, Rollouts
from cuemind.game.rules import Call, Judgement, advance, judge
from cuemind.sim.execute import SimulationError, extract_events, simulate
from cuemind.sim.noise import get_noise, rng_for
from cuemind.sim.state import TableState, positions_from_system, to_system

FOUL_THREAT = 1.0


def call_for(c: Candidate) -> Call:
    if c.shot_type == "pot":
        return Call(ball=c.target_ball, pocket=c.pocket)
    return Call(ball=c.target_ball, pocket=None, safety=True)


def _best_ease(state: TableState) -> tuple[float, int]:
    """(ease of the shooter's best direct pot, number of viable pots) in `state`."""
    if state.cue is None or not state.legal_targets:
        return 0.0, 0
    pots = direct_pots(to_system(state), state.legal_targets)
    return (pots[0].ease if pots else 0.0), len(pots)


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


def preview(state: TableState, cands: list[Candidate]) -> list[Candidate]:
    system = to_system(state)
    out = []
    for c in cands:
        j, nxt, shot = _outcome(state, system, c, c.stroke)
        cue_end, _ = positions_from_system(shot)
        n_next = 0
        if nxt is not None and not j.turn_over:
            n_next = _best_ease(nxt)[1]
        won = j.game_over and j.winner == state.shooter
        out.append(c.model_copy(update={"preview": Preview(
            pots_called=bool(j.called_ball_made and j.legal) or won,
            scratch=(j.foul or "").startswith("scratch"),
            legal=j.legal, foul=j.foul,
            cue_end=None if cue_end is None else (round(cue_end[0], 3), round(cue_end[1], 3)),
            next_viable_pots=n_next)}))
    return out


def rollouts(state: TableState, cands: list[Candidate], n: int, noise: str,
             rollout_key: tuple = ()) -> list[Candidate]:
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
            stroke = nm.apply(c.stroke, rng_for(*rollout_key, *c.key, i, "rollout"))
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
                next_e.append(_best_ease(nxt)[0])
            elif j.foul is not None:   # opponent gets ball in hand
                threat.append(FOUL_THREAT)
            else:
                threat.append(_best_ease(nxt)[0])
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


def ease_from(state: TableState) -> float:
    """Geometric ease of the shooter's best direct pot in `state` (0 if none)."""
    return _best_ease(state)[0]
