"""Deterministic, seeded break shot shared by every policy (not LLM-controlled)."""

from __future__ import annotations

import math

import numpy as np

from cuemind.sim.execute import Stroke
from cuemind.sim.state import TableState

BREAK_SPEED = 8.0         # m/s cue speed
BREAK_JITTER_DEG = 1.0    # uniform aim jitter around the apex ball


def apex_ball(state: TableState) -> str:
    """The rack ball closest to the cue ball (the head of the rack)."""
    cx, cy = state.cue
    return min(state.balls, key=lambda b: math.dist((cx, cy), state.balls[b]))


def break_stroke(state: TableState, rng: np.random.Generator) -> Stroke:
    bx, by = state.balls[apex_ball(state)]
    cx, cy = state.cue
    phi = math.degrees(math.atan2(by - cy, bx - cx))
    phi += float(rng.uniform(-BREAK_JITTER_DEG, BREAK_JITTER_DEG))
    return Stroke(phi=phi, V0=BREAK_SPEED, a=0.0, b=0.0, theta=0.0)
