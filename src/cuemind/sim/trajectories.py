"""Ball trajectories of one stroke, resampled to common animation frames (for the UI)."""

from __future__ import annotations

import numpy as np
import pooltool.constants as const

from cuemind.sim.execute import Stroke, simulate
from cuemind.sim.state import TableState, to_system


def trajectories(state: TableState, stroke: Stroke, fps: int = 25,
                 max_seconds: float = 12.0) -> dict:
    """Return {"t": [...], "balls": {id: {"x": [...], "y": [...], "pocketed_at": t|None}}}.

    Positions are in the pooltool frame (metres). Pocketed balls are hidden (NaN) from
    the frame in which they drop.
    """
    shot = simulate(to_system(state), stroke, continuous=True, dt=1.0 / fps)
    t_end = min(max(float(shot.t or 0.0), 0.04), max_seconds)
    frames = np.arange(0.0, t_end + 1e-9, 1.0 / fps)
    out = {"t": frames.round(3).tolist(), "balls": {}}
    for bid, ball in shot.balls.items():
        if ball.history_cts.empty:
            xy = ball.state.rvw[0][:2]
            out["balls"][bid] = {"x": [float(xy[0])] * len(frames),
                                 "y": [float(xy[1])] * len(frames), "pocketed_at": None}
            continue
        rvw, s, ts = ball.history_cts.vectorize()
        xs = np.interp(frames, ts, rvw[:, 0, 0])
        ys = np.interp(frames, ts, rvw[:, 0, 1])
        pocketed_at = None
        idx = np.where(s == const.pocketed)[0]
        if idx.size:
            pocketed_at = float(ts[idx[0]])
            xs = np.where(frames >= pocketed_at, np.nan, xs)
            ys = np.where(frames >= pocketed_at, np.nan, ys)
        out["balls"][bid] = {"x": xs.round(4).tolist(), "y": ys.round(4).tolist(),
                             "pocketed_at": pocketed_at}
    return out
