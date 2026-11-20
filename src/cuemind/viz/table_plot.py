"""Top-down 2D table rendering with Plotly (used by the demo and for report figures).

Coordinates are drawn in the pooltool frame (metres); axis ticks show diamonds so the
pictures match the coordinates used in the LLM prompts.
"""

from __future__ import annotations

import math

import plotly.graph_objects as go

from cuemind.candidates.model import Candidate
from cuemind.sim.state import BALL_R, TABLE, TABLE_L, TABLE_W, TableState

BALL_COLORS = {1: "#f2c200", 2: "#1f4fd1", 3: "#d62020", 4: "#6a2fa0", 5: "#f07c00",
               6: "#138a36", 7: "#7a1f1f", 8: "#111111"}
CLOTH, RAIL = "#1d6b3a", "#5b3a1e"
PAD = 0.07


def ball_style(bid: str) -> tuple[str, str, float]:
    """(fill colour, outline colour, outline width); stripes get a thick white outline."""
    if bid == "cue":
        return "#f8f8f2", "#999999", 1.0
    n = int(bid)
    if n <= 8:
        return BALL_COLORS[n], "#222222", 1.0
    return BALL_COLORS[n - 8], "#ffffff", 4.0


def _base(height: int) -> go.Figure:
    fig = go.Figure()
    fig.add_shape(type="rect", x0=-PAD, y0=-PAD, x1=TABLE_W + PAD, y1=TABLE_L + PAD,
                  fillcolor=RAIL, line_width=0, layer="below")
    fig.add_shape(type="rect", x0=0, y0=0, x1=TABLE_W, y1=TABLE_L, fillcolor=CLOTH,
                  line=dict(color="#0f3d20", width=2), layer="below")
    for p in TABLE.pockets.values():
        cx, cy, r = float(p.center[0]), float(p.center[1]), float(p.radius)
        fig.add_shape(type="circle", x0=cx - r, y0=cy - r, x1=cx + r, y1=cy + r,
                      fillcolor="#000000", line_width=0, layer="below")
    for i in range(1, 4):  # diamonds on the short rails
        for y in (-PAD / 2, TABLE_L + PAD / 2):
            fig.add_shape(type="circle", x0=TABLE_W * i / 4 - 0.006, x1=TABLE_W * i / 4 + 0.006,
                          y0=y - 0.006, y1=y + 0.006, fillcolor="#e8e0c8", line_width=0)
    for j in range(1, 8):  # diamonds on the long rails
        for x in (-PAD / 2, TABLE_W + PAD / 2):
            fig.add_shape(type="circle", x0=x - 0.006, x1=x + 0.006, y0=TABLE_L * j / 8 - 0.006,
                          y1=TABLE_L * j / 8 + 0.006, fillcolor="#e8e0c8", line_width=0)
    fig.update_xaxes(range=[-PAD - 0.01, TABLE_W + PAD + 0.01], showgrid=False, zeroline=False,
                     tickvals=[TABLE_W * i / 4 for i in range(5)], ticktext=[str(i) for i in range(5)],
                     title_text="diamonds")
    fig.update_yaxes(range=[-PAD - 0.01, TABLE_L + PAD + 0.01], showgrid=False, zeroline=False,
                     scaleanchor="x", scaleratio=1,
                     tickvals=[TABLE_L * j / 8 for j in range(9)], ticktext=[str(j) for j in range(9)])
    fig.update_layout(height=height, width=int(height * 0.62), margin=dict(l=30, r=10, t=30, b=30),
                      plot_bgcolor="white", showlegend=False)
    return fig


def _balls_trace(xy: dict[str, tuple[float, float]]) -> go.Scatter:
    ids = list(xy)
    fills, lines, widths = zip(*(ball_style(b) for b in ids)) if ids else ((), (), ())
    return go.Scatter(
        x=[xy[b][0] for b in ids], y=[xy[b][1] for b in ids], mode="markers+text",
        text=["" if b == "cue" else b for b in ids], textfont=dict(color="white", size=9),
        marker=dict(size=17, color=list(fills), line=dict(color=list(lines), width=list(widths))),
        hovertext=ids, hoverinfo="text")


def table_figure(state: TableState, candidates: list[Candidate] | None = None,
                 highlight: str | None = None, title: str = "", height: int = 640) -> go.Figure:
    """Static table with optional candidate aim lines (highlighted one drawn bold)."""
    fig = _base(height)
    if candidates and state.cue is not None:
        for c in candidates:
            hot = c.shot_id == highlight
            if not hot and highlight is not None and len(candidates) > 6:
                continue
            ux, uy = math.cos(math.radians(c.phi)), math.sin(math.radians(c.phi))
            length = (c.geometry.cue_to_ball_m or 0.4) + BALL_R
            x1, y1 = state.cue[0] + ux * length, state.cue[1] + uy * length
            color = "#ffe066" if hot else ("#9ad1ff" if c.shot_type == "pot" else "#ffb3b3")
            fig.add_trace(go.Scatter(x=[state.cue[0], x1], y=[state.cue[1], y1], mode="lines",
                                     line=dict(color=color, width=4 if hot else 1.2,
                                               dash="solid" if hot else "dot"),
                                     hovertext=f"{c.shot_id}: {c.description}", hoverinfo="text"))
            if hot and c.pocket and c.target_ball in state.balls:
                p = TABLE.pockets[c.pocket].center
                bx, by = state.balls[c.target_ball]
                fig.add_trace(go.Scatter(x=[bx, float(p[0])], y=[by, float(p[1])], mode="lines",
                                         line=dict(color="#ffe066", width=2, dash="dash"),
                                         hoverinfo="skip"))
    xy = dict(state.balls)
    if state.cue is not None:
        xy["cue"] = state.cue
    fig.add_trace(_balls_trace(xy))
    if title:
        fig.update_layout(title=dict(text=title, x=0.5, font=dict(size=13)))
    return fig


def animation_figure(traj: dict, title: str = "", height: int = 640, max_frames: int = 160) -> go.Figure:
    """Animated replay of one stroke from `sim.trajectories.trajectories` output."""
    fig = _base(height)
    t = traj["t"]
    step = max(1, len(t) // max_frames)
    idx = list(range(0, len(t), step))
    if idx[-1] != len(t) - 1:
        idx.append(len(t) - 1)
    ids = list(traj["balls"])

    def frame_xy(i):
        return {b: (traj["balls"][b]["x"][i], traj["balls"][b]["y"][i]) for b in ids}

    fig.add_trace(_balls_trace(frame_xy(0)))
    fig.frames = [go.Frame(data=[_balls_trace(frame_xy(i))], name=str(k)) for k, i in enumerate(idx)]
    dur = int(1000 * (t[1] - t[0]) * step) if len(t) > 1 else 40
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=13)),
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=-0.04, xanchor="left",
                          buttons=[dict(label="Play", method="animate",
                                        args=[None, {"frame": {"duration": dur, "redraw": True},
                                                     "fromcurrent": True, "transition": {"duration": 0}}])])])
    return fig
