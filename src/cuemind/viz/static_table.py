"""Static matplotlib table drawings (PNG) for reports, slides and hard-case review."""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from cuemind.candidates.model import Candidate  # noqa: E402
from cuemind.sim.state import BALL_R, TABLE, TABLE_L, TABLE_W, TableState  # noqa: E402
from cuemind.viz.table_plot import CLOTH, PAD, RAIL, ball_style  # noqa: E402


def draw_table(state: TableState, path: str, title: str = "",
               candidates: list[Candidate] | None = None, highlight: str | None = None) -> str:
    fig, ax = plt.subplots(figsize=(3.6, 6.6))
    ax.add_patch(Rectangle((-PAD, -PAD), TABLE_W + 2 * PAD, TABLE_L + 2 * PAD, color=RAIL))
    ax.add_patch(Rectangle((0, 0), TABLE_W, TABLE_L, color=CLOTH))
    for p in TABLE.pockets.values():
        ax.add_patch(Circle((float(p.center[0]), float(p.center[1])), float(p.radius), color="black"))
    if candidates and state.cue is not None:
        for c in candidates:
            hot = c.shot_id == highlight
            ux, uy = math.cos(math.radians(c.phi)), math.sin(math.radians(c.phi))
            L = (c.geometry.cue_to_ball_m or 0.4) + BALL_R
            ax.plot([state.cue[0], state.cue[0] + ux * L], [state.cue[1], state.cue[1] + uy * L],
                    color="#ffe066" if hot else "#cfe8ff", lw=2.5 if hot else 0.8,
                    ls="-" if hot else ":", zorder=2)
    pts = dict(state.balls)
    if state.cue is not None:
        pts["cue"] = state.cue
    for bid, (x, y) in pts.items():
        fill, edge, w = ball_style(bid)
        ax.add_patch(Circle((x, y), BALL_R, facecolor=fill, edgecolor=edge, linewidth=min(w, 2.5), zorder=3))
        if bid != "cue":
            ax.text(x, y, bid, ha="center", va="center", fontsize=6, color="white", zorder=4)
    ax.set_xlim(-PAD, TABLE_W + PAD)
    ax.set_ylim(-PAD, TABLE_L + PAD)
    ax.set_aspect("equal")
    ax.set_xticks([TABLE_W * i / 4 for i in range(5)], [str(i) for i in range(5)], fontsize=7)
    ax.set_yticks([TABLE_L * j / 8 for j in range(9)], [str(j) for j in range(9)], fontsize=7)
    if title:
        ax.set_title(title, fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
