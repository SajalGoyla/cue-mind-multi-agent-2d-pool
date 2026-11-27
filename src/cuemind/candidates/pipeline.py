"""Decision preparation: build the shot menu that every player receives (D-029, D-031).

Two menu rules and three information levels, combined per experiment condition:

| condition                      | menu        | info       | reference player |
|--------------------------------|-------------|------------|------------------|
| v2 (forecasts for Stat only)   | "forecast"  | "geometry" | Stat             |
| v3-A (no forecasts for anyone) | "geometric" | "hints"    | Rule             |
| v3-B (forecasts for everyone)  | "forecast"  | "forecasts"| Stat             |

`menu="forecast"`: every variant forecast 3x (noisy) and the top 15 by Stat's objective kept
(`menu_v2`). `menu="geometric"`: selected by geometry and simulation-free physics hints
(`menu_geo`). `info` sets what the LLMs read in each candidate line: geometry only,
geometry + physics hint, or geometry + forecast statistics. Every player in a comparison
receives the identical menu for the same state and decision key.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from cuemind.candidates.describe import describe
from cuemind.candidates.hints import Hint, hint_text
from cuemind.candidates.menu_geo import build_menu_geo
from cuemind.candidates.menu_v2 import build_menu_v2
from cuemind.candidates.model import Candidate
from cuemind.sim.state import TableState

MENUS = ("forecast", "geometric")
INFOS = ("geometry", "hints", "forecasts")


@dataclass(frozen=True)
class PrepConfig:
    n_forecast: int = 3          # noisy simulations per variant (forecast menu only)
    forecast_noise: str = "low"  # noise preset used for forecasting (not for execution)
    max_pots: int = 12
    max_safeties: int = 3
    menu: str = "forecast"
    info: str = "geometry"

    def __post_init__(self):
        if self.menu not in MENUS or self.info not in INFOS:
            raise ValueError(f"menu must be one of {MENUS}, info one of {INFOS}")
        if self.info == "forecasts" and self.menu != "forecast":
            raise ValueError("info='forecasts' needs menu='forecast'")
        if self.info == "hints" and self.menu != "geometric":
            raise ValueError("info='hints' needs menu='geometric'")

    def to_dict(self) -> dict:
        return asdict(self)


def describe_for(c: Candidate, info: str) -> str:
    if info == "forecasts":
        return describe(c, "L2")
    text = describe(c, "L0")
    if info == "hints" and c.hint is not None:
        text = text.rstrip(".") + "; " + hint_text(Hint(**c.hint)) + "."
    return text


def prepare(state: TableState, cfg: PrepConfig, rollout_key: tuple = ()) -> list[Candidate]:
    """Build the decision menu for the shooter in `state` (cue ball already placed)."""
    if cfg.menu == "geometric":
        menu = build_menu_geo(state, max_pots=cfg.max_pots, max_safeties=cfg.max_safeties,
                              key=rollout_key)
    else:
        menu = build_menu_v2(state, n_forecast=cfg.n_forecast, noise=cfg.forecast_noise,
                             max_pots=cfg.max_pots, max_safeties=cfg.max_safeties, key=rollout_key)
    return [c.model_copy(update={"description": describe_for(c, cfg.info)}) for c in menu]
