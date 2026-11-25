"""Decision preparation (v2, D-029): build the shot menu that every player receives.

Every policy in a comparison receives the output of the *same* `prepare` call for a given
state and decision key (fairness contract: identical menus). The forecasts used to select
the menu stay attached to the candidates for the reference player (Stat); the LLM systems
only read the geometry-only `description` of each candidate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from cuemind.candidates.menu_v2 import build_menu_v2
from cuemind.candidates.model import Candidate
from cuemind.sim.state import TableState


@dataclass(frozen=True)
class PrepConfig:
    n_forecast: int = 3          # noisy simulations per variant, averaged (Stat's forecast)
    forecast_noise: str = "low"  # noise preset used for forecasting (not for execution)
    max_pots: int = 12
    max_safeties: int = 3

    def to_dict(self) -> dict:
        return asdict(self)


def prepare(state: TableState, cfg: PrepConfig, rollout_key: tuple = ()) -> list[Candidate]:
    """Build the decision menu for the shooter in `state` (cue ball already placed)."""
    return build_menu_v2(state, n_forecast=cfg.n_forecast, noise=cfg.forecast_noise,
                         max_pots=cfg.max_pots, max_safeties=cfg.max_safeties, key=rollout_key)
