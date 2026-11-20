"""Shot-execution noise (imperfect stroke) with common-random-number seeding.

The same (game seed, shot index, purpose) always yields the same random stream, so two
systems that choose the same shot at the same point of a paired game get the same
execution error. This removes luck as a source of difference between systems.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from cuemind.sim.execute import Stroke


@dataclass(frozen=True)
class NoiseModel:
    phi_sd_deg: float = 0.0   # aim error (degrees)
    v0_rel_sd: float = 0.0    # relative speed error
    tip_sd: float = 0.0       # tip-offset error (units of ball radius)
    name: str = "custom"

    @property
    def is_zero(self) -> bool:
        return self.phi_sd_deg == 0 and self.v0_rel_sd == 0 and self.tip_sd == 0

    def apply(self, stroke: Stroke, rng: np.random.Generator) -> Stroke:
        """Return the stroke actually executed. Always draws 4 numbers so streams stay aligned."""
        z = rng.standard_normal(4)
        return Stroke(
            phi=stroke.phi + self.phi_sd_deg * z[0],
            V0=stroke.V0 * (1.0 + self.v0_rel_sd * z[1]),
            a=stroke.a + self.tip_sd * z[2],
            b=stroke.b + self.tip_sd * z[3],
            theta=stroke.theta,
        ).clipped()


NOISE_PRESETS: dict[str, NoiseModel] = {
    "none": NoiseModel(0.0, 0.0, 0.0, "none"),
    "low": NoiseModel(0.25, 0.03, 0.01, "low"),
    "medium": NoiseModel(0.5, 0.05, 0.02, "medium"),
    "high": NoiseModel(1.0, 0.10, 0.04, "high"),
}


def get_noise(name_or_model: str | NoiseModel) -> NoiseModel:
    if isinstance(name_or_model, NoiseModel):
        return name_or_model
    try:
        return NOISE_PRESETS[name_or_model]
    except KeyError:
        raise ValueError(f"unknown noise preset {name_or_model!r}; "
                         f"choose from {list(NOISE_PRESETS)}") from None


def rng_for(*keys: object) -> np.random.Generator:
    """Deterministic generator from arbitrary keys, e.g. rng_for(seed, shot_idx, "execute")."""
    digest = hashlib.sha256("|".join(map(str, keys)).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))
