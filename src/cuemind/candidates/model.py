"""Candidate-shot data model shared by the generator, annotator, policies, logs and UI."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel

from cuemind.sim.execute import Stroke

ShotType = Literal["pot", "safety", "contact_only"]
PowerLabel = Literal["soft", "medium", "firm"]
SpinLabel = Literal["draw", "stun", "follow"]

POWERS: dict[str, float] = {"soft": 1.5, "medium": 2.5, "firm": 4.0}     # cue speed V0, m/s
SPINS: dict[str, float] = {"draw": -0.3, "stun": 0.0, "follow": 0.3}     # tip offset b, x R


class Geometry(BaseModel):
    cut_deg: float = 0.0
    cue_to_ball_m: float = 0.0
    ball_to_pocket_m: float | None = None
    required_precision_deg: float | None = None
    ease: float = 0.0  # geometric ease in [0, 1], higher = easier


class Preview(BaseModel):
    """Access level L1: one noise-free simulation."""

    pots_called: bool
    scratch: bool
    legal: bool
    foul: str | None = None
    cue_end: tuple[float, float] | None = None
    next_viable_pots: int = 0


class Rollouts(BaseModel):
    """Access level L2: Monte-Carlo statistics over noisy executions."""

    n: int
    p_pot: float              # called ball in called pocket, legal shot
    p_scratch: float
    p_foul: float
    p_keep_turn: float
    cue_end_mean: tuple[float, float] | None = None
    cue_end_sd_m: float | None = None
    next_ease: float = 0.0    # ease of my best next pot from where the cue ball stops (mean)
    opp_threat: float = 0.0   # ease of the opponent's best pot if the turn passes (mean)


class Candidate(BaseModel):
    shot_id: str = ""
    shot_type: ShotType
    target_ball: str
    pocket: str | None = None
    power_label: PowerLabel
    spin_label: SpinLabel = "stun"
    phi: float
    V0: float
    a: float = 0.0
    b: float = 0.0
    theta: float = 0.0
    geometry: Geometry = Geometry()
    preview: Preview | None = None
    rollouts: Rollouts | None = None
    hint: dict | None = None  # v3 condition A: simulation-free physics hint (candidates/hints.py)
    description: str = ""

    @property
    def stroke(self) -> Stroke:
        return Stroke(phi=self.phi, V0=self.V0, a=self.a, b=self.b, theta=self.theta)

    @property
    def key(self) -> tuple:
        return (self.shot_type, self.target_ball, self.pocket or "", self.power_label,
                self.spin_label, round(self.phi, 3))


def candidates_hash(cands: list[Candidate]) -> str:
    """Hash of the menu as an LLM system would see it (ids, shots and annotations)."""
    blob = json.dumps([c.model_dump(exclude={"description"}) for c in cands], sort_keys=True)
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()[:16]
