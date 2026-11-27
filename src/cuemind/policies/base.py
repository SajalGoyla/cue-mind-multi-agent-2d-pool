"""Policy interface: every decision maker (baseline bot or LLM system) implements this."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np

from cuemind.candidates.model import Candidate
from cuemind.sim.state import TableState

Intent = Literal["offense", "position", "safety"]


@dataclass
class DecisionContext:
    state: TableState
    candidates: list[Candidate]
    history: list[dict] = field(default_factory=list)  # last few shots, most recent last
    game_id: str = ""
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    ball_in_hand: bool = False  # the cue ball was just placed after an opponent's foul
    request_tag: str = ""       # stable id of this decision (logging only)
    sample_idx: int = 0         # >0 draws an independent LLM sample (decision-variance runs)
    info: str = "geometry"      # what the candidate lines contain: geometry | hints | forecasts

    def by_id(self, shot_id: str) -> Candidate | None:
        return next((c for c in self.candidates if c.shot_id == shot_id), None)


@dataclass
class Decision:
    shot_id: str
    intent: Intent = "offense"
    rationale: str = ""
    confidence: float | None = None
    fallback: bool = False
    # Everything needed to explain the decision afterwards: per-agent prompts/outputs,
    # tokens, latency. Baselines put their scores here.
    trace: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, float] = field(default_factory=dict)  # calls, tokens, cost_usd, latency_s


class Policy(Protocol):
    name: str

    def decide(self, ctx: DecisionContext) -> Decision: ...


def fallback_choice(cands: list[Candidate]) -> Candidate:
    """The shared fallback rule for LLM systems (fairness contract), geometry only (D-029):
    pots before safeties, highest geometric ease, medium stun preferred, then lowest id."""
    def key(c: Candidate):
        plain = (c.power_label, c.spin_label) == ("medium", "stun")
        return (c.shot_type != "pot", -c.geometry.ease, not plain, c.shot_id)
    return min(cands, key=key)


def intent_of(c: Candidate) -> Intent:
    return "offense" if c.shot_type == "pot" else "safety"
