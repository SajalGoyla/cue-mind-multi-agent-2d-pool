"""Policy interface: every decision maker (bot or LLM system) implements this."""

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
    candidates: list[Candidate]                         # the raw menu (identical for every policy)
    history: list[dict] = field(default_factory=list)  # last few shots, most recent last
    game_id: str = ""
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    ball_in_hand: bool = False  # the cue ball was just placed after an opponent's foul
    request_tag: str = ""       # stable id of this decision (logging only)
    sample_idx: int = 0         # >0 draws an independent LLM sample
    noise: str = "none"         # execution noise of this game (B simulates with it)
    sim_key: tuple = ()         # seeds B's private simulations (game seed, shot index)

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
    # tokens, latency. B puts its private simulation scores here.
    trace: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, float] = field(default_factory=dict)  # calls, tokens, latency_s


class Policy(Protocol):
    name: str

    def decide(self, ctx: DecisionContext) -> Decision: ...


def fallback_choice(cands: list[Candidate]) -> Candidate:
    """The shared fallback rule: pots before safeties, then the highest geometric ease,
    then the lowest id. Uses only what every policy can see."""
    return min(cands, key=lambda c: (c.shot_type != "pot", -c.geometry.ease, c.shot_id))


def intent_of(c: Candidate) -> Intent:
    return "offense" if c.shot_type == "pot" else "safety"
