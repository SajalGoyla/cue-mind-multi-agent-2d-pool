"""Structured-output schemas for every LLM agent (enforced by the model server and again in code).

Word limits and the shortlist length are stated in the prompts and enforced in code, not in
the JSON schema.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

IntentLit = Literal["offense", "position", "safety"]
AgentLit = Literal["offense", "position", "defense"]


class SingleDecision(BaseModel):
    """SA's answer."""

    shot_id: str
    intent: IntentLit
    rationale: str
    confidence: float


class ShortlistItem(BaseModel):
    shot_id: str
    reason: str


class SpecialistShortlist(BaseModel):
    """A specialist's ranked shortlist (best first)."""

    shortlist: list[ShortlistItem]


class CoordinatorDecision(BaseModel):
    """The MA coordinator's answer; the shot must come from the specialists' shortlists."""

    shot_id: str
    intent: IntentLit
    rationale: str
    confidence: float
    followed: list[AgentLit]


def clamp01(x: float) -> float:
    """Confidence in [0, 1]. Models sometimes answer on a 0-100 scale (seen with Qwen 2.5);
    values in (1, 100] are read as percentages."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.5
    if 1.0 < v <= 100.0:
        v /= 100.0
    return max(0.0, min(1.0, v))
