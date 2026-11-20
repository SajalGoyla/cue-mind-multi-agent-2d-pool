"""Structured-output schemas for every LLM agent (validated by the SDK and again in code).

Numeric ranges and word limits are stated in the prompts and enforced in code, not in the
JSON schema, to keep the schemas within what structured outputs support.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

IntentLit = Literal["offense", "position", "safety"]
AgentLit = Literal["offense", "position", "safety"]


class SingleDecision(BaseModel):
    shot_id: str
    intent: IntentLit
    rationale: str
    confidence: float


class Assessment(BaseModel):
    shot_id: str
    score: int
    note: str


class SpecialistReport(BaseModel):
    assessments: list[Assessment]
    top_choice: str
    summary: str


class CoordinatorDecision(BaseModel):
    shot_id: str
    intent: IntentLit
    rationale: str
    confidence: float
    overruled: list[AgentLit]


class CriticVerdict(BaseModel):
    approve: bool
    reason: str
    alternative_shot_id: str | None


def clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.5
