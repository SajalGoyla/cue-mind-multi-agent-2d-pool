"""Structured-output schemas (v2, D-029). Validated by constrained decoding and again in code
(shot ids must be in the menu; shortlist length and word limits are enforced in code)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

AgentLit = Literal["offense", "position", "safety"]


class SingleDecision(BaseModel):
    shot_id: str
    rationale: str


class Pick(BaseModel):
    shot_id: str
    reason: str


class Shortlist(BaseModel):
    picks: list[Pick]  # ranked, best first; k = 3 requested


class CoordinatorDecision(BaseModel):
    shot_id: str
    rationale: str
    followed: list[AgentLit]
