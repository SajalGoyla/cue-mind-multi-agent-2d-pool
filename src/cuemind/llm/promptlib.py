"""Prompt assembly. Prompt text lives in llm/prompts/*.md (versioned; hash logged per call).

Fairness (plan section 2.3, item 6): the single agent's system prompt contains the
union of the three specialists' strategy knowledge, verbatim. Tested in tests/unit.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from cuemind.candidates.describe import state_summary
from cuemind.candidates.model import Candidate
from cuemind.sim.state import TableState

PROMPT_DIR = Path(__file__).parent / "prompts"
PERSPECTIVES = ("offense", "position", "safety")


@lru_cache(maxsize=None)
def _read(name: str) -> str:
    return (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


def knowledge(perspective: str) -> str:
    return _read(f"knowledge_{perspective}")


def all_knowledge() -> str:
    return "\n\n".join(knowledge(p) for p in PERSPECTIVES)


def system_single() -> str:
    return "\n\n".join([_read("rules"), all_knowledge(), _read("role_single")])


def system_specialist(perspective: str) -> str:
    role = _read("role_specialist").replace("{perspective}", perspective)
    return "\n\n".join([_read("rules"), knowledge(perspective), role])


def system_coordinator() -> str:
    return "\n\n".join([_read("rules"), all_knowledge(), _read("role_coordinator")])


def system_critic() -> str:
    return "\n\n".join([_read("rules"), all_knowledge(), _read("role_critic")])


def prompt_version() -> str:
    """Hash of every prompt file; logged with each decision."""
    h = hashlib.sha256()
    for p in sorted(PROMPT_DIR.glob("*.md")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def candidate_table(cands: list[Candidate]) -> str:
    return "\n".join(f"{c.shot_id} | {c.description}" for c in cands)


def history_text(history: list[dict], shooter: str) -> str:
    if not history:
        return "No previous shots in this game (or this is the first decision)."
    lines = []
    for h in history:
        who = "You" if h["seat"] == shooter else "Opponent"
        lines.append(f"- {who}: {h['intent']} ({h['shot_id']}), result: {h['result']}")
    return "\n".join(lines)


def decision_context_text(state: TableState, cands: list[Candidate], history: list[dict],
                          ball_in_hand: bool = False) -> str:
    parts = ["## Position", state_summary(state)]
    if ball_in_hand:
        parts.append("You had ball in hand; the cue ball has already been placed.")
    parts += ["## Recent shots", history_text(history, state.shooter),
              "## Candidate shots", candidate_table(cands)]
    return "\n".join(parts)


def user_single(ctx_text: str) -> str:
    return ctx_text + "\n\nChoose one shot id."


def user_specialist(ctx_text: str, perspective: str) -> str:
    return ctx_text + f"\n\nAssess every candidate from the {perspective} perspective only."


def user_coordinator(ctx_text: str, reports: dict[str, dict]) -> str:
    rep = "\n".join(f"### {name} specialist\n{json.dumps(r, ensure_ascii=False)}"
                    for name, r in reports.items())
    return f"{ctx_text}\n\n## Specialist reports\n{rep}\n\nChoose one shot id."


def user_critic(ctx_text: str, decision: dict) -> str:
    return (f"{ctx_text}\n\n## Chosen shot\n{json.dumps(decision, ensure_ascii=False)}\n\n"
            "Approve or reject this choice.")
