"""Prompt assembly. Prompt text lives in llm/prompts/*.md (versioned; hash logged per call).

Fairness: the single agent's system prompt contains the union of the three specialists'
strategy knowledge, verbatim (tested in tests/unit/test_llm_agents.py).
"""

from __future__ import annotations

import hashlib
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


# The safety specialist is presented to the model as the "risk" specialist: a 3B model told it
# is the "safety" specialist nominated safety shots first in ~90 % of dev decisions (D-030).
DISPLAY = {"offense": "offense", "position": "position", "safety": "risk"}
RISK_RULE = ("\n\nAs the risk specialist, judge which shots are least likely to foul, scratch or hand "
             "the opponent an easy shot. A pot with easy or moderate geometry is usually the lowest-risk "
             "shot; nominate a safety first only if every pot on the list has hard geometry.")


def system_specialist(perspective: str) -> str:
    role = _read("role_specialist").replace("{perspective}", DISPLAY[perspective])
    if perspective == "safety":
        role += RISK_RULE
    return "\n\n".join([_read("rules"), knowledge(perspective), role])


def system_coordinator() -> str:
    return "\n\n".join([_read("rules"), all_knowledge(), _read("role_coordinator")])


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
    return ctx_text + f"\n\nNominate your 3 best shots from the {DISPLAY[perspective]} perspective."


def user_coordinator(ctx_text: str, shortlists: dict[str, list[dict]],
                     nominated: list[str], descriptions: dict[str, str] | None = None) -> str:
    """The coordinator sees the position, the full candidate table and every specialist's
    ranked nominations, each repeated with its full shot line (a small model otherwise mixes
    up which id is which shot) and the specialist's reason."""
    desc = descriptions or {}
    parts = []
    for name, picks in shortlists.items():
        lines = "\n".join(f"  {i + 1}. {p['shot_id']} ({desc.get(p['shot_id'], '?')}): {p['reason']}"
                          for i, p in enumerate(picks))
        parts.append(f"### {name} specialist\n{lines or '  (no valid nominations)'}")
    return (f"{ctx_text}\n\n## Specialist nominations\n" + "\n".join(parts)
            + f"\n\nChoose one of the nominated shots: {', '.join(nominated)}.")
