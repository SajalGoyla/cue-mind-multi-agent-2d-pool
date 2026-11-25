"""Prompt assembly. Prompt text lives in llm/prompts/*.md (versioned; hash logged per call).

Who gets what (fairness: SA's system prompt holds every strategy block a specialist has):

    SA            rules + offense + position + defense knowledge + role_single
    specialist X  rules + X knowledge + role_specialist            (sees the full raw menu)
    coordinator   rules + all three knowledge blocks + role_coordinator
                  (sees the position and only the shortlisted shots with the specialists' reasons)
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from cuemind.candidates.describe import state_summary
from cuemind.candidates.model import Candidate
from cuemind.sim.state import TableState

PROMPT_DIR = Path(__file__).parent / "prompts"
PERSPECTIVES = ("offense", "position", "defense")
SHORTLIST_K = 3


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
    role = _read("role_specialist").replace("{perspective}", perspective).replace("{k}", str(SHORTLIST_K))
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


def position_text(state: TableState, history: list[dict], ball_in_hand: bool = False) -> str:
    parts = ["## Position", state_summary(state)]
    if ball_in_hand:
        parts.append("You had ball in hand; the cue ball has already been placed.")
    parts += ["## Recent shots", history_text(history, state.shooter)]
    return "\n".join(parts)


def decision_context_text(state: TableState, cands: list[Candidate], history: list[dict],
                          ball_in_hand: bool = False) -> str:
    return (position_text(state, history, ball_in_hand)
            + "\n## Candidate shots\n" + candidate_table(cands))


def user_single(ctx_text: str) -> str:
    return ctx_text + "\n\nChoose one shot id."


def user_specialist(ctx_text: str, perspective: str) -> str:
    return (ctx_text + f"\n\nFrom the {perspective} perspective only, shortlist up to "
            f"{SHORTLIST_K} shot ids, best first.")


def user_coordinator(pos_text: str, shortlisted: list[Candidate], reports: dict[str, dict]) -> str:
    lines = [pos_text, "## Shortlisted shots", candidate_table(shortlisted),
             "## Specialist shortlists (best first)"]
    for name, rep in reports.items():
        lines.append(f"### {name} specialist")
        if "shortlist" not in rep:
            lines.append(f"(no report: {rep.get('error', 'failed')})")
            continue
        for rank, item in enumerate(rep["shortlist"], 1):
            lines.append(f"{rank}. {item['shot_id']}: {item['reason']}")
    lines.append("\nChoose one shot id from the shortlisted shots.")
    return "\n".join(lines)
