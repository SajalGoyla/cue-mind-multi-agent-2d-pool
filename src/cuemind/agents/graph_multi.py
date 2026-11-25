"""MA v2 decision graph (LangGraph, D-029).

    START ──► offense ──┐
          ├─► position ─┼─► join ──(all three rank the same shot first)──► END
          └─► safety ───┘      └──(otherwise)──► coordinator ──► END

Each specialist sees the rules, its own strategy block and the full candidate table, and
returns a ranked shortlist of k = 3 shot ids with one short reason each. The coordinator
sees the rules, all three strategy blocks, the candidate table and the nominations, and must
choose from the union of the shortlists.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from cuemind.agents.common import call_with_check
from cuemind.llm import promptlib as P
from cuemind.llm.client import LLMClient
from cuemind.llm.schemas import CoordinatorDecision, Shortlist

K = 3


class MAState(TypedDict, total=False):
    ctx_text: str
    valid_ids: list[str]
    sample_idx: int
    shortlists: Annotated[dict[str, list[dict]], operator.or_]
    attempts: Annotated[list[dict], operator.add]
    decision: dict | None
    unanimous: bool


def clean_picks(sl: Shortlist, valid: set[str]) -> list[dict]:
    """Valid, de-duplicated picks, best first, at most K; reasons trimmed to 20 words."""
    out, seen = [], set()
    for p in sl.picks:
        if p.shot_id in valid and p.shot_id not in seen:
            out.append({"shot_id": p.shot_id, "reason": " ".join(p.reason.split()[:20])})
            seen.add(p.shot_id)
        if len(out) == K:
            break
    return out


def _specialist_node(client: LLMClient, perspective: str):
    system = P.system_specialist(perspective)

    def node(state: MAState) -> dict[str, Any]:
        valid = set(state["valid_ids"])
        res, att = call_with_check(client, system=system,
                                   user=P.user_specialist(state["ctx_text"], perspective),
                                   schema=Shortlist, agent=perspective, valid_ids=valid,
                                   is_valid=lambda s: bool(clean_picks(s, valid)),
                                   repeat_idx=state.get("sample_idx", 0))
        picks = clean_picks(res.parsed, valid) if res.parsed is not None else []
        return {"shortlists": {perspective: picks}, "attempts": att}

    return node


def _join(state: MAState) -> dict[str, Any]:
    firsts = [sl[0]["shot_id"] for sl in state.get("shortlists", {}).values() if sl]
    if len(firsts) == len(P.PERSPECTIVES) and len(set(firsts)) == 1:
        return {"unanimous": True,
                "decision": {"shot_id": firsts[0], "rationale": "all three specialists ranked "
                             "this shot first", "followed": list(P.PERSPECTIVES)}}
    return {"unanimous": False}


def _route(state: MAState) -> str:
    return END if state.get("unanimous") else "coordinator"


def _coordinator_node(client: LLMClient):
    system = P.system_coordinator()

    def node(state: MAState) -> dict[str, Any]:
        shortlists = state.get("shortlists", {})
        nominated = sorted({p["shot_id"] for sl in shortlists.values() for p in sl})
        if not nominated:  # every specialist failed: the policy applies the fallback rule
            return {"decision": None}
        allowed = set(nominated)
        res, att = call_with_check(client, system=system,
                                   user=P.user_coordinator(state["ctx_text"], shortlists, nominated),
                                   schema=CoordinatorDecision, agent="coordinator",
                                   valid_ids=allowed,
                                   is_valid=lambda d: d.shot_id in allowed,
                                   repeat_idx=state.get("sample_idx", 0))
        dec = None
        if res.parsed is not None and res.parsed.shot_id in allowed:
            d = res.parsed
            followed = [n for n, sl in shortlists.items() if any(p["shot_id"] == d.shot_id for p in sl)]
            dec = {"shot_id": d.shot_id, "rationale": d.rationale, "followed": followed,
                   "claimed_followed": list(d.followed)}
        return {"decision": dec, "attempts": att}

    return node


def build_graph(client: LLMClient):
    g = StateGraph(MAState)
    for s in P.PERSPECTIVES:
        g.add_node(s, _specialist_node(client, s))
        g.add_edge(START, s)
    g.add_node("join", _join)
    g.add_edge(list(P.PERSPECTIVES), "join")
    g.add_node("coordinator", _coordinator_node(client))
    g.add_conditional_edges("join", _route, ["coordinator", END])
    g.add_edge("coordinator", END)
    return g.compile()
