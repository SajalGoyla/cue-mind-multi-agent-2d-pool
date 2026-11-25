"""Multi-agent decision graph, v2 (LangGraph, D-029).

    START ──► offense ──┐
          ├─► position ─┼─► merge ──► (all three rank the same shot first) ──► END
          └─► defense ──┘         └─► otherwise ──► coordinator ──► END

1. The three specialists run in parallel (one LangGraph superstep). Each sees the full raw
   menu and returns a ranked shortlist of up to k shot ids with short reasons.
2. `merge` builds the union of the shortlists (rank-1 picks first). If every specialist put
   the same shot first, that shot is played without a coordinator call.
3. Otherwise the coordinator sees the position and only the shortlisted shots, with each
   specialist's ranking and reasons, and must choose one of them. If every specialist failed,
   it chooses from the full menu instead (logged).
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from cuemind.agents.common import call_with_id_check
from cuemind.candidates.model import Candidate
from cuemind.llm import promptlib as P
from cuemind.llm.client import LLMClient
from cuemind.llm.schemas import CoordinatorDecision, SpecialistShortlist


class MAState(TypedDict, total=False):
    ctx_text: str                 # position + full raw menu (what specialists read)
    pos_text: str                 # position only (the coordinator gets shots separately)
    menu: dict[str, dict]         # shot_id -> candidate (model_dump), in menu order
    sample_idx: int
    reports: Annotated[dict[str, dict], operator.or_]
    attempts: Annotated[list[dict], operator.add]
    shortlisted: list[str]
    decision: dict | None


def _shortlist_ids(parsed: SpecialistShortlist) -> list[str]:
    return [s.shot_id for s in parsed.shortlist]


def _specialist_node(client: LLMClient, perspective: str):
    system = P.system_specialist(perspective)

    def node(state: MAState) -> dict[str, Any]:
        valid = set(state["menu"])
        res, att = call_with_id_check(client, system=system,
                                      user=P.user_specialist(state["ctx_text"], perspective),
                                      schema=SpecialistShortlist, agent=perspective,
                                      valid_ids=valid, ids_of=_shortlist_ids,
                                      repeat_idx=state.get("sample_idx", 0))
        items: list[dict] = []
        if res.parsed is not None:
            for s in res.parsed.shortlist:  # keep valid, unique ids, best first, at most k
                if s.shot_id in valid and s.shot_id not in {i["shot_id"] for i in items}:
                    items.append({"shot_id": s.shot_id, "reason": s.reason})
        items = items[: P.SHORTLIST_K]
        report = {"shortlist": items} if items else {"error": res.error or "no valid shot ids"}
        return {"reports": {perspective: report}, "attempts": att}

    return node


def _merge_node(specialists: tuple[str, ...]):
    def node(state: MAState) -> dict[str, Any]:
        reports = state.get("reports", {})
        lists = [reports[s]["shortlist"] for s in specialists if "shortlist" in reports.get(s, {})]
        union: list[str] = []
        for rank in range(P.SHORTLIST_K):
            for lst in lists:
                if rank < len(lst) and lst[rank]["shot_id"] not in union:
                    union.append(lst[rank]["shot_id"])
        firsts = {lst[0]["shot_id"] for lst in lists}
        if len(lists) == len(specialists) and len(firsts) == 1:
            sid = firsts.pop()
            return {"shortlisted": union, "decision": {
                "shot_id": sid, "intent": None, "confidence": None, "followed": list(specialists),
                "rationale": f"All specialists ranked {sid} first; no coordinator call needed.",
                "unanimous": True}}
        return {"shortlisted": union}

    return node


def _route(state: MAState) -> str:
    return END if state.get("decision") else "coordinator"


def _coordinator_node(client: LLMClient):
    system = P.system_coordinator()

    def node(state: MAState) -> dict[str, Any]:
        ids = state.get("shortlisted") or list(state["menu"])  # all specialists failed -> full menu
        shots = [Candidate.model_validate(state["menu"][i]) for i in ids]
        user = P.user_coordinator(state["pos_text"], shots, state.get("reports", {}))
        res, att = call_with_id_check(client, system=system, user=user, schema=CoordinatorDecision,
                                      agent="coordinator", valid_ids=set(ids),
                                      repeat_idx=state.get("sample_idx", 0))
        dec = None
        if res.parsed is not None and res.parsed.shot_id in ids:
            dec = {**res.parsed.model_dump(), "unanimous": False,
                   "from_full_menu": not state.get("shortlisted")}
        return {"decision": dec, "attempts": att}

    return node


def build_graph(client: LLMClient, specialists: tuple[str, ...] = P.PERSPECTIVES):
    g = StateGraph(MAState)
    for s in specialists:
        g.add_node(s, _specialist_node(client, s))
        g.add_edge(START, s)
    g.add_node("merge", _merge_node(specialists))
    g.add_edge(list(specialists), "merge")
    g.add_node("coordinator", _coordinator_node(client))
    g.add_conditional_edges("merge", _route, ["coordinator", END])
    g.add_edge("coordinator", END)
    return g.compile()
