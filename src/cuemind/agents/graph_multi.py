"""Multi-agent decision graph (LangGraph).

    START ──► offense ──┐
          ├─► position ─┼─► coordinator ──► (critic, optional) ──► END
          └─► safety ───┘

The three specialists run in the same LangGraph superstep (concurrently) on the same user
message, each with its own system prompt. The coordinator sees the reports and decides.
The optional critic may veto once; the coordinator is then asked again with the critic's
reason. Specialists can be disabled for ablations (E6).
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from cuemind.agents.common import call_with_id_check
from cuemind.llm import promptlib as P
from cuemind.llm.client import LLMClient
from cuemind.llm.schemas import CoordinatorDecision, CriticVerdict, SpecialistReport


class MAState(TypedDict, total=False):
    ctx_text: str
    valid_ids: list[str]
    sample_idx: int
    reports: Annotated[dict[str, dict], operator.or_]
    attempts: Annotated[list[dict], operator.add]
    decision: dict | None
    critic: dict | None


def _specialist_node(client: LLMClient, perspective: str):
    system = P.system_specialist(perspective)

    def node(state: MAState) -> dict[str, Any]:
        valid = set(state["valid_ids"])
        res, att = call_with_id_check(client, system=system,
                                      user=P.user_specialist(state["ctx_text"], perspective),
                                      schema=SpecialistReport, agent=perspective,
                                      valid_ids=valid, id_field="top_choice",
                                      repeat_idx=state.get("sample_idx", 0))
        report = None
        if res.parsed is not None:
            r = res.parsed
            report = {"top_choice": r.top_choice if r.top_choice in valid else None,
                      "summary": r.summary,
                      "scores": {a.shot_id: max(0, min(10, a.score)) for a in r.assessments
                                 if a.shot_id in valid},
                      "notes": {a.shot_id: a.note for a in r.assessments if a.shot_id in valid}}
        return {"reports": {perspective: report or {"error": res.error or "invalid report"}},
                "attempts": att}

    return node


def _coordinator_node(client: LLMClient):
    system = P.system_coordinator()

    def node(state: MAState) -> dict[str, Any]:
        valid = set(state["valid_ids"])
        user = P.user_coordinator(state["ctx_text"], state.get("reports", {}))
        critic = state.get("critic")
        if critic and not critic.get("approve", True):
            user += (f"\n\nA critic rejected your previous choice {critic.get('rejected')!r}: "
                     f"{critic.get('reason')} Reconsider and choose again.")
        res, att = call_with_id_check(client, system=system, user=user,
                                      schema=CoordinatorDecision, agent="coordinator",
                                      valid_ids=valid, repeat_idx=state.get("sample_idx", 0))
        dec = res.parsed.model_dump() if (res.parsed and res.parsed.shot_id in valid) else None
        return {"decision": dec, "attempts": att}

    return node


def _critic_node(client: LLMClient):
    system = P.system_critic()

    def node(state: MAState) -> dict[str, Any]:
        dec = state.get("decision")
        if dec is None or state.get("critic") is not None:  # nothing to review / one veto max
            return {}
        res = client.complete(system=system, user=P.user_critic(state["ctx_text"], dec),
                              schema=CriticVerdict, agent="critic",
                              repeat_idx=state.get("sample_idx", 0))
        verdict = {"approve": True, "reason": res.error or "", "rejected": None}
        if res.parsed is not None:
            verdict = {**res.parsed.model_dump(), "rejected": dec["shot_id"]}
        return {"critic": verdict, "attempts": [{"agent": "critic", "attempt": 0, **res.log()}]}

    return node


def _after_critic(state: MAState) -> str:
    c = state.get("critic")
    if c and not c.get("approve", True):
        return "coordinator_retry"
    return END


def build_graph(client: LLMClient, specialists: tuple[str, ...] = P.PERSPECTIVES,
                critic: bool = False):
    g = StateGraph(MAState)
    for s in specialists:
        g.add_node(s, _specialist_node(client, s))
        g.add_edge(START, s)
    g.add_node("coordinator", _coordinator_node(client))
    if specialists:
        g.add_edge(list(specialists), "coordinator")
    else:
        g.add_edge(START, "coordinator")
    if critic:
        g.add_node("critic", _critic_node(client))
        g.add_node("coordinator_retry", _coordinator_node(client))
        g.add_edge("coordinator", "critic")
        g.add_conditional_edges("critic", _after_critic, ["coordinator_retry", END])
        g.add_edge("coordinator_retry", END)
    else:
        g.add_edge("coordinator", END)
    return g.compile()
