"""MA v2: offense, position and defense specialists shortlist shots; a coordinator picks one
of the shortlisted shots (skipped when all three rank the same shot first). See
agents/graph_multi.py for the graph."""

from __future__ import annotations

import time

from cuemind.agents.common import sum_usage
from cuemind.agents.graph_multi import build_graph
from cuemind.llm import promptlib as P
from cuemind.llm.client import LLMClient
from cuemind.llm.schemas import clamp01
from cuemind.policies.base import Decision, DecisionContext, fallback_choice, intent_of


class MultiAgent:
    name = "MA"

    def __init__(self, client: LLMClient):
        self.client = client
        self.graph = build_graph(client, P.PERSPECTIVES)

    def decide(self, ctx: DecisionContext) -> Decision:
        t0 = time.perf_counter()
        out = self.graph.invoke({
            "ctx_text": P.decision_context_text(ctx.state, ctx.candidates, ctx.history,
                                                ctx.ball_in_hand),
            "pos_text": P.position_text(ctx.state, ctx.history, ctx.ball_in_hand),
            "menu": {c.shot_id: c.model_dump() for c in ctx.candidates},
            "sample_idx": ctx.sample_idx, "reports": {}, "attempts": [], "decision": None})
        usage = sum_usage(out.get("attempts", []), time.perf_counter() - t0)
        reports = out.get("reports", {})
        dec = out.get("decision")
        trace = {"prompt_version": P.prompt_version(), "reports": reports,
                 "shortlisted": out.get("shortlisted", []), "agents": out.get("attempts", []),
                 "context_prompt": out.get("ctx_text", "")}
        if dec is None:
            fb = fallback_choice(ctx.candidates)
            return Decision(shot_id=fb.shot_id, intent=intent_of(fb), fallback=True,
                            rationale="fallback: coordinator gave no valid answer",
                            trace=trace, usage=usage)
        sid = dec["shot_id"]
        trace.update(unanimous=dec.get("unanimous", False), followed=dec.get("followed", []),
                     from_full_menu=dec.get("from_full_menu", False),
                     shortlisted_by=[a for a, r in reports.items()
                                     if sid in {i["shot_id"] for i in r.get("shortlist", [])}])
        cand = ctx.by_id(sid)
        return Decision(shot_id=sid, intent=dec.get("intent") or intent_of(cand),
                        rationale=dec["rationale"],
                        confidence=None if dec.get("confidence") is None else clamp01(dec["confidence"]),
                        trace=trace, usage=usage)
