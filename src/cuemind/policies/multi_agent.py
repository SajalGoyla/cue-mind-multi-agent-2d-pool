"""MA v2: offense, position and safety specialists nominate shortlists; a coordinator chooses
from their union (skipped when all three rank the same shot first)."""

from __future__ import annotations

import time

from cuemind.agents.common import sum_usage
from cuemind.agents.graph_multi import build_graph
from cuemind.llm import promptlib as P
from cuemind.llm.client import LLMClient
from cuemind.policies.base import Decision, DecisionContext, fallback_choice, intent_of


class MultiAgent:
    name = "MA"

    def __init__(self, client: LLMClient):
        self.client = client
        self._graphs: dict[str, object] = {}  # one compiled graph per information level

    def decide(self, ctx: DecisionContext) -> Decision:
        t0 = time.perf_counter()
        ctx_text = P.decision_context_text(ctx.state, ctx.candidates, ctx.history, ctx.ball_in_hand)
        if ctx.info not in self._graphs:
            self._graphs[ctx.info] = build_graph(self.client, ctx.info)
        out = self._graphs[ctx.info].invoke({"ctx_text": ctx_text,
                                 "valid_ids": [c.shot_id for c in ctx.candidates],
                                 "descriptions": {c.shot_id: c.description for c in ctx.candidates},
                                 "sample_idx": ctx.sample_idx, "shortlists": {}, "attempts": [],
                                 "decision": None, "unanimous": False})
        usage = sum_usage(out.get("attempts", []), time.perf_counter() - t0)
        dec = out.get("decision")
        trace = {"prompt_version": P.prompt_version(), "shortlists": out.get("shortlists", {}),
                 "unanimous": bool(out.get("unanimous")), "agents": out.get("attempts", []),
                 "context_prompt": ctx_text}
        if dec is None:
            fb = fallback_choice(ctx.candidates)
            return Decision(shot_id=fb.shot_id, intent=intent_of(fb), fallback=True,
                            rationale="fallback: no valid team decision", trace=trace, usage=usage)
        trace["followed"] = dec["followed"]
        c = ctx.by_id(dec["shot_id"])
        return Decision(shot_id=dec["shot_id"], intent=intent_of(c), rationale=dec["rationale"],
                        trace=trace, usage=usage)
