"""MA: offense, position and safety specialists plus a coordinator (optionally a critic)."""

from __future__ import annotations

import time

from cuemind.agents.common import sum_usage
from cuemind.agents.graph_multi import build_graph
from cuemind.llm import promptlib as P
from cuemind.llm.client import LLMClient
from cuemind.llm.schemas import clamp01
from cuemind.policies.base import Decision, DecisionContext, fallback_choice, intent_of


class MultiAgent:
    def __init__(self, client: LLMClient, specialists: tuple[str, ...] = P.PERSPECTIVES,
                 critic: bool = False, name: str | None = None):
        self.client = client
        self.specialists = tuple(specialists)
        self.critic = critic
        default = "MA" + ("+critic" if critic else "")
        if set(self.specialists) != set(P.PERSPECTIVES):
            dropped = sorted(set(P.PERSPECTIVES) - set(self.specialists))
            default += "-no-" + "-".join(dropped)
        self.name = name or default
        self.graph = build_graph(client, self.specialists, critic)

    def decide(self, ctx: DecisionContext) -> Decision:
        t0 = time.perf_counter()
        ctx_text = P.decision_context_text(ctx.state, ctx.candidates, ctx.history, ctx.ball_in_hand)
        out = self.graph.invoke({"ctx_text": ctx_text,
                                 "valid_ids": [c.shot_id for c in ctx.candidates],
                                 "sample_idx": ctx.sample_idx,
                                 "reports": {}, "attempts": [], "decision": None, "critic": None})
        usage = sum_usage(out.get("attempts", []), time.perf_counter() - t0)
        reports = out.get("reports", {})
        dec = out.get("decision")
        trace = {"prompt_version": P.prompt_version(), "reports": reports,
                 "critic": out.get("critic"), "agents": out.get("attempts", []),
                 "context_prompt": ctx_text}
        if dec is None:
            fb = fallback_choice(ctx.candidates)
            return Decision(shot_id=fb.shot_id, intent=intent_of(fb), fallback=True,
                            rationale="fallback: coordinator gave no valid answer",
                            trace=trace, usage=usage)
        trace["followed"] = [a for a, r in reports.items() if r.get("top_choice") == dec["shot_id"]]
        trace["overruled"] = dec.get("overruled", [])
        return Decision(shot_id=dec["shot_id"], intent=dec["intent"], rationale=dec["rationale"],
                        confidence=clamp01(dec["confidence"]), trace=trace, usage=usage)
