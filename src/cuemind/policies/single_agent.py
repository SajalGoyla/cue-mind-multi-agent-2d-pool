"""SA: one LLM call per decision. Its system prompt contains the same strategy knowledge as
the union of the three specialists' prompts (fairness contract)."""

from __future__ import annotations

import time

from cuemind.agents.common import call_with_check, sum_usage
from cuemind.llm import promptlib as P
from cuemind.llm.client import LLMClient
from cuemind.llm.schemas import SingleDecision
from cuemind.policies.base import Decision, DecisionContext, fallback_choice, intent_of


class SingleAgent:
    name = "SA"

    def __init__(self, client: LLMClient):
        self.client = client

    def decide(self, ctx: DecisionContext) -> Decision:
        t0 = time.perf_counter()
        user = P.user_single(P.decision_context_text(ctx.state, ctx.candidates, ctx.history,
                                                     ctx.ball_in_hand))
        valid = {c.shot_id for c in ctx.candidates}
        res, attempts = call_with_check(self.client, system=P.system_single(ctx.info), user=user,
                                        schema=SingleDecision, agent="single", valid_ids=valid,
                                        is_valid=lambda d: d.shot_id in valid,
                                        repeat_idx=ctx.sample_idx)
        usage = sum_usage(attempts, time.perf_counter() - t0)
        trace = {"prompt_version": P.prompt_version(), "agents": attempts, "user_prompt": user}
        if res.parsed is None or res.parsed.shot_id not in valid:
            fb = fallback_choice(ctx.candidates)
            return Decision(shot_id=fb.shot_id, intent=intent_of(fb), fallback=True,
                            rationale="fallback: no valid model answer", trace=trace, usage=usage)
        c = ctx.by_id(res.parsed.shot_id)
        return Decision(shot_id=c.shot_id, intent=intent_of(c), rationale=res.parsed.rationale,
                        trace=trace, usage=usage)
