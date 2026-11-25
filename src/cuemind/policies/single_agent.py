"""SA: one LLM call per decision on the raw menu.

The system prompt holds the rules and all three strategy blocks (offense, position, defense),
so MA's specialists never know anything SA does not (fairness contract).
"""

from __future__ import annotations

import time

from cuemind.agents.common import call_with_id_check, sum_usage
from cuemind.llm import promptlib as P
from cuemind.llm.client import LLMClient
from cuemind.llm.schemas import SingleDecision, clamp01
from cuemind.policies.base import Decision, DecisionContext, fallback_choice, intent_of


class SingleAgent:
    name = "SA"

    def __init__(self, client: LLMClient):
        self.client = client
        self.system = P.system_single()

    def decide(self, ctx: DecisionContext) -> Decision:
        t0 = time.perf_counter()
        ctx_text = P.decision_context_text(ctx.state, ctx.candidates, ctx.history, ctx.ball_in_hand)
        user = P.user_single(ctx_text)
        valid = {c.shot_id for c in ctx.candidates}
        res, attempts = call_with_id_check(self.client, system=self.system, user=user,
                                           schema=SingleDecision, agent="single", valid_ids=valid,
                                           repeat_idx=ctx.sample_idx)
        usage = sum_usage(attempts, time.perf_counter() - t0)
        trace = {"prompt_version": P.prompt_version(), "agents": attempts, "user_prompt": user}
        if res.parsed is None or res.parsed.shot_id not in valid:
            fb = fallback_choice(ctx.candidates)
            return Decision(shot_id=fb.shot_id, intent=intent_of(fb), fallback=True,
                            rationale="fallback: no valid model answer", trace=trace, usage=usage)
        p = res.parsed
        return Decision(shot_id=p.shot_id, intent=p.intent, rationale=p.rationale,
                        confidence=clamp01(p.confidence), trace=trace, usage=usage)
