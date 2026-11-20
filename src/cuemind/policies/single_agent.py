"""SA (single agent) and SA-vote (compute-matched control: k independent samples, majority vote).

The single agent's system prompt contains the same strategy knowledge as the union of the
three specialists' prompts (fairness contract), so any multi-agent advantage cannot come
from extra domain knowledge.
"""

from __future__ import annotations

import time
from collections import Counter

from cuemind.agents.common import call_with_id_check, sum_usage
from cuemind.llm import promptlib as P
from cuemind.llm.client import LLMClient
from cuemind.llm.schemas import SingleDecision, clamp01
from cuemind.policies.base import Decision, DecisionContext, fallback_choice, intent_of


class SingleAgent:
    def __init__(self, client: LLMClient, n_votes: int = 1, name: str | None = None):
        self.client, self.n_votes = client, n_votes
        self.name = name or ("SA" if n_votes == 1 else f"SA-vote{n_votes}")
        self.system = P.system_single()

    def decide(self, ctx: DecisionContext) -> Decision:
        t0 = time.perf_counter()
        ctx_text = P.decision_context_text(ctx.state, ctx.candidates, ctx.history, ctx.ball_in_hand)
        user = P.user_single(ctx_text)
        valid = {c.shot_id for c in ctx.candidates}
        attempts, votes = [], []
        for i in range(self.n_votes):
            res, att = call_with_id_check(self.client, system=self.system, user=user,
                                          schema=SingleDecision, agent="single", valid_ids=valid,
                                          repeat_idx=ctx.sample_idx * 10 + i)
            attempts += att
            if res.parsed is not None and res.parsed.shot_id in valid:
                votes.append(res.parsed)
        usage = sum_usage(attempts, time.perf_counter() - t0)
        trace = {"prompt_version": P.prompt_version(), "agents": attempts, "user_prompt": user}
        if not votes:
            fb = fallback_choice(ctx.candidates)
            return Decision(shot_id=fb.shot_id, intent=intent_of(fb), fallback=True,
                            rationale="fallback: no valid model answer", trace=trace, usage=usage)
        counts = Counter(v.shot_id for v in votes)
        top = max(counts.values())
        tied = [s for s, n in counts.items() if n == top]
        conf = {s: sum(clamp01(v.confidence) for v in votes if v.shot_id == s) / counts[s] for s in tied}
        winner = max(tied, key=lambda s: (conf[s], s))
        chosen = next(v for v in votes if v.shot_id == winner)
        if self.n_votes > 1:
            trace["votes"] = dict(counts)
        return Decision(shot_id=winner, intent=chosen.intent, rationale=chosen.rationale,
                        confidence=clamp01(chosen.confidence), trace=trace, usage=usage)
