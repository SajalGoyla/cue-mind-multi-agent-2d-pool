"""Stat: the simulation-based reference player (v1 name: B2). It chooses from the same menu
as the LLM systems but uses the menu's attached forecasts (3 noisy simulations per variant,
averaged) and picks the candidate with the highest expected value under a transparent,
hand-written objective. It therefore has more information than the LLMs (D-029).

    EV = p_pot * (1 + W_NEXT * next_ease)
         - W_SCRATCH * p_scratch - W_FOUL * p_foul
         - W_THREAT * (1 - p_keep_turn) * opp_threat

Weights live in `SearchWeights` (D-011, unchanged in v2).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from cuemind.candidates.model import Candidate
from cuemind.policies.base import Decision, DecisionContext, intent_of


@dataclass(frozen=True)
class SearchWeights:
    next: float = 0.5
    scratch: float = 1.5
    foul: float = 1.0
    threat: float = 0.6


def expected_value(c: Candidate, w: SearchWeights) -> float:
    if c.rollouts is not None:
        r = c.rollouts
        return (r.p_pot * (1 + w.next * r.next_ease) - w.scratch * r.p_scratch
                - w.foul * (r.p_foul - r.p_scratch) - w.threat * (1 - r.p_keep_turn) * r.opp_threat)
    if c.preview is not None:
        p = c.preview
        nxt = min(p.next_viable_pots, 5) / 5
        return (1 + w.next * nxt if p.pots_called else 0.0) - (w.scratch if p.scratch else 0.0) \
            - (w.foul if (p.foul and not p.scratch) else 0.0)
    return c.geometry.ease if c.shot_type == "pot" else 0.0


class SearchBot:
    name = "Stat"

    def __init__(self, weights: SearchWeights | None = None):
        self.w = weights or SearchWeights()

    def decide(self, ctx: DecisionContext) -> Decision:
        scored = sorted(((expected_value(c, self.w), c) for c in ctx.candidates),
                        key=lambda t: (-t[0], t[1].shot_id))
        ev, best = scored[0]
        return Decision(shot_id=best.shot_id, intent=intent_of(best),
                        rationale=f"highest expected value ({ev:.3f})",
                        trace={"ev": {c.shot_id: round(v, 4) for v, c in scored},
                               "weights": asdict(self.w)})
