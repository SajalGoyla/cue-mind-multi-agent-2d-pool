"""B: the simulation-based reference player (D-029).

B chooses from the same raw menu as the LLM systems, but before choosing it simulates every
candidate privately with the game's execution noise (n runs; one run when there is no noise,
because the result is then exact) and picks the highest expected value:

    EV = p_pot * (1 + W_NEXT * next_ease)
         - W_SCRATCH * p_scratch - W_FOUL * (p_foul - p_scratch)
         - W_THREAT * (1 - p_keep_turn) * opp_threat

B therefore has more information than SA and MA. That is intended: it is the common
yardstick every system plays against. The weights are the frozen D-011 weights.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from cuemind.candidates.annotate import rollouts
from cuemind.candidates.model import Rollouts
from cuemind.policies.base import Decision, DecisionContext, intent_of
from cuemind.sim.noise import get_noise


@dataclass(frozen=True)
class SearchWeights:
    next: float = 0.5
    scratch: float = 1.5
    foul: float = 1.0
    threat: float = 0.6


def expected_value(r: Rollouts, w: SearchWeights) -> float:
    return (r.p_pot * (1 + w.next * r.next_ease) - w.scratch * r.p_scratch
            - w.foul * (r.p_foul - r.p_scratch) - w.threat * (1 - r.p_keep_turn) * r.opp_threat)


class SearchBot:
    name = "B_search"

    def __init__(self, weights: SearchWeights | None = None, n_rollouts: int = 8):
        self.w = weights or SearchWeights()
        self.n_rollouts = n_rollouts

    def decide(self, ctx: DecisionContext) -> Decision:
        n = 1 if get_noise(ctx.noise).is_zero else self.n_rollouts
        sims = rollouts(ctx.state, ctx.candidates, n, ctx.noise, ctx.sim_key)
        scored = sorted(((expected_value(c.rollouts, self.w), c) for c in sims),
                        key=lambda t: (-t[0], t[1].shot_id))
        ev, best = scored[0]
        return Decision(shot_id=best.shot_id, intent=intent_of(best),
                        rationale=f"highest expected value in private simulation ({ev:.3f})",
                        trace={"ev": {c.shot_id: round(v, 4) for v, c in scored},
                               "sim": {c.shot_id: c.rollouts.model_dump(
                                   include={"p_pot", "p_scratch", "p_foul", "next_ease", "opp_threat"})
                                   for _, c in scored},
                               "n_rollouts": n, "weights": asdict(self.w)})
