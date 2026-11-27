"""Rule: the reference player of v3 condition A (D-031). It reads exactly what the LLMs read
(geometry + simulation-free physics hints) and applies a fixed, hand-written rule:

    score(pot) = ease x (1 + 0.5 x hinted next-shot ease) - 1.0 x scratch warning

It plays the best-scoring pot; if no pot scores above SAFETY_BELOW it plays a safety (one of
the menu's safeties, medium speed first). Weights were fixed before any v3 game was played.
"""

from __future__ import annotations

from cuemind.candidates.hints import Hint, rule_score
from cuemind.policies.base import Decision, DecisionContext

SAFETY_BELOW = 0.2


class RuleBot:
    name = "Rule"

    def decide(self, ctx: DecisionContext) -> Decision:
        scored = sorted(((rule_score(c, Hint(**c.hint) if c.hint else None), c) for c in ctx.candidates),
                        key=lambda t: (-t[0], t[1].shot_id))
        best_score, best = scored[0]
        safeties = [c for c in ctx.candidates if c.shot_type == "safety"]
        if best_score < SAFETY_BELOW and safeties:
            s = min(safeties, key=lambda c: (c.power_label != "medium", c.geometry.cut_deg, c.shot_id))
            return Decision(shot_id=s.shot_id, intent="safety",
                            rationale=f"no pot scores above {SAFETY_BELOW} (best {best_score:.2f})",
                            trace={"scores": {c.shot_id: round(v, 4) for v, c in scored}})
        return Decision(shot_id=best.shot_id, intent="offense" if best.shot_type == "pot" else "safety",
                        rationale=f"best rule score {best_score:.2f}",
                        trace={"scores": {c.shot_id: round(v, 4) for v, c in scored}})
