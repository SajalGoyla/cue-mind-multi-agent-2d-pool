"""B1: geometric bot (easiest pot). Used in the noise pilot to check that decision quality
shows up in game results (B should beat B1 clearly)."""

from __future__ import annotations

from cuemind.policies.base import Decision, DecisionContext, intent_of


class GeometricBot:
    """The geometrically easiest pot, preferring medium speed with a stun (centre) hit.
    With no pot available it plays the first listed safety (or the contact fallback)."""

    name = "B1_geometric"

    def decide(self, ctx: DecisionContext) -> Decision:
        pots = [c for c in ctx.candidates if c.shot_type == "pot"]
        if pots:
            best_ease = max(c.geometry.ease for c in pots)
            easiest = sorted((c for c in pots if c.geometry.ease == best_ease), key=lambda c: c.shot_id)
            c = next((x for x in easiest if x.power_label == "medium" and x.spin_label == "stun"),
                     easiest[0])
            return Decision(shot_id=c.shot_id, intent="offense",
                            rationale=f"easiest pot (ease {best_ease:.2f})",
                            trace={"ease": best_ease})
        c = min(ctx.candidates, key=lambda x: x.shot_id)
        return Decision(shot_id=c.shot_id, intent=intent_of(c), rationale="no pot available")
