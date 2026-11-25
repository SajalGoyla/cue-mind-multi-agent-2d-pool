"""B1: geometric easiest-pot bot (used only in the v2 pilot)."""

from __future__ import annotations

from cuemind.policies.base import Decision, DecisionContext, intent_of


class GeometricBot:
    """B1: the geometrically easiest pot, played at medium speed with a stun (centre) hit.
    With no pot available it plays the first safety (or the contact fallback)."""

    name = "B1_geometric"

    def decide(self, ctx: DecisionContext) -> Decision:
        pots = [c for c in ctx.candidates if c.shot_type == "pot"]
        if pots:
            best_ease = max(c.geometry.ease for c in pots)
            easiest = [c for c in pots if c.geometry.ease == best_ease]
            c = next((x for x in easiest if x.power_label == "medium" and x.spin_label == "stun"),
                     easiest[0])
            return Decision(shot_id=c.shot_id, intent="offense",
                            rationale=f"easiest pot (ease {best_ease:.2f})",
                            trace={"ease": best_ease})
        c = ctx.candidates[0]
        return Decision(shot_id=c.shot_id, intent=intent_of(c), rationale="no pot available")
