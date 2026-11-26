"""Live check of the local model (v2): one single-agent and one team decision on a real menu.

    python scripts/check_llm.py

Needs `ollama serve` and the model (default qwen2.5:3b-instruct, or CUEMIND_OLLAMA_MODEL).
Responses are cached, so re-running is instant.
"""

from __future__ import annotations

import json
import sys

from cuemind.candidates.pipeline import PrepConfig, prepare
from cuemind.policies.base import DecisionContext
from cuemind.policies.multi_agent import MultiAgent
from cuemind.policies.registry import make_llm_client
from cuemind.policies.single_agent import SingleAgent
from cuemind.sim.execute import warm_up
from cuemind.sim.state import TableState

DEMO = TableState(cue=(0.5, 0.6), balls={"1": (0.20, 1.70), "3": (0.75, 1.2), "4": (0.3, 0.9),
                                          "10": (0.5, 1.0), "12": (0.8, 1.7), "8": (0.3, 1.3)},
                  groups={"P1": "solids", "P2": "stripes"})


def main() -> int:
    warm_up()
    client = make_llm_client(llm="ollama")
    print(f"model={client.model} temperature={client.temperature}")
    menu = prepare(DEMO, PrepConfig(), rollout_key=("check",))
    print("\n".join(f"  {c.shot_id} | {c.description}" for c in menu))
    ctx = DecisionContext(state=DEMO, candidates=menu)
    ok = True
    for policy in (SingleAgent(client), MultiAgent(client)):
        d = policy.decide(ctx)
        ok &= not d.fallback
        print(f"\n{policy.name}: {d.shot_id} fallback={d.fallback} usage={json.dumps(d.usage)}")
        print("  rationale:", d.rationale)
        for name, picks in (d.trace.get("shortlists") or {}).items():
            print(f"  [{name}] " + "; ".join(f"{p['shot_id']} ({p['reason']})" for p in picks))
        if "unanimous" in d.trace:
            print("  unanimous:", d.trace["unanimous"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
