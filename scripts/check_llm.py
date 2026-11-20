"""Live check of the LLM setup (costs well under one US cent with Haiku 5.5).

    python scripts/check_llm.py            # one structured call, then one SA and one MA decision
    python scripts/check_llm.py --quick    # only the single structured call

Reads .env (GOOGLE_APPLICATION_CREDENTIALS, GCP_PROJECT_ID, VERTEX_REGION, CUEMIND_MODEL).
Responses are cached in CUEMIND_LLM_CACHE, so re-running is free.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from cuemind.candidates.pipeline import PrepConfig, prepare  # noqa: E402
from cuemind.llm.schemas import SingleDecision  # noqa: E402
from cuemind.policies.base import DecisionContext  # noqa: E402
from cuemind.policies.multi_agent import MultiAgent  # noqa: E402
from cuemind.policies.registry import make_llm_client  # noqa: E402
from cuemind.policies.single_agent import SingleAgent  # noqa: E402
from cuemind.sim.state import TableState  # noqa: E402

DEMO = TableState(cue=(0.5, 0.6), balls={"1": (0.20, 1.70), "3": (0.75, 1.2), "4": (0.3, 0.9),
                                          "10": (0.5, 1.0), "12": (0.8, 1.7), "8": (0.3, 1.3)},
                  groups={"P1": "solids", "P2": "stripes"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-usd", type=float, default=0.25)
    a = ap.parse_args()
    client = make_llm_client(run_id="check_llm", max_usd=a.max_usd)
    print(f"model={client.model} effort={client.effort}")

    r = client.complete(system="You answer with the requested JSON only.",
                        user="Candidates:\nS01 | easy pot\nS02 | hard pot\nPick the easier shot.",
                        schema=SingleDecision, agent="check")
    print("structured call:", "OK" if r.parsed else "FAILED", r.error or "",
          json.dumps(r.parsed.model_dump() if r.parsed else None), r.usage,
          f"{r.latency_s:.1f}s", "(cached)" if r.cached else "")
    if r.parsed is None:
        return 1
    if a.quick:
        return 0

    cands = prepare(DEMO, PrepConfig(access_level="L2", n_rollouts=8), rollout_key=(0,))
    ctx = DecisionContext(state=DEMO, candidates=cands)
    for policy in (SingleAgent(client), MultiAgent(client)):
        d = policy.decide(ctx)
        print(f"\n{policy.name}: {d.shot_id} intent={d.intent} fallback={d.fallback} "
              f"conf={d.confidence} usage={json.dumps(d.usage)}")
        print("  rationale:", d.rationale)
        for name, rep in (d.trace.get("reports") or {}).items():
            print(f"  [{name}] top={rep.get('top_choice')} :: {rep.get('summary') or rep.get('error')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
