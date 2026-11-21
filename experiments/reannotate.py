"""Re-annotate an existing positions file at another execution-noise level (E5, RQ3).

    python experiments/reannotate.py --src data/positions/test.jsonl --noise low --out data/positions/test_noise_low.jsonl

The table states are kept; the menu (L2 rollouts and pruning) and the oracle are rebuilt
with the new noise level, because both depend on how reliably strokes are executed.
Position ids get a suffix (e.g. pos_0123@low) so results never mix noise levels.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from cuemind.candidates.pipeline import PrepConfig
from cuemind.eval.oracle import OracleConfig
from cuemind.eval.positions import build_record, load_positions
from cuemind.sim.state import TableState

ROOT = Path(__file__).resolve().parents[1]


def _init():
    from cuemind.sim.execute import warm_up

    warm_up()


def _one(args):
    rec, noise, m = args
    out = build_record(f"{rec['position_id']}@{noise}", rec["split"], rec["source"],
                       TableState.model_validate(rec["state"]),
                       PrepConfig(access_level="L2", noise=noise), OracleConfig(m=m, noise=noise))
    if out is not None:
        out["base_position_id"] = rec["position_id"]
        out["noise"] = noise
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--noise", required=True, choices=["none", "low", "medium", "high"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--oracle-m", type=int, default=24)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args(argv)
    recs = load_positions(ROOT / a.src)
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init) as pool:
        futs = [pool.submit(_one, (r, a.noise, a.oracle_m)) for r in recs]
        for k, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r is not None:
                rows.append(r)
            if k % 30 == 0:
                print(f"  {k}/{len(recs)}", flush=True)
    rows.sort(key=lambda r: r["position_id"])
    with open(ROOT / a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} positions to {a.out}")


if __name__ == "__main__":
    main()
