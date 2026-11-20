"""Pick sample runs for the presentation from REAL experiment logs and export PNGs.

    python scripts/export_demo_assets.py --exp e1

Selection rules (plan section 13.2), applied to results/<exp>/decisions.jsonl at L2:
    S1 agreement: SA and MA choose the same shot, and it is the oracle's best
    S2 position:  a `position`-tagged position where SA and MA differ (largest regret gap)
    S3 safety:    a `defense`-tagged position where MA chose a safety
    S4 failure:   MA's highest-regret decision
Writes results/demo_assets/<slot>_<position>.png and results/demo_assets/sample_runs.json
(the demo can load the latter). Missing slots are reported, never invented.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cuemind.eval.positions import load_positions, menu_at_level
from cuemind.viz.static_table import draw_table

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="e1")
    a = ap.parse_args(argv)
    dec = [json.loads(x) for x in (ROOT / "results" / a.exp / "decisions.jsonl").read_text(
        encoding="utf-8").splitlines() if x.strip()]
    dec = [d for d in dec if d["access_level"] == "L2" and d["repeat"] == 0]
    positions = {}
    for f in ("data/positions/test.jsonl", "data/positions/hard_cases.jsonl"):
        if (ROOT / f).exists():
            positions.update({r["position_id"]: r for r in load_positions(ROOT / f)})
    by = {}
    for d in dec:
        by.setdefault(d["position_id"], {})[d["system_key"]] = d
    pairs = [(pid, v["sa"], v["ma"]) for pid, v in by.items() if "sa" in v and "ma" in v and pid in positions]

    picks = {}
    s1 = [p for p in pairs if p[1]["shot_id"] == p[2]["shot_id"] and p[2]["is_best"] == 1.0]
    if s1:
        picks["S1_agreement"] = s1[0]
    s2 = sorted((p for p in pairs if "position" in positions[p[0]]["tags"] and p[1]["shot_id"] != p[2]["shot_id"]),
                key=lambda p: -(p[1]["regret"] - p[2]["regret"]))
    if s2:
        picks["S2_position"] = s2[0]
    s3 = [p for p in pairs if "defense" in positions[p[0]]["tags"] and p[2]["intent"] == "safety"]
    if s3:
        picks["S3_safety"] = s3[0]
    if pairs:
        picks["S4_failure"] = max(pairs, key=lambda p: p[2]["regret"])

    out_dir = ROOT / "results" / "demo_assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for slot, (pid, sa, ma) in picks.items():
        rec = positions[pid]
        state, menu = menu_at_level(rec, "L2")
        png = draw_table(state, str(out_dir / f"{slot}_{pid}.png"),
                         title=f"{slot}: SA {sa['shot_id']} vs MA {ma['shot_id']}",
                         candidates=menu, highlight=ma["shot_id"])
        manifest.append({"slot": slot, "position_id": pid, "png": str(Path(png).relative_to(ROOT)),
                         "sa": {k: sa[k] for k in ("shot_id", "intent", "regret", "rationale")},
                         "ma": {k: ma[k] for k in ("shot_id", "intent", "regret", "rationale")},
                         "state": rec["state"]})
    missing = sorted({"S1_agreement", "S2_position", "S3_safety", "S4_failure"} - set(picks))
    (out_dir / "sample_runs.json").write_text(json.dumps({"exp": a.exp, "runs": manifest,
                                                          "missing": missing}, indent=2), encoding="utf-8")
    print(f"exported {len(manifest)} sample runs; missing: {missing or 'none'}")


if __name__ == "__main__":
    main()
