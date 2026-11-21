"""Rule-based error categories for every high-regret decision (complements the hand-read
cases in docs/failure_analysis_<exp>.md).

    python experiments/categorize_errors.py --exp e1 --threshold 0.3

A decision is an error if its regret exceeds the threshold (oracle EIV units). Each error
gets exactly one category, checked in this order:
    variant        same target ball and pocket as the best shot, wrong speed/spin
    scratch_risk   chosen shot's scratch probability >= best shot's + 0.10
    needless_safety  chose a safety while the best shot is a pot with p_pot >= 0.6
    pot_over_safety  chose a pot while the best shot is a safety
    lower_pot      chose a pot whose p_pot is >= 0.15 below the best shot's
    other_target   a different target/pocket with similar pot probability (leave misjudged)
For MA it also records whether the coordinator followed a specialist whose top choice was
the chosen shot while another specialist had named the best shot ("wrong_specialist").
Writes results/<exp>/error_categories.csv and prints a table.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from cuemind.eval.positions import load_positions

ROOT = Path(__file__).resolve().parents[1]


def categorise(rec: dict, d: dict) -> str:
    cands = {c["shot_id"]: c for c in rec["candidates"]}
    orc = rec["oracle"]
    best = cands[max(orc, key=lambda k: orc[k]["eiv"])]
    ch = cands[d["shot_id"]]
    rb, rc = best.get("rollouts") or {}, ch.get("rollouts") or {}
    if ch["shot_type"] == best["shot_type"] == "pot" and \
            (ch["target_ball"], ch["pocket"]) == (best["target_ball"], best["pocket"]):
        return "variant"
    if rc.get("p_scratch", 0) >= rb.get("p_scratch", 0) + 0.10:
        return "scratch_risk"
    if ch["shot_type"] != "pot" and best["shot_type"] == "pot" and rb.get("p_pot", 0) >= 0.6:
        return "needless_safety"
    if ch["shot_type"] == "pot" and best["shot_type"] != "pot":
        return "pot_over_safety"
    if ch["shot_type"] == "pot" and rc.get("p_pot", 0) <= rb.get("p_pot", 0) - 0.15:
        return "lower_pot"
    return "other_target"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="e1")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--level", default="L2")
    a = ap.parse_args(argv)
    cfg_path = ROOT / "results" / a.exp / "config_used.json"
    files = json.loads(cfg_path.read_text(encoding="utf-8"))["positions"] if cfg_path.exists() \
        else ["data/positions/test.jsonl"]
    recs = {}
    for f in files:
        recs.update({r["position_id"]: r for r in load_positions(ROOT / f)})
    rows = []
    for line in (ROOT / "results" / a.exp / "decisions.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if d["access_level"] != a.level or d["position_id"] not in recs:
            continue
        err = d["regret"] > a.threshold
        cat = categorise(recs[d["position_id"]], d) if err else "ok"
        wrong_spec = None
        reps = (d.get("trace") or {}).get("reports") or {}
        if err and reps:
            orc = recs[d["position_id"]]["oracle"]
            best_id = max(orc, key=lambda k: orc[k]["eiv"])
            followed = [k for k, r in reps.items() if r.get("top_choice") == d["shot_id"]]
            knew = [k for k, r in reps.items() if r.get("top_choice") == best_id]
            wrong_spec = bool(followed and knew)
        rows.append({"system": d["system_key"], "position_id": d["position_id"], "regret": d["regret"],
                     "category": cat, "wrong_specialist": wrong_spec})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / a.exp / "error_categories.csv", index=False)
    tab = pd.crosstab(df.system, df.category)
    tab["errors"] = tab.drop(columns=["ok"], errors="ignore").sum(axis=1)
    tab["n"] = df.groupby("system").size()
    print(f"Errors = decisions with regret > {a.threshold} at {a.level}")
    print(tab.to_string())
    ma = df[(df.system.str.startswith("ma")) & (df.category != "ok")]
    if len(ma):
        print(f"\nMA errors where a specialist had named the best shot but the coordinator followed "
              f"another: {int(ma.wrong_specialist.sum())} of {len(ma)}")


if __name__ == "__main__":
    main()
