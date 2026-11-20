"""Build hand-made hard cases from data/positions/hard_cases.yaml.

    python experiments/build_hard_cases.py

Writes data/positions/hard_cases.jsonl (benchmark records, source="hand") and a PNG plus
an oracle table per case in data/positions/hard_cases/ for human review. Draft labels:
`expected_intent` = intent of the best-EIV shot and `acceptable_shot_ids` = shots within
one standard error of the best. A team member must review each case and set
`reviewed_by` in the YAML before the case is used for reported results.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from cuemind.candidates.model import Candidate
from cuemind.candidates.pipeline import PrepConfig
from cuemind.eval.oracle import OracleConfig
from cuemind.eval.positions import build_record
from cuemind.sim.execute import warm_up
from cuemind.sim.state import TABLE_L, TABLE_W, TableState, validate
from cuemind.viz.static_table import draw_table

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "positions" / "hard_cases.yaml"
OUT = ROOT / "data" / "positions" / "hard_cases.jsonl"
PNG_DIR = ROOT / "data" / "positions" / "hard_cases"


def to_m(xy):
    return (xy[0] / 4 * TABLE_W, xy[1] / 8 * TABLE_L)


def main() -> None:
    warm_up()
    cfg = yaml.safe_load(SRC.read_text(encoding="utf-8"))
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for case in cfg["cases"]:
        bih = bool(case.get("ball_in_hand"))
        state = TableState(cue=None if bih else to_m(case["cue"]),
                           balls={k: to_m(v) for k, v in case["balls"].items()},
                           shooter="P1", groups=case["groups"], ball_in_hand=bih)
        problems = validate(state)
        if problems:
            print(f"{case['id']}: INVALID {problems}")
            continue
        rec = build_record(case["id"], "hard", "hand", state, PrepConfig(access_level="L2"),
                           OracleConfig(m=48))
        if rec is None:
            print(f"{case['id']}: fewer than 2 candidates, skipped")
            continue
        orc = rec["oracle"]
        best = max(orc, key=lambda k: orc[k]["eiv"])
        cands = {c["shot_id"]: Candidate.model_validate(c) for c in rec["candidates"]}
        acceptable = sorted(k for k, v in orc.items() if v["eiv"] >= orc[best]["eiv"] - orc[best]["se"])
        rec.update(category=case["category"], intended=case["intended"],
                   expected_intent="offense" if cands[best].shot_type == "pot" else "safety",
                   acceptable_shot_ids=acceptable, reviewed_by=case.get("reviewed_by"),
                   label_status="reviewed" if case.get("reviewed_by") else "draft (oracle-proposed)")
        out.append(rec)
        st = TableState.model_validate(rec["state"])
        draw_table(st, str(PNG_DIR / f"{case['id']}.png"), title=f"{case['id']} ({case['category']})",
                   candidates=list(cands.values()), highlight=best)
        lines = [f"# {case['id']} ({case['category']})", "", case["intended"], "",
                 f"Draft label: intent **{rec['expected_intent']}**, acceptable {', '.join(acceptable)}", "",
                 "| id | EIV | SE | description |", "|---|---|---|---|"]
        for k in sorted(orc, key=lambda k: -orc[k]["eiv"]):
            lines.append(f"| {k} | {orc[k]['eiv']:.2f} | {orc[k]['se']:.2f} | {cands[k].description} |")
        (PNG_DIR / f"{case['id']}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{case['id']}: {len(cands)} candidates, best {best} ({cands[best].shot_type}, "
              f"EIV {orc[best]['eiv']:.2f}), {rec['difficulty']}, tags {rec['tags']}")
    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out)} cases to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
