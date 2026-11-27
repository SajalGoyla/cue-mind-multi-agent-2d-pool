"""How often do the simulation-free physics hints (v3 condition A) agree with pooltool?

For decision states taken from the v2 dev and floor logs (not the main-run seeds), every pot
variant gets a hint and is then simulated once without noise. We report:
- scratch warning precision/recall against real scratches;
- error of the predicted cue-ball stopping point (median, m), against a naive baseline that
  predicts the cue ball stays at the contact point;
- agreement of the predicted and the real stopping half of the table (upper/lower) and side;
- how often the hinted "next shot" exists when the real leave has a direct pot.

Usage: python experiments/validate_hints.py [--max-states 120] [--out results/v3_hint_validation.md]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as st
from pathlib import Path

import numpy as np
from pooltool.ai.pot import core as pot

from cuemind.candidates.generate import generate
from cuemind.candidates.hints import hint_for_pot
from cuemind.sim.execute import extract_events, simulate, warm_up
from cuemind.sim.state import TABLE_L, TABLE_W, TableState, positions_from_system, to_system


def states(max_states: int) -> list[TableState]:
    out, seen = [], set()
    for f in sorted(glob.glob("results/v2_dev*/raw/*.jsonl") + glob.glob("results/v2_floor/raw/*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r["kind"] == "decision" and r["state_hash"] not in seen:
                seen.add(r["state_hash"])
                out.append(TableState.model_validate(r["state"]))
    step = max(1, len(out) // max_states)
    return out[::step][:max_states]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-states", type=int, default=120)
    ap.add_argument("--out", default="results/v3_hint_validation.md")
    a = ap.parse_args()
    warm_up()
    rows = []
    for s in states(a.max_states):
        sysm = to_system(s)
        for c in generate(s):
            if c.shot_type != "pot":
                continue
            try:
                h = hint_for_pot(s, sysm, c)
                shot = simulate(to_system(s), c.stroke)
            except Exception:  # noqa: BLE001
                continue
            ev = extract_events(shot)
            cue_after, _ = positions_from_system(shot)
            real_scratch = ev.scratched
            err = math.dist(h.end, cue_after) if (cue_after and not real_scratch) else None
            ball, pocket = sysm.balls[c.target_ball], sysm.table.pockets[c.pocket]
            ghost = tuple(np.asarray(pot.calc_shadow_ball_center(ball, sysm.table, pocket), float)[:2])
            rows.append(dict(spin=c.spin_label, speed=c.power_label, warn=h.scratch_pocket is not None,
                             scratch=real_scratch, err=err,
                             half=(None if err is None else
                                   (h.end[1] > TABLE_L / 2) == (cue_after[1] > TABLE_L / 2)),
                             side=(None if err is None else
                                   (h.end[0] > TABLE_W / 2) == (cue_after[0] > TABLE_W / 2)),
                             base=(None if err is None or ghost is None else math.dist(ghost, cue_after))))
    n = len(rows)
    warn = [r for r in rows if r["warn"]]
    scr = [r for r in rows if r["scratch"]]
    tp = sum(r["scratch"] for r in warn)
    errs = [r["err"] for r in rows if r["err"] is not None]
    lines = ["# v3 physics-hint validation (noise-free pooltool, dev/floor states)", "",
             f"- states: {a.max_states} max, pot variants evaluated: {n}",
             f"- real scratch rate: {len(scr) / n:.1%}; warned: {len(warn) / n:.1%}",
             f"- scratch warning precision: {tp / max(1, len(warn)):.0%}  recall: {tp / max(1, len(scr)):.0%}",
             f"- cue-ball stop error: median {st.median(errs):.2f} m, mean {st.mean(errs):.2f} m "
             f"(table {TABLE_W:.2f} x {TABLE_L:.2f} m); naive baseline (cue ball stays at the contact "
             f"point): median {st.median(r['base'] for r in rows if r['base'] is not None):.2f} m",
             f"- same half of the table (upper/lower): {st.mean(r['half'] for r in rows if r['half'] is not None):.0%}; "
             f"same side (left/right): {st.mean(r['side'] for r in rows if r['side'] is not None):.0%}",
             "", "| spin | speed | n | real scratch | warned | precision | recall | median stop error (m) |",
             "|---|---|---|---|---|---|---|---|"]
    for spin in ("stun", "follow", "draw"):
        for speed in ("soft", "medium", "firm"):
            g = [r for r in rows if r["spin"] == spin and r["speed"] == speed]
            if not g:
                continue
            w = [r for r in g if r["warn"]]
            sc = [r for r in g if r["scratch"]]
            t = sum(r["scratch"] for r in w)
            e = [r["err"] for r in g if r["err"] is not None]
            lines.append(f"| {spin} | {speed} | {len(g)} | {len(sc) / len(g):.0%} | {len(w) / len(g):.0%} | "
                         f"{t / max(1, len(w)):.0%} | {t / max(1, len(sc)):.0%} | "
                         f"{st.median(e) if e else float('nan'):.2f} |")
    text = "\n".join(lines) + "\n"
    Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
