"""Run a paired tournament from a YAML config and write a summary table.

    python experiments/run_tournament.py --config experiments/configs/e0.yaml
    python experiments/run_tournament.py --config experiments/configs/e0.yaml --seeds 2 --workers 4  # quick check

Re-running the same command resumes: games already in results/<exp_id>/games.jsonl are skipped.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from cuemind.candidates.pipeline import PrepConfig
from cuemind.eval.metrics import pairing_summary, selfplay_summary
from cuemind.eval.tournament import build_jobs, run_tournament

ROOT = Path(__file__).resolve().parents[1]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None, help="override the number of seeds")
    ap.add_argument("--out", default=None, help="output directory (default results/<exp_id>)")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    exp_id = cfg["exp_id"]
    out = Path(a.out) if a.out else ROOT / "results" / exp_id
    seeds: list[int] = []
    if "seeds" in cfg:
        n_seeds = a.seeds or cfg["seeds"]["count"]
        seeds = list(range(cfg["seeds"]["start"], cfg["seeds"]["start"] + n_seeds))
    prep = PrepConfig(**cfg.get("prep", {}))
    pairings = [tuple(p) for p in cfg["pairings"]]
    workers = a.workers or cfg.get("workers", 8)

    out.mkdir(parents=True, exist_ok=True)
    (out / "config_used.json").write_text(json.dumps(
        {"config": cfg, "seeds": seeds, "git_commit": git_commit(), "prep": prep.to_dict()},
        indent=2), encoding="utf-8")

    rows = []
    for noise in cfg.get("noise_levels", ["medium"]):
        jobs = build_jobs(exp_id, pairings, seeds, noise, prep, cfg.get("max_shots", 150),
                          cfg.get("policy_kwargs"))
        summaries = run_tournament(jobs, out, workers=workers)
        for x, y in pairings:
            if x == y:
                r = selfplay_summary(summaries, x)
                if r:
                    print(f"[{noise}] {x} vs {x}: breaker (P1) wins {r['breaker_win_rate']:.0%} "
                          f"{r['breaker_win_ci']} of {r['games']} games, {r['mean_shots']:.1f} shots/game")
                continue
            for s, o in ((x, y), (y, x)):
                r = pairing_summary(summaries, s, o)
                if r:
                    rows.append({"noise": noise, **r})
    df = pd.DataFrame(rows)
    if df.empty:
        return 0
    df.to_csv(out / "summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        cols = ["noise", "system", "opponent", "games", "win_rate", "win_rate_ci", "paired_mean",
                "sign_test_p", "mean_shots", "pot_success", "fouls_per_game", "safety_rate",
                "errors", "shot_cap"]
        print(df[[c for c in cols if c in df]].round(3).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
