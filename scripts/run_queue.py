"""Run experiments one after another (they share one GPU), resumably.

    python scripts/run_queue.py                       # default queue after E1
    python scripts/run_queue.py --wait-for e1          # first wait until results/e1 is complete

Each step is an ordinary experiment command; finished work is skipped on re-run because
every experiment script resumes from its own results file. Progress goes to
results/queue.log.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

QUEUE = [
    ("e4", [PY, "experiments/run_positions.py", "--config", "experiments/configs/e4.yaml"]),
    ("e10", [PY, "experiments/run_positions.py", "--config", "experiments/configs/e10.yaml", "--limit", "30"]),
    ("e2", [PY, "experiments/run_tournament.py", "--config", "experiments/configs/e2.yaml"]),
    ("e5_low", [PY, "experiments/run_positions.py", "--config", "experiments/configs/e5_low.yaml"]),
    ("e5_high", [PY, "experiments/run_positions.py", "--config", "experiments/configs/e5_high.yaml"]),
    ("e6", [PY, "experiments/run_positions.py", "--config", "experiments/configs/e6.yaml"]),
    ("e3", [PY, "experiments/run_tournament.py", "--config", "experiments/configs/e3.yaml"]),
    ("e9", [PY, "experiments/run_positions.py", "--config", "experiments/configs/e9.yaml"]),
    ("analyze", [PY, "experiments/analyze.py"]),
]


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(ROOT / "results" / "queue.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-for", default=None, help="experiment whose run log must finish first")
    ap.add_argument("--only", default=None, help="comma-separated subset of step names")
    ap.add_argument("--concurrency", type=int, default=2,
                    help="threads per positions run / workers per tournament (memory-bound; D-026)")
    a = ap.parse_args()
    if a.wait_for:
        run_log = ROOT / "results" / f"{a.wait_for}_run.log"
        log(f"waiting for {run_log.name} to finish")
        while not (run_log.exists() and "access_level" in run_log.read_text(encoding="utf-8", errors="replace")):
            time.sleep(30)
    steps = [s for s in QUEUE if not a.only or s[0] in a.only.split(",")]
    for name, cmd in steps:
        if "run_positions.py" in cmd[1]:
            cmd = cmd + ["--threads", str(a.concurrency)]
        elif "run_tournament.py" in cmd[1]:
            cmd = cmd + ["--workers", str(a.concurrency)]
        log(f"start {name}: {' '.join(cmd[1:])}")
        t0 = time.time()
        with open(ROOT / "results" / f"{name}_run.log", "a", encoding="utf-8") as out:
            rc = subprocess.call(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
        log(f"end {name}: exit {rc} after {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
