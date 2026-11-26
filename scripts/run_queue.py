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
    ("v2_floor", [PY, "experiments/run_tournament.py", "--config", "experiments/configs/v2_floor.yaml"]),
    ("v2_dev", [PY, "experiments/run_tournament.py", "--config", "experiments/configs/v2_dev.yaml"]),
    ("v2_main", [PY, "experiments/run_tournament.py", "--config", "experiments/configs/v2_main.yaml"]),
    ("analyze", [PY, "experiments/analyze.py"]),
]


def keep_awake() -> None:
    """Ask Windows not to sleep while the queue runs (no settings are changed; the request
    ends with the process). A sleeping laptop stopped an earlier run (2026-11-21)."""
    try:
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    except (AttributeError, OSError):
        pass


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
                    help="worker processes per tournament (memory-bound; D-026)")
    ap.add_argument("--watchdog-mb", type=float, default=3500,
                    help="restart Ollama when its runner exceeds this RSS (0 = off; D-028)")
    a = ap.parse_args()
    keep_awake()
    if a.watchdog_mb > 0:
        sys.path.insert(0, str(Path(__file__).parent))
        from ollama_watchdog import Watchdog

        Watchdog(a.watchdog_mb, period_s=60, parallel=a.concurrency, log=log).start()
        log(f"Ollama watchdog on (limit {a.watchdog_mb:.0f} MB)")
    if a.wait_for:
        run_log = ROOT / "results" / f"{a.wait_for}_run.log"
        log(f"waiting for {run_log.name} to finish")
        while not (run_log.exists() and "access_level" in run_log.read_text(encoding="utf-8", errors="replace")):
            time.sleep(30)
    steps = [s for s in QUEUE if not a.only or s[0] in a.only.split(",")]
    for name, cmd in steps:
        if "run_tournament.py" in cmd[1]:
            cmd = cmd + ["--workers", str(a.concurrency)]
        log(f"start {name}: {' '.join(cmd[1:])}")
        t0 = time.time()
        with open(ROOT / "results" / f"{name}_run.log", "a", encoding="utf-8") as out:
            rc = subprocess.call(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
        log(f"end {name}: exit {rc} after {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
