"""Model-size check (docs/PLAN_SCALE_14B.md): the three pre-registered metrics, 3B vs 14B.

    python experiments/analyze_scale14b.py

Reads results/v2_main (3B, zero playing noise only) and results/v2_scale14b (14B); raw shot logs
come from results/raw_logs_v2.zip and results/raw_logs_scale14b.zip. Seconds per decision use
only decisions with no cached LLM call (seed 9000 repeats the smoke test, and the games running
when the spot VM was preempted were replayed, so some 14B decisions came from the response cache).
"""

from __future__ import annotations

import json
import math
import random
import statistics as st
import zipfile
from collections import defaultdict
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = {"3B": ("v2_main", "raw_logs_v2.zip"), "14B": ("v2_scale14b", "raw_logs_scale14b.zip")}


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def boot_ci(x: list[int], reps: int = 10000, seed: int = 0) -> tuple[float, float]:
    r = random.Random(seed)
    m = sorted(st.mean(r.choices(x, k=len(x))) for _ in range(reps))
    return m[int(0.025 * reps)], m[int(0.975 * reps) - 1]


def sign_p(x: list[int]) -> float:
    pos, neg = sum(v > 0 for v in x), sum(v < 0 for v in x)
    n = pos + neg
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(pos, neg) + 1)) / 2 ** n)


def per_seed_wins(exp: str, system: str) -> dict[int, int]:
    """Wins of `system` against Stat per seed (0..2 over the two seat orders), zero noise."""
    w: dict[int, int] = defaultdict(int)
    n: dict[int, int] = defaultdict(int)
    for line in open(ROOT / "results" / exp / "games.jsonl", encoding="utf-8"):
        g = json.loads(line)
        seats = {v: k for k, v in g["players"].items()}
        if g["noise"] != "none" or set(seats) != {system, "stat"}:
            continue
        n[g["seed"]] += 1
        w[g["seed"]] += g["winner"] == seats[system]
    assert n and all(v == 2 for v in n.values()), "each seed needs both seat orders"
    return dict(w)


def decision_stats(exp: str, zip_name: str, system: str) -> dict[str, float]:
    calls, tin, tout, lat = [], [], [], []
    with zipfile.ZipFile(ROOT / "results" / zip_name) as z:
        for name in z.namelist():
            if not (name.startswith(f"results/{exp}/raw/") and "_none_" in name and name.endswith(".jsonl")):
                continue
            for line in z.read(name).decode("utf-8").splitlines():
                r = json.loads(line)
                if r["kind"] != "decision" or r["system"] != system:
                    continue
                u = r.get("usage") or {}
                calls.append(u.get("calls", 0))
                tin.append(u.get("input_tokens", 0))
                tout.append(u.get("output_tokens", 0))
                if u.get("cached_calls", 0) == 0:
                    lat.append(r["decision_latency_s"])
    return {"decisions": len(calls), "calls": st.mean(calls), "in": st.mean(tin), "out": st.mean(tout),
            "uncached": len(lat), "s_mean": st.mean(lat), "s_median": st.median(lat)}


def paired(a: dict[int, int], b: dict[int, int]) -> str:
    d = [a[s] - b[s] for s in sorted(a)]
    lo, hi = boot_ci(d)
    return (f"mean {st.mean(d):+.3f} wins/seed, bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}], "
            f"seeds better/equal/worse {sum(x > 0 for x in d)}/{sum(x == 0 for x in d)}/{sum(x < 0 for x in d)}, "
            f"sign test p = {sign_p(d):.3f}")


def main() -> None:
    wins = {(m, s): per_seed_wins(exp, s) for m, (exp, _) in RUNS.items() for s in ("sa", "ma")}

    print("1) Win rate against Stat (zero playing noise; 30 seeds x 2 seat orders = 60 games)")
    for (model, system), w in wins.items():
        k, n = sum(w.values()), 2 * len(w)
        lo, hi = wilson(k, n)
        print(f"   {model:>3} {system.upper()}: {k}/{n} = {k / n:.1%}, 95% Wilson CI [{lo:.1%}, {hi:.1%}]")

    print("\n2) Team effect, MA - SA paired per seed (wins in the two games against Stat)")
    for model in RUNS:
        print(f"   {model:>3}: {paired(wins[model, 'ma'], wins[model, 'sa'])}")
    print("   Model effect, 14B - 3B paired per seed (same system)")
    for system in ("sa", "ma"):
        print(f"   {system.upper():>3}: {paired(wins['14B', system], wins['3B', system])}")

    print("\n3) Cost per LLM decision (seconds: uncached decisions only; 3B on an RTX 4060 laptop, 14B on an L4)")
    for model, (exp, zip_name) in RUNS.items():
        for system in ("SA", "MA"):
            s = decision_stats(exp, zip_name, system)
            print(f"   {model:>3} {system}: {s['calls']:.2f} calls, {s['in']:.0f} in / {s['out']:.0f} out tokens, "
                  f"{s['s_mean']:.1f} s mean ({s['s_median']:.1f} s median); "
                  f"{s['decisions']} decisions, {s['uncached']} uncached")


if __name__ == "__main__":
    main()
