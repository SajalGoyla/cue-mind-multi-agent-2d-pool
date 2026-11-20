"""Decision-level experiments on the positions benchmark (E1, E4, E6, E7, E10).

    python experiments/run_positions.py --config experiments/configs/e1.yaml
    python experiments/run_positions.py --config experiments/configs/e1.yaml --limit 5 --llm fake   # offline check

Every (position, system, access level, repeat) is decided once and scored against the
oracle. Results are appended to results/<exp_id>/decisions.jsonl (resumable) and a summary
by system / access level / difficulty / situation tag is written to summary.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from cuemind.eval.positions import load_positions, menu_at_level, score_choice
from cuemind.llm.cache import BudgetExceeded
from cuemind.policies.base import DecisionContext
from cuemind.policies.registry import make_policy

ROOT = Path(__file__).resolve().parents[1]


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["system", "access_level"]
    for (sysname, lvl), d in df.groupby(keys):
        base = {"system": sysname, "access_level": lvl, "n": len(d),
                "regret": d.regret.mean(), "top1": d.top1.mean(), "normalised": d.normalised.mean(),
                "fallback_rate": d.fallback.mean(), "safety_rate": (d.intent == "safety").mean(),
                "calls_per_decision": d.calls.mean(), "cost_usd_per_decision": d.cost_usd.mean(),
                "latency_s": d.latency_s.mean(), "latency_p95_s": d.latency_s.quantile(0.95)}
        for diff in ("easy", "medium", "hard"):
            sub = d[d.difficulty == diff]
            base[f"regret_{diff}"] = sub.regret.mean() if len(sub) else float("nan")
        for tag in ("offense", "position", "defense"):
            sub = d[d.tags.apply(lambda t, tag=tag: tag in t)]
            base[f"regret_{tag}"] = sub.regret.mean() if len(sub) else float("nan")
        rows.append(base)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None, help="only the first N positions")
    ap.add_argument("--llm", default=None, help="override: fake | claude")
    ap.add_argument("--max-usd", type=float, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    exp_id = cfg["exp_id"]
    out = Path(a.out) if a.out else ROOT / "results" / exp_id
    out.mkdir(parents=True, exist_ok=True)
    positions = []
    for p in cfg["positions"]:
        positions += load_positions(ROOT / p)
    if a.limit:
        positions = positions[: a.limit]
    llm_kw = dict(cfg.get("llm", {}))
    if a.llm:
        llm_kw["llm"] = a.llm
    llm_kw.setdefault("run_id", exp_id)
    if a.max_usd is not None:
        llm_kw["max_usd"] = a.max_usd
    repeats = cfg.get("repeats", 1)
    levels = cfg.get("access_levels", ["L2"])
    policies = {name: make_policy(name, **llm_kw) for name in cfg["systems"]}

    path = out / "decisions.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["position_id"], r["system_key"], r["access_level"], r["repeat"]))
    jobs = [(rec, name, lvl, rep) for rec in positions for name in cfg["systems"] for lvl in levels
            for rep in range(repeats) if (rec["position_id"], name, lvl, rep) not in done]
    print(f"{len(jobs)} decisions to run ({len(done)} already done)", flush=True)

    def run(job):
        rec, name, lvl, rep = job
        state, cands = menu_at_level(rec, lvl)
        ctx = DecisionContext(state=state, candidates=cands, sample_idx=rep,
                              request_tag=f"{rec['position_id']}:{rep}")
        policy = policies[name]
        t0 = time.perf_counter()
        d = policy.decide(ctx)
        score = score_choice(rec, d.shot_id)
        return {"exp_id": exp_id, "position_id": rec["position_id"], "system_key": name,
                "system": getattr(policy, "name", name), "access_level": lvl, "repeat": rep,
                "difficulty": rec["difficulty"], "tags": rec["tags"],
                "shot_id": d.shot_id, "intent": d.intent, "rationale": d.rationale,
                "confidence": d.confidence, "fallback": d.fallback, **score,
                "calls": d.usage.get("calls", 0), "cost_usd": d.usage.get("cost_usd", 0.0),
                "input_tokens": d.usage.get("input_tokens", 0),
                "output_tokens": d.usage.get("output_tokens", 0),
                "cache_read_tokens": d.usage.get("cache_read_tokens", 0),
                "latency_s": d.usage.get("latency_s", time.perf_counter() - t0),
                "trace": d.trace}

    threads = a.threads or cfg.get("threads", 4)
    stopped = None
    with ThreadPoolExecutor(max_workers=threads) as pool, open(path, "a", encoding="utf-8") as f:
        futs = [pool.submit(run, j) for j in jobs]
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                row = fut.result()
            except BudgetExceeded as exc:
                stopped = str(exc)
                for other in futs:
                    other.cancel()
                break
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()
            if k % 25 == 0:
                print(f"  {k}/{len(jobs)}", flush=True)
    if stopped:
        print("STOPPED:", stopped, flush=True)

    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    df = pd.DataFrame(rows)
    if not df.empty:
        summ = summarise(df)
        summ.to_csv(out / "summary.csv", index=False)
        with pd.option_context("display.width", 220, "display.max_columns", 30):
            print(summ.round(3).to_string(index=False))
    return 1 if stopped else 0


if __name__ == "__main__":
    sys.exit(main())
