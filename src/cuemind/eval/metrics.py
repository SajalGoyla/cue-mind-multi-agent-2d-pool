"""Per-game, per-seat metrics computed from the runner's shot records, and tournament
summaries computed from per-game summaries."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from cuemind.eval.stats import bootstrap_mean_ci, sign_test, wilson_ci

USAGE_KEYS = ("calls", "input_tokens", "output_tokens", "latency_s")


def seat_metrics(records: list[dict]) -> dict[str, dict[str, Any]]:
    """Shot-level statistics for each seat in one game."""
    out: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(float))
    decisions = [r for r in records if r["kind"] == "decision"]
    prev_seat = None
    for r in records:  # an inning (turn) starts whenever the shooter changes
        if r["kind"] == "decision" and r["seat"] != prev_seat:
            out[r["seat"]]["innings"] += 1
        prev_seat = r["seat"]
    for r in decisions:
        if (r.get("trace") or {}).get("unanimous"):
            out[r["seat"]]["unanimous"] += 1
    for i, r in enumerate(decisions):
        m = out[r["seat"]]
        m["decisions"] += 1
        chosen = next(c for c in r["candidates"] if c["shot_id"] == r["decision"]["shot_id"])
        o = r["outcome"]
        if chosen["shot_type"] == "pot":
            m["pot_attempts"] += 1
            m["pots_made"] += int(o["called_ball_made"] and o["legal"])
        else:
            m["safeties"] += 1
            nxt = next((d for d in decisions[i + 1:]), None)
            if nxt is not None and nxt["seat"] != r["seat"]:
                m["safety_successes"] += int(not (nxt["outcome"]["called_ball_made"]
                                                  and nxt["outcome"]["legal"]))
        m["fouls"] += int(o["foul"] is not None)
        m["scratches"] += int((o["foul"] or "").startswith("scratch"))
        m["balls_potted"] += len([b for b in o["potted"] if b != "8"])
        m["fallbacks"] += int(bool(r.get("fallback")))
        m["ball_in_hand"] += int(bool(r.get("ball_in_hand")))
        m["decision_latency_s"] += float(r.get("decision_latency_s") or 0.0)
        for k in USAGE_KEYS:
            m[k] += float((r.get("usage") or {}).get(k, 0.0))
    return {seat: dict(m) for seat, m in out.items()}


def game_rows(summaries: list[dict]) -> pd.DataFrame:
    """One row per (game, seat) with the seat's system, score and shot statistics."""
    rows = []
    for g in summaries:
        for seat, system in g["players"].items():
            score = 0.5 if g["winner"] is None else float(g["winner"] == seat)
            rows.append({"game_id": g["game_id"], "seed": g["seed"], "seat": seat,
                         "system": system, "opponent": g["players"]["P2" if seat == "P1" else "P1"],
                         "breaks": seat == g.get("breaker", "P1"), "score": score,
                         "status": g["status"], "n_shots": g["n_shots"],
                         **g.get("seat_metrics", {}).get(seat, {})})
    return pd.DataFrame(rows)


def pairing_summary(summaries: list[dict], system: str, opponent: str) -> dict[str, Any]:
    """Head-to-head summary for `system` against `opponent` over paired seeds."""
    df = game_rows(summaries)
    d = df[(df.system == system) & (df.opponent == opponent)]
    if d.empty:
        return {}
    n = len(d)
    wins = float(d.score.sum())
    lo, hi = wilson_ci(wins, n)
    per_seed = d.groupby("seed").score.sum()  # 0..2 when both seat orders were played
    complete = d.groupby("seed").size() == 2
    paired = (per_seed[complete] - 1.0).tolist()  # >0 means system won the seed pair
    mean, blo, bhi = bootstrap_mean_ci(paired) if paired else (float("nan"),) * 3
    tot = lambda col: float(d[col].sum()) if col in d else 0.0  # noqa: E731
    return {
        "system": system, "opponent": opponent, "games": n, "score": wins,
        "win_rate": wins / n, "win_rate_ci": (round(lo, 3), round(hi, 3)),
        "paired_seeds": len(paired), "paired_mean": mean, "paired_ci": (blo, bhi),
        "sign_test_p": sign_test(paired) if paired else float("nan"),
        "errors": int((d.status == "error").sum()), "shot_cap": int((d.status == "shot_cap").sum()),
        "mean_shots": float(d.n_shots.mean()),
        "pot_success": tot("pots_made") / max(tot("pot_attempts"), 1),
        "fouls_per_game": tot("fouls") / n, "scratches_per_game": tot("scratches") / n,
        "safety_rate": tot("safeties") / max(tot("decisions"), 1),
        "safety_success": tot("safety_successes") / max(tot("safeties"), 1),
        "balls_per_game": tot("balls_potted") / n,
        "fallback_rate": tot("fallbacks") / max(tot("decisions"), 1),
        "balls_per_inning": tot("balls_potted") / max(tot("innings"), 1),
        "calls_per_decision": tot("calls") / max(tot("decisions"), 1),
        "input_tokens_per_decision": tot("input_tokens") / max(tot("decisions"), 1),
        "output_tokens_per_decision": tot("output_tokens") / max(tot("decisions"), 1),
        "latency_s_per_decision": tot("decision_latency_s") / max(tot("decisions"), 1),
        "unanimous_rate": tot("unanimous") / max(tot("decisions"), 1),
    }


def selfplay_summary(summaries: list[dict], system: str) -> dict[str, Any]:
    """Self-pairing (one game per seed): how often the breaker (seat P1) wins."""
    gs = [g for g in summaries if g["players"]["P1"] == system == g["players"]["P2"]]
    if not gs:
        return {}
    n = len(gs)
    p1 = sum(1.0 if g["winner"] == "P1" else 0.5 if g["winner"] is None else 0.0 for g in gs)
    lo, hi = wilson_ci(p1, n)
    return {"system": system, "games": n, "breaker_win_rate": p1 / n,
            "breaker_win_ci": (round(lo, 3), round(hi, 3)),
            "mean_shots": sum(g["n_shots"] for g in gs) / n,
            "errors": sum(g["status"] == "error" for g in gs)}
