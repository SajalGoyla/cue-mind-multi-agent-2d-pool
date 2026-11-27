"""Regenerate every v2 and v3 table and figure from results/ in one command.

    python experiments/analyze.py

Reads results/v2_pilot, results/v2_floor and results/v2_main (games.jsonl + raw logs) and writes
    results/summary.md          all numbers, with the files they come from
    results/figures/*.png       win rate vs Stat by system and noise condition
    results/v2_main/moments.md  candidate moments from real games for the qualitative slides
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from cuemind.eval.metrics import game_rows, pairing_summary, selfplay_summary  # noqa: E402
from cuemind.eval.stats import bootstrap_mean_ci, sign_test  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"
COLOR = {"sa": "#eda100", "ma": "#008300", "b1": "#eb6834", "stat": "#2a78d6"}  # validated palette
LABEL = {"sa": "SA (single agent)", "ma": "MA (team)", "b1": "B1 (easiest pot)", "stat": "Stat",
         "rule": "Rule"}
# (experiment, reference player, title, figure) for every main run
CONDITIONS = [
    ("v2_main", "stat", "v2: forecasts for Stat only (SA/MA see geometry)", "v2_win_rates.png"),
    ("v3a_main", "rule", "v3-A: no forecasts for anyone (geometry + physics hints for all)", "v3a_win_rates.png"),
    ("v3b_main", "stat", "v3-B: forecasts for everyone", "v3b_win_rates.png"),
]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"


def load(exp: str) -> list[dict]:
    p = RES / exp / "games.jsonl"
    if not p.exists():
        return []
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [g for g in rows if g.get("status") != "error"]


def fmt_ci(ci) -> str:
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def paired_systems(games: list[dict], a: str, b: str, ref: str = "stat") -> dict:
    """Per seed, each system's score over its two games against `ref` (0..2); b minus a."""
    df = game_rows(games)
    sc = {}
    for s in (a, b):
        d = df[(df.system == s) & (df.opponent == ref)]
        full = d.groupby("seed").filter(lambda x: len(x) == 2)
        sc[s] = full.groupby("seed").score.sum()
    seeds = sorted(set(sc[a].index) & set(sc[b].index))
    diffs = [float(sc[b][s] - sc[a][s]) for s in seeds]
    if not diffs:
        return {}
    m, lo, hi = bootstrap_mean_ci(diffs)
    return {"seeds": len(seeds), "mean_diff": m, "ci": (lo, hi), "p": sign_test(diffs),
            "b_better": sum(d > 0 for d in diffs), "a_better": sum(d < 0 for d in diffs),
            "ties": sum(d == 0 for d in diffs)}


def section_pilot(md: list[str]) -> None:
    games = load("v2_pilot")
    if not games:
        return
    md += ["\n## Pilot: bots only (`results/v2_pilot/games.jsonl`)\n",
           "| playing noise | Stat vs B1 games | Stat win rate [95% CI] | pass (>= 60%, CI above 50%) | Stat vs Stat: breaker wins [95% CI] | shots/game |",
           "|---|---|---|---|---|---|"]
    for noise in ("none", "low"):
        g = [x for x in games if x.get("noise") == noise]
        r = pairing_summary(g, "stat", "b1")
        s = selfplay_summary(g, "stat")
        if r:
            ok = r["win_rate"] >= 0.60 and r["win_rate_ci"][0] > 0.50
            md.append(f"| {noise} | {r['games']} | {r['win_rate']:.2f} {fmt_ci(r['win_rate_ci'])} | "
                      f"{'PASS' if ok else 'fail'} | "
                      + (f"{s['breaker_win_rate']:.2f} {fmt_ci(s['breaker_win_ci'])} ({s['games']} games)" if s else "-")
                      + f" | {r['mean_shots']:.1f} |")


def section_floor(md: list[str]) -> None:
    games = load("v2_floor")
    if not games:
        return
    md += ["\n## Floor check: Stat vs SA, 10 games per noise level (`results/v2_floor/games.jsonl`)\n",
           "| playing noise | games | SA win rate [95% CI] | above 10% floor |", "|---|---|---|---|"]
    for noise in ("none", "low"):
        r = pairing_summary([x for x in games if x.get("noise") == noise], "sa", "stat")
        if r:
            md.append(f"| {noise} | {r['games']} | {r['win_rate']:.2f} {fmt_ci(r['win_rate_ci'])} | "
                      f"{'yes' if r['win_rate'] >= 0.10 else 'NO'} |")


def section_main(md: list[str], exp: str = "v2_main", ref: str = "stat",
                 title: str = "Main results", fig_name: str = "v2_win_rates.png") -> None:
    games = load(exp)
    if not games:
        return
    R = LABEL[ref]
    md.append(f"\n## {title} (`results/{exp}/games.jsonl`)")
    fig_rows = []
    for noise in ("low", "none"):
        g = [x for x in games if x.get("noise") == noise]
        if not g:
            continue
        md.append(f"\n### Playing noise: {noise}\n")
        s = selfplay_summary(g, ref)
        if s:
            md.append(f"Control, {R} vs {R}: the breaker wins {s['breaker_win_rate']:.0%} "
                      f"{fmt_ci(s['breaker_win_ci'])} of {s['games']} games ({s['mean_shots']:.1f} shots per game).\n")
        md += [f"| system | games vs {R} | win rate [95% CI] | pot success | fouls/game | balls/inning | "
               "shots/game | safety rate | fallback | calls/decision | in/out tokens per decision | "
               "latency s/decision | unanimous |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for sysname in ("sa", "ma"):
            r = pairing_summary(g, sysname, ref)
            if not r:
                continue
            fig_rows.append((noise, sysname, r["win_rate"], r["win_rate_ci"]))
            md.append(f"| {sysname} | {r['games']} | {r['win_rate']:.2f} {fmt_ci(r['win_rate_ci'])} | "
                      f"{r['pot_success']:.2f} | {r['fouls_per_game']:.2f} | {r['balls_per_inning']:.2f} | "
                      f"{r['mean_shots']:.1f} | {r['safety_rate']:.2f} | {r['fallback_rate']:.3f} | "
                      f"{r['calls_per_decision']:.2f} | {r['input_tokens_per_decision']:.0f} / "
                      f"{r['output_tokens_per_decision']:.0f} | {r['latency_s_per_decision']:.1f} | "
                      + (f"{r['unanimous_rate']:.0%}" if sysname == "ma" else "-") + " |")
        p = paired_systems(g, "sa", "ma", ref)
        if p:
            md.append(f"\nPaired comparison per seed (score = wins in the two games against {R}, MA minus SA): "
                      f"mean {p['mean_diff']:+.2f} wins [95% CI {p['ci'][0]:+.2f}, {p['ci'][1]:+.2f}] over "
                      f"{p['seeds']} seeds; MA better on {p['b_better']}, SA better on {p['a_better']}, "
                      f"ties {p['ties']}; sign test p = {p['p']:.3g}.")
    if fig_rows:
        FIG.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6.2, 3.2))
        noises = [n for n in ("low", "none") if any(r[0] == n for r in fig_rows)]
        for (noise, sysname, wr, ci) in fig_rows:
            x = noises.index(noise) + (-0.12 if sysname == "sa" else 0.12)
            ax.errorbar(x, wr, yerr=[[wr - ci[0]], [ci[1] - wr]], fmt="o", color=COLOR[sysname],
                        ecolor=COLOR[sysname], elinewidth=2, capsize=0, markersize=8,
                        label=LABEL[sysname] if noise == noises[0] else None)
            ax.text(x + 0.05, wr, f"{wr:.0%}", va="center", fontsize=9, color=INK)
        ax.axhline(0.5, color=MUTED, linewidth=1, linestyle="--")
        ax.set_xticks(range(len(noises)), [f"playing noise: {n}" for n in noises])
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"win rate against {R} (95% Wilson CI)")
        ax.grid(axis="y", color=GRID)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=9, loc="upper right")
        ax.set_title(title.split(":")[0] + f": full games against {R}", loc="left", fontsize=11)
        fig.savefig(FIG / fig_name, dpi=180, bbox_inches="tight", facecolor="#fcfcfb")
        plt.close(fig)
        md.append(f"\nFigure: `results/figures/{fig_name}`")

def shot_quality(exp: str) -> dict:
    """Per (noise, system): pot success of attempted pots, scratch and foul rate per decision,
    safety share, and (when forecasts exist) agreement with Stat's pick on the same menu."""
    raw = RES / exp / "raw"
    out: dict = {}
    if not raw.exists():
        return out
    from cuemind.candidates.model import Candidate
    from cuemind.policies.search_bot import SearchWeights, expected_value

    w = SearchWeights()
    for f in sorted(raw.glob("*.jsonl")):
        noise = "none" if "_none_" in f.name else "low"
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("kind") != "decision" or r["system"] not in ("SA", "MA"):
                continue
            d = out.setdefault((noise, r["system"].lower()), {"n": 0, "pots": 0, "made": 0, "scratch": 0,
                                                              "foul": 0, "safety": 0, "agree": 0, "agree_n": 0})
            o = r["outcome"]
            cands = r["candidates"]
            ch = next(c for c in cands if c["shot_id"] == r["decision"]["shot_id"])
            d["n"] += 1
            d["foul"] += bool(o.get("foul"))
            d["scratch"] += "scratch" in (o.get("foul") or "")
            if ch["shot_type"] == "pot":
                d["pots"] += 1
                d["made"] += bool(o.get("called_ball_made"))
            else:
                d["safety"] += 1
            if cands and cands[0].get("rollouts"):
                cs = [Candidate.model_validate(c) for c in cands]
                best = max(cs, key=lambda c: (expected_value(c, w), c.shot_id))
                d["agree"] += best.shot_id == ch["shot_id"]
                d["agree_n"] += 1
    return out


def section_compare(md: list[str]) -> None:
    rows = []
    for exp, ref, title, _ in CONDITIONS:
        for (noise, s), d in sorted(shot_quality(exp).items()):
            if d["n"]:
                rows.append((title.split(":")[0], noise, s, d))
    if not rows:
        return
    md += ["\n## Across conditions: how well SA and MA play (shot level, all decisions)\n",
           "Opponents differ between conditions (Stat or Rule), so compare win rates only within a "
           "condition; these shot-level numbers describe the agents' own play.\n",
           "| condition | playing noise | system | decisions | pot success | scratch / decision | "
           "foul / decision | safety share | same pick as Stat's objective |",
           "|---|---|---|---|---|---|---|---|---|"]
    for cond, noise, s, d in rows:
        agree = f"{d['agree'] / d['agree_n']:.0%}" if d["agree_n"] else "-"
        md.append(f"| {cond} | {noise} | {s.upper()} | {d['n']} | {d['made'] / max(1, d['pots']):.2f} | "
                  f"{d['scratch'] / d['n']:.3f} | {d['foul'] / d['n']:.3f} | {d['safety'] / d['n']:.2f} | {agree} |")


def section_hints(md: list[str]) -> None:
    p = RES / "v3_hint_validation.md"
    if p.exists():
        md.append("\n## Physics-hint accuracy (v3-A; `results/v3_hint_validation.md`)\n")
        md += [x for x in p.read_text(encoding="utf-8").splitlines()[2:7] if x.strip()]



def moments(max_items: int = 12) -> None:
    """Candidate moments from real MA games (coordinator decisions with clear outcomes)."""
    raw = sorted((RES / "v2_main" / "raw").glob("*_ma_*.jsonl")) if (RES / "v2_main" / "raw").exists() else []
    good, bad = [], []
    for f in raw:
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("kind") != "decision" or r["system"] != "MA" or (r.get("trace") or {}).get("unanimous"):
                continue
            o = r["outcome"]
            item = (f.stem, r["shot_index"], r["decision"]["shot_id"], r["decision"]["rationale"],
                    r["trace"].get("shortlists", {}), o)
            if o.get("called_ball_made") and not o.get("turn_over"):
                good.append(item)
            elif o.get("foul") or o.get("game_over"):
                bad.append(item)
    if not good and not bad:
        return
    lines = ["# Candidate moments from MA games (for the qualitative slides)", "",
             "Non-unanimous MA decisions; pick 3-4 and check them in the demo's Replay tab.", ""]
    for title, items in (("Team decision worked (called ball potted, turn kept)", good),
                         ("Team decision went wrong (foul or game lost)", bad)):
        lines += [f"## {title}", ""]
        for game, k, sid, why, sl, o in items[:max_items // 2]:
            noms = "; ".join(f"{n}: {', '.join(p['shot_id'] for p in picks)}" for n, picks in sl.items())
            lines.append(f"- `{game}` shot {k}: chose {sid} ({o.get('foul') or 'potted ' + ','.join(o.get('potted', {}))}). "
                         f"Nominations: {noms}. Rationale: {why}")
        lines.append("")
    (RES / "v2_main" / "moments.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    md = ["# CueMind results (v2 and v3)", "",
          "Generated by `python experiments/analyze.py`; do not edit by hand. v2: Stat simulates "
          "every option, SA and MA see geometry only (D-029). v3-A: nobody sees forecasts; all "
          "players read geometry + simulation-free physics hints, reference player Rule (D-031). "
          "v3-B: everybody sees the forecasts, reference player Stat (D-031)."]
    section_pilot(md)
    section_floor(md)
    for exp, ref, title, fig in CONDITIONS:
        section_main(md, exp, ref, title, fig)
    section_compare(md)
    section_hints(md)
    moments()
    RES.mkdir(exist_ok=True)
    (RES / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
