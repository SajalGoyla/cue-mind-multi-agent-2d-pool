"""Regenerate every results table and figure from results/ in one command.

    python experiments/analyze.py

Reads results/<exp>/games.jsonl for every experiment and writes:
    results/summary.md          headline numbers with uncertainty and their source files
    results/figures/<exp>_win_rates.png
Every number in the report and slides should come from these outputs.

Per experiment and noise level:
  1. head-to-head table for every pairing (win rate with Wilson 95% CI, paired seed score,
     sign test, game and shot statistics, LLM cost);
  2. how games ended (legal 8, 8 with a foul, 8 early or in the wrong pocket, shot cap);
  3. SA vs MA through the common opponent B: for each seed both systems played twice against
     B, so the difference of their seed scores is a paired comparison (bootstrap CI, sign test).
"""

from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from cuemind.eval.metrics import game_rows, pairing_summary  # noqa: E402
from cuemind.eval.stats import bootstrap_mean_ci, sign_test  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
LABEL = {"b": "B (simulation)", "b1": "B1 (geometric)", "sa": "SA single agent",
         "ma": "MA multi-agent"}
COLOR = {"b": "#1baf7a", "b1": "#eb6834", "sa": "#eda100", "ma": "#008300"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"


def load_games(path: Path) -> list[dict]:
    games = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [g for g in games if g.get("status") != "error"]


def head_to_head(md: list[str], games: list[dict]) -> pd.DataFrame:
    rows = []
    for a, b in sorted({tuple(sorted(g["players"].values())) for g in games}):
        for s, o in ((a, b),) if a == b else ((a, b), (b, a)):
            r = pairing_summary(games, s, o)
            if r:
                rows.append(r)
    df = pd.DataFrame(rows)
    md += ["| system | vs | games | win rate [95% CI] | paired seed score [95% CI] | sign-test p | "
           "shots/game | pot success | balls/inning | fouls/game | safety rate | fallback | "
           "calls/dec | tokens/dec | latency s/dec |", "|" + "---|" * 15]
    for r in df.itertuples():
        selfplay = r.system == r.opponent
        paired = "(breaker win rate)" if selfplay else (
            f"{r.paired_mean:+.2f} [{r.paired_ci[0]:+.2f}, {r.paired_ci[1]:+.2f}]")
        p = "" if selfplay else f"{r.sign_test_p:.3g}"
        md.append(f"| {r.system} | {r.opponent} | {r.games} | {r.win_rate:.2f} "
                  f"[{r.win_rate_ci[0]:.2f}, {r.win_rate_ci[1]:.2f}] | {paired} | {p} | "
                  f"{r.mean_shots:.1f} | {r.pot_success:.2f} | {r.balls_per_inning:.2f} | "
                  f"{r.fouls_per_game:.2f} | {r.safety_rate:.2f} | {r.fallback_rate:.2f} | "
                  f"{r.calls_per_decision:.1f} | {r.tokens_per_decision:.0f} | "
                  f"{r.latency_s_per_decision:.1f} |")
    return df


def endings(md: list[str], games: list[dict]) -> None:
    md += ["\nHow games ended:\n", "| pairing | games | legal 8 | 8 with foul | 8 early or wrong pocket | "
           "shot cap |", "|---|---|---|---|---|---|"]
    by: dict[str, Counter] = {}
    for g in games:
        key = " vs ".join(sorted(g["players"].values()))
        by.setdefault(key, Counter())[g.get("ending") or g["status"]] += 1
    for key, c in sorted(by.items()):
        n = sum(c.values())
        md.append(f"| {key} | {n} | {c['legal 8']} | {c['8 with foul']} | "
                  f"{c['8 early or wrong pocket']} | {c['shot_cap']} |")


def sa_vs_ma(md: list[str], games: list[dict]) -> None:
    df = game_rows(games)
    df = df[df.opponent == "b"]
    seed_scores = {}
    for system in ("sa", "ma"):
        d = df[df.system == system]
        full = d.groupby("seed").filter(lambda x: len(x) == 2)
        seed_scores[system] = full.groupby("seed").score.sum()
    if any(s.empty for s in seed_scores.values()):
        return
    common = seed_scores["sa"].index.intersection(seed_scores["ma"].index)
    if not len(common):
        return
    diff = (seed_scores["ma"][common] - seed_scores["sa"][common]).tolist()
    m, lo, hi = bootstrap_mean_ci(diff)
    md += ["\nMA vs SA through the common opponent B (per seed: MA's score in its two games against "
           "B minus SA's; positive = MA better):\n",
           "| seeds | mean difference [95% CI] | MA better / equal / worse | sign-test p |",
           "|---|---|---|---|",
           f"| {len(diff)} | {m:+.2f} [{lo:+.2f}, {hi:+.2f}] | {sum(d > 0 for d in diff)} / "
           f"{sum(d == 0 for d in diff)} / {sum(d < 0 for d in diff)} | {sign_test(diff):.3g} |"]


def figure(exp: str, noise: str, df: pd.DataFrame) -> str | None:
    vs_b = df[(df.opponent == "b") & (df.system != "b")]
    if vs_b.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 0.6 * len(vs_b) + 1.2))
    for i, r in enumerate(vs_b.itertuples()):
        c = COLOR.get(r.system, "#2a78d6")
        ax.errorbar(r.win_rate, i, xerr=[[r.win_rate - r.win_rate_ci[0]], [r.win_rate_ci[1] - r.win_rate]],
                    fmt="o", color=c, ecolor=c, elinewidth=2, capsize=0, markersize=8)
        ax.text(r.win_rate_ci[1] + 0.02, i, f"{r.win_rate:.0%} ({r.games} games)", va="center",
                color=INK, fontsize=9)
    ax.set_yticks(range(len(vs_b)))
    ax.set_yticklabels([LABEL.get(s, s) for s in vs_b.system])
    ax.axvline(0.5, color=MUTED, linewidth=1, linestyle="--")
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("win rate against B (95% Wilson CI; both seat orders)")
    ax.invert_yaxis()
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID)
    ax.set_title(f"{exp}, noise {noise}: win rate against B", loc="left", color=INK, fontsize=11)
    FIG.mkdir(parents=True, exist_ok=True)
    name = f"{exp}_{noise}_win_rates.png"
    fig.savefig(FIG / name, dpi=180, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    return f"results/figures/{name}"


def main() -> None:
    md = ["# Results summary", "",
          "Generated by `python experiments/analyze.py` from the files named in each section. "
          "Do not edit by hand."]
    for path in sorted(glob.glob(str(ROOT / "results" / "*" / "games.jsonl"))):
        exp = Path(path).parent.name
        games = load_games(Path(path))
        for noise in sorted({g.get("noise", "none") for g in games}):
            gs = [g for g in games if g.get("noise", "none") == noise]
            md.append(f"\n## {exp}, noise `{noise}` (`results/{exp}/games.jsonl`)\n")
            df = head_to_head(md, gs)
            endings(md, gs)
            sa_vs_ma(md, gs)
            fig = figure(exp, noise, df)
            if fig:
                md.append(f"\nFigure: `{fig}`")
    out = ROOT / "results" / "summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
