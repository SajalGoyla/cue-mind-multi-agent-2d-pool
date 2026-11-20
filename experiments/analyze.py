"""Regenerate every results table and figure from results/ in one command.

    python experiments/analyze.py

Reads whatever exists among results/e0*/games.jsonl (tournaments) and
results/<exp>/decisions.jsonl (positions experiments) and writes:
    results/figures/*.png     report/slide figures
    results/summary.md        headline numbers with uncertainty and their source files
Every number in the report and slides should come from these outputs.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cuemind.eval.metrics import pairing_summary  # noqa: E402
from cuemind.eval.stats import bootstrap_mean_ci, holm, wilcoxon_paired  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"

# Fixed colour per system (validated categorical palette; colour follows the entity).
SYSTEM_ORDER = ["b0", "b1", "b2", "sa", "sa-vote", "ma", "ma+critic",
                "ma-no-offense", "ma-no-position", "ma-no-safety"]
SYSTEM_COLOR = {"b0": "#2a78d6", "b1": "#eb6834", "b2": "#1baf7a", "sa": "#eda100",
                "sa-vote": "#e87ba4", "ma": "#008300", "ma+critic": "#6250d6",
                "ma-no-offense": "#e34948", "ma-no-position": "#e34948", "ma-no-safety": "#e34948"}
LABEL = {"b0": "B0 random", "b1": "B1 geometric", "b2": "B2 search", "sa": "SA single agent",
         "sa-vote": "SA-vote (4 samples)", "ma": "MA multi-agent", "ma+critic": "MA + critic",
         "ma-no-offense": "MA w/o offense", "ma-no-position": "MA w/o position",
         "ma-no-safety": "MA w/o safety"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    return f"results/figures/{name}.png"


# ----------------------------------------------------------------------------- tournaments
def tournaments(md: list[str]) -> None:
    for path in sorted(glob.glob(str(ROOT / "results" / "*" / "games.jsonl"))):
        exp = Path(path).parent.name
        games = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
        games = [g for g in games if g.get("status") != "error"]
        if not games:
            continue
        rows = []
        for noise in sorted({g.get("noise", "medium") for g in games}):
            gs = [g for g in games if g.get("noise", "medium") == noise]
            pairs = {tuple(sorted(g["players"].values())) for g in gs}
            for a, b in sorted(pairs):
                for s, o in ((a, b), (b, a)):
                    r = pairing_summary(gs, s, o)
                    if r:
                        rows.append({"noise": noise, **r})
        df = pd.DataFrame(rows)
        if df.empty:
            continue
        df.to_csv(ROOT / "results" / exp / "summary.csv", index=False)
        md.append(f"\n## {exp}: tournaments (`results/{exp}/games.jsonl`)\n")
        md.append("| noise | system | vs | games | win rate [95% CI] | paired seed score [95% CI] | sign-test p | shots/game | pot success | fouls/game | safety rate |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in df.itertuples():
            md.append(f"| {r.noise} | {r.system} | {r.opponent} | {r.games} | {r.win_rate:.2f} "
                      f"[{r.win_rate_ci[0]:.2f}, {r.win_rate_ci[1]:.2f}] | {r.paired_mean:+.2f} "
                      f"[{r.paired_ci[0]:+.2f}, {r.paired_ci[1]:+.2f}] | {r.sign_test_p:.3g} | "
                      f"{r.mean_shots:.1f} | {r.pot_success:.2f} | {r.fouls_per_game:.2f} | {r.safety_rate:.2f} |")
        md.append(f"\nErrors: {int(df.errors.sum()) // 2} games; shot-cap draws: {int(df.shot_cap.sum()) // 2}.")

        # Figure: win rate with Wilson CI, one row per (system vs opponent), first listed system.
        seen, keep = set(), []
        rank = {s: i for i, s in enumerate(SYSTEM_ORDER)}
        ordered = sorted(df.itertuples(), key=lambda r: (r.noise, -rank.get(r.system, 99)))
        for r in ordered:
            key = (r.noise, frozenset((r.system, r.opponent)))
            if key not in seen:
                seen.add(key)
                keep.append(r)
        fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(keep) + 1.2))
        for i, r in enumerate(keep):
            c = SYSTEM_COLOR.get(r.system, "#2a78d6")
            ax.errorbar(r.win_rate, i, xerr=[[r.win_rate - r.win_rate_ci[0]], [r.win_rate_ci[1] - r.win_rate]],
                        fmt="o", color=c, ecolor=c, elinewidth=2, capsize=0, markersize=8)
            ax.text(r.win_rate_ci[1] + 0.02, i, f"{r.win_rate:.0%}", va="center", color=INK, fontsize=9)
        ax.set_yticks(range(len(keep)))
        ax.set_yticklabels([f"{LABEL.get(r.system, r.system)} vs {LABEL.get(r.opponent, r.opponent)}"
                            + (f" ({r.noise})" if df.noise.nunique() > 1 else "") for r in keep])
        ax.axvline(0.5, color=MUTED, linewidth=1, linestyle="--")
        ax.set_xlim(0, 1.08)
        ax.set_xlabel("win rate of the first-named system (95% Wilson CI; both seat orders)")
        ax.invert_yaxis()
        style(ax)
        ax.set_title(f"{exp}: head-to-head win rates", loc="left", color=INK, fontsize=11)
        md.append(f"\nFigure: `{save(fig, f'{exp}_win_rates')}`")


# ----------------------------------------------------------------------------- positions
def positions(md: list[str]) -> None:
    for path in sorted(glob.glob(str(ROOT / "results" / "*" / "decisions.jsonl"))):
        exp = Path(path).parent.name
        df = pd.DataFrame([json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()
                           if x.strip()])
        if df.empty:
            continue
        md.append(f"\n## {exp}: decision-level results (`results/{exp}/decisions.jsonl`)\n")
        md.append("| system | info | n | mean regret [95% CI] | top-1 | normalised | fallback | calls | $/decision | latency s |")
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        stats = []
        for (s, lvl), d in df.groupby(["system_key", "access_level"]):
            m, lo, hi = bootstrap_mean_ci(d.regret.tolist())
            stats.append((s, lvl, m, lo, hi, d))
            md.append(f"| {s} | {lvl} | {len(d)} | {m:.3f} [{lo:.3f}, {hi:.3f}] | {d.top1.mean():.2f} | "
                      f"{d.normalised.mean():.2f} | {d.fallback.mean():.2f} | {d.calls.mean():.1f} | "
                      f"{d.cost_usd.mean():.4f} | {d.latency_s.mean():.1f} |")
        # Paired tests: MA vs SA, MA vs SA-vote, MA vs B2 (same positions, same access level).
        tests = []
        for lvl in sorted(df.access_level.unique()):
            sub = df[(df.access_level == lvl) & (df.repeat == 0)]
            piv = sub.pivot_table(index="position_id", columns="system_key", values="regret")
            for other in ("sa", "sa-vote", "b2"):
                if "ma" in piv and other in piv:
                    pr = piv[["ma", other]].dropna()
                    p, r = wilcoxon_paired(pr["ma"], pr[other])
                    tests.append((lvl, other, len(pr), float((pr["ma"] - pr[other]).mean()), p, r))
        if tests:
            adj = holm([t[4] for t in tests])
            md.append("\nPaired comparisons of regret (MA minus other; negative = MA better), Wilcoxon signed-rank, Holm-adjusted:\n")
            md.append("| info | MA vs | n | mean diff | p | p (Holm) | rank-biserial |")
            md.append("|---|---|---|---|---|---|---|")
            for (lvl, other, n, diff, p, r), pa in zip(tests, adj):
                md.append(f"| {lvl} | {other} | {n} | {diff:+.3f} | {p:.3g} | {pa:.3g} | {r:+.2f} |")

        # Figure: mean regret per system (one panel per access level).
        levels = sorted(df.access_level.unique())
        fig, axes = plt.subplots(1, len(levels), figsize=(5.2 * len(levels), 0.5 * df.system_key.nunique() + 1.4),
                                 sharex=True, squeeze=False)
        for ax, lvl in zip(axes[0], levels):
            rows = [t for t in stats if t[1] == lvl]
            rows.sort(key=lambda t: SYSTEM_ORDER.index(t[0]) if t[0] in SYSTEM_ORDER else 99)
            for i, (s, _, m, lo, hi, _) in enumerate(rows):
                c = SYSTEM_COLOR.get(s, "#2a78d6")
                ax.barh(i, m, color=c, height=0.6)
                ax.plot([lo, hi], [i, i], color=INK, linewidth=1.5)
                ax.text(hi + 0.01, i, f"{m:.2f}", va="center", fontsize=9, color=INK)
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels([LABEL.get(t[0], t[0]) for t in rows])
            ax.invert_yaxis()
            ax.set_xlabel("mean regret (oracle EIV units; lower is better)")
            ax.set_title(f"{exp}, information {lvl}", loc="left", fontsize=11, color=INK)
            style(ax)
        md.append(f"\nFigure: `{save(fig, f'{exp}_regret')}`")

        # Figure: regret by situation tag (small multiples, one panel per tag).
        tags = ("offense", "position", "defense")
        lvl = "L2" if "L2" in levels else levels[0]
        sub = df[df.access_level == lvl]
        systems = [s for s in SYSTEM_ORDER if s in set(sub.system_key)]
        fig, axes = plt.subplots(1, 3, figsize=(12, 0.45 * len(systems) + 1.4), sharex=True)
        for ax, tag in zip(axes, tags):
            t = sub[sub.tags.apply(lambda x, tag=tag: tag in x)]
            for i, s in enumerate(systems):
                v = t[t.system_key == s].regret
                if len(v):
                    ax.barh(i, v.mean(), color=SYSTEM_COLOR.get(s), height=0.6)
                    ax.text(v.mean() + 0.01, i, f"{v.mean():.2f} (n={len(v)})", va="center", fontsize=8, color=INK)
            ax.set_yticks(range(len(systems)))
            ax.set_yticklabels([LABEL.get(s, s) for s in systems] if tag == "offense" else [])
            ax.invert_yaxis()
            ax.set_title(f"{tag} situations", loc="left", fontsize=11, color=INK)
            style(ax)
        axes[1].set_xlabel(f"mean regret at {lvl} (lower is better)")
        md.append(f"\nFigure: `{save(fig, f'{exp}_regret_by_situation')}`")

        # Figure: cost vs quality (LLM systems only).
        llm = sub[sub.calls > 0]
        if not llm.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            for s, d in llm.groupby("system_key"):
                x, y = max(d.cost_usd.mean(), 1e-6), d.normalised.mean()
                ax.scatter(x, y, s=70, color=SYSTEM_COLOR.get(s), edgecolor="#fcfcfb", linewidth=2, zorder=3)
                ax.annotate(LABEL.get(s, s), (x, y), textcoords="offset points", xytext=(8, 4), fontsize=9, color=INK)
            ax.set_xscale("log")
            ax.set_xlabel("mean cost per decision (USD, log scale)")
            ax.set_ylabel("normalised decision score (0 = random, 1 = best)")
            ax.grid(color=GRID)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.set_title(f"{exp}: quality vs cost at {lvl}", loc="left", fontsize=11, color=INK)
            md.append(f"\nFigure: `{save(fig, f'{exp}_cost_quality')}`")


def main() -> None:
    md = ["# Results summary", "",
          "Generated by `python experiments/analyze.py` from the files named in each section. "
          "Do not edit by hand."]
    tournaments(md)
    positions(md)
    (ROOT / "results" / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
