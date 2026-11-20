"""Statistics used in the evaluation (kept small, explicit and unit-tested)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy import stats as sps


def wilson_ci(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (draws may be counted as 0.5)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_mean_ci(values: Sequence[float], n_boot: int = 10_000, alpha: float = 0.05,
                      seed: int = 0) -> tuple[float, float, float]:
    """(mean, low, high) percentile bootstrap CI of the mean."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(x.mean()), float(lo), float(hi))


def sign_test(diffs: Sequence[float]) -> float:
    """Two-sided sign test p-value on paired differences (zeros dropped)."""
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    if pos + neg == 0:
        return 1.0
    return float(sps.binomtest(pos, pos + neg, 0.5).pvalue)


def wilcoxon_paired(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """(p-value, rank-biserial effect size) for paired samples a vs b (zeros dropped)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]
    if d.size == 0:
        return 1.0, 0.0
    res = sps.wilcoxon(d)
    ranks = sps.rankdata(np.abs(d))
    r_plus, r_minus = ranks[d > 0].sum(), ranks[d < 0].sum()
    return float(res.pvalue), float((r_plus - r_minus) / ranks.sum())


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values (same order as the input)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj
