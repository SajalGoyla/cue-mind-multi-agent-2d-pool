import pytest

from cuemind.eval.stats import bootstrap_mean_ci, sign_test, wilson_ci


def test_wilson_ci_known_values():
    lo, hi = wilson_ci(7, 10)
    assert lo == pytest.approx(0.3968, abs=1e-3) and hi == pytest.approx(0.8922, abs=1e-3)
    assert wilson_ci(0, 0) == (0.0, 1.0)
    lo, hi = wilson_ci(10, 10)
    assert hi == 1.0 and lo > 0.69


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    vals = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    m, lo, hi = bootstrap_mean_ci(vals, n_boot=2000, seed=1)
    assert m == pytest.approx(0.7) and lo <= m <= hi
    assert bootstrap_mean_ci(vals, n_boot=2000, seed=1) == (m, lo, hi)


def test_sign_test():
    assert sign_test([1] * 10) == pytest.approx(2 / 1024)
    assert sign_test([0, 0]) == 1.0
