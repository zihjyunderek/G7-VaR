"""Backtests against their definitions in the literature.

Tests marked xfail record where the code that produced the thesis
departs from the literature; README.md explains each. They are
strict: once the code is changed to follow the definition, they fail as
"unexpectedly passing" and the mark must be removed.
"""

import numpy as np
import pytest
from scipy import stats

from g7var.backtest import basel_zone, christoffersen as cc_test, dynamic_quantile as dq_test, kupiec as kupiec_test

RNG = np.random.default_rng(2026)


def hits_with(n_obs: int, n_breach: int) -> np.ndarray:
    """A hit sequence with n_breach evenly spread violations."""
    h = np.zeros(n_obs, dtype=int)
    h[np.linspace(0, n_obs - 1, n_breach, dtype=int)] = 1
    return h


# ── Kupiec (1995) proportion-of-failures test ──────────────────────────────
def test_kupiec_matches_the_likelihood_ratio():
    t, x, a = 250, 7, 0.01
    lr, p = kupiec_test(hits_with(t, x), a)
    pi = x / t
    expected = -2 * (np.log((1 - a) ** (t - x) * a ** x) - np.log((1 - pi) ** (t - x) * pi ** x))
    assert lr == pytest.approx(expected, rel=1e-12)
    assert p == pytest.approx(stats.chi2.sf(expected, 1), rel=1e-9)


def test_kupiec_is_zero_at_the_nominal_rate():
    lr, p = kupiec_test(hits_with(1000, 10), 0.01)
    assert lr == pytest.approx(0.0, abs=1e-12) and p == pytest.approx(1.0)


@pytest.mark.xfail(strict=True, reason="the thesis code returns NaN for zero breaches; the LR is -2 T ln(1 - alpha)")
def test_kupiec_rejects_a_model_with_no_breaches():
    lr, p = kupiec_test(np.zeros(714, dtype=int), 0.01)
    assert lr == pytest.approx(-2 * 714 * np.log(0.99)) and p < 0.001


# ── Christoffersen (1998) conditional coverage ─────────────────────────────
def test_conditional_coverage_is_uc_plus_independence():
    h = (RNG.random(1500) < 0.02).astype(int)
    lr_uc, lr_ind, lr_cc, p_cc = cc_test(h, 0.01)
    assert lr_cc == pytest.approx(lr_uc + lr_ind)
    assert p_cc == pytest.approx(stats.chi2.sf(lr_cc, 2))


def test_independence_detects_clustered_breaches():
    h = np.zeros(1000, dtype=int)
    for start in range(50, 1000, 100):
        h[start:start + 3] = 1  # breaches come in runs of three
    _, lr_ind, _, _ = cc_test(h, 0.03)
    assert stats.chi2.sf(lr_ind, 1) < 0.001


# ── Basel traffic light (BCBS, 1996) ───────────────────────────────────────
@pytest.mark.parametrize("n_breach, zone", [(0, "Green"), (4, "Green"), (5, "Yellow"), (9, "Yellow"), (10, "Red")])
def test_basel_zones_for_99_percent_var(n_breach, zone):
    assert basel_zone(hits_with(250, n_breach), window=250) == zone


@pytest.mark.xfail(strict=True, reason="the thesis code applies the 99% thresholds (4 and 9 breaches) to 97.5% and 95% VaR too")
def test_well_calibrated_95_percent_var_is_not_red():
    # 13 breaches in 250 days is what a correct 95% VaR produces on average
    assert basel_zone(hits_with(250, 13), window=250) != "Red"


# ── Engle and Manganelli (2004) dynamic quantile test ──────────────────────
@pytest.mark.xfail(strict=True, reason="the thesis code leaves the intercept out of the Wald test, so DQ misses wrong coverage")
def test_dq_rejects_wrong_unconditional_coverage():
    # iid breaches at 5% against a 1% VaR: unconditional coverage is wrong, nothing else
    h = (RNG.random(1000) < 0.05).astype(int)
    var_pred = 0.01 + 0.001 * RNG.random(1000)
    _, p = dq_test(h, var_pred, 0.01, n_lags=4)
    assert p < 0.01
