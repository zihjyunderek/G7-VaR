"""Section 4.1: the benchmarks. HS, RiskMetrics, GJR-GARCH and FHS-GARCH use returns only;
FHS-IV scales filtered historical simulation by the ATM implied volatility.
"""
from __future__ import annotations

import numpy as np
from arch import arch_model
from scipy.stats import norm, t as student_t

from g7var import CFG
from g7var.backtest import forecasts
from g7var.data import master, split


def _out_of_sample(df, run, first):
    """Positions t >= first whose date is on or after the start of the test sample."""
    return [t for t in range(first, len(df)) if str(df.index[t].date()) >= run["test_start"]]


def hs(pair: str, run: dict):
    """Historical simulation: the empirical quantile of the last 250 returns."""
    df = master(pair, test_end=run["test_end"])
    r, w = df["return"].values, CFG["hs"]["window"]
    lag = 0 if CFG["hs"]["window_end"] == "inclusive" else 1
    ts = _out_of_sample(df, run, w)
    var = {q: [float(-np.quantile(r[t - w + 1 - lag: t + 1 - lag], q)) for t in ts] for q in CFG["quantiles"]}
    return forecasts(pair, "HS", df.index[ts], df["return_fwd"].values[ts], var)


def riskmetrics(pair: str, run: dict):
    """RiskMetrics: a normal quantile times a 250-day rolling standard deviation (or the EWMA of RiskMetrics, 1996)."""
    df = master(pair, test_end=run["test_end"])
    r, rm, n = df["return"].values, CFG["riskmetrics"], len(df)
    sigma = np.full(n, np.nan)
    if rm["method"] == "ewma":
        first, s2 = 1, float(np.var(r[: rm["seed_window"]]))
        for t in range(1, n):
            sigma[t] = np.sqrt(s2)
            s2 = rm["lambda"] * s2 + (1.0 - rm["lambda"]) * r[t - 1] ** 2
    else:
        first = rm["window"]
        for t in range(first, n):
            sigma[t] = np.std(r[t - first + 1: t + 1], ddof=1)
    ts = _out_of_sample(df, run, first)
    var = {q: [float(-norm.ppf(q) * float(sigma[t])) for t in ts] for q in CFG["quantiles"]}
    return forecasts(pair, "RiskMetrics", df.index[ts], df["return_fwd"].values[ts], var)


def _gjr(pair: str, run: dict):
    """GJR-GARCH(1,1) with Student-t errors, fitted on the training sample and run over the whole sample."""
    df = master(pair, test_end=run["test_end"])
    train, _ = split(df, run)
    res = arch_model(train["return"].values * 100.0, vol="Garch", p=1, o=1, q=1, dist="t").fit(disp="off", show_warning=False)
    full = arch_model(df["return"].values * 100.0, vol="Garch", p=1, o=1, q=1, dist="t").fix(res.params)
    return df, res, full.conditional_volatility / 100.0


def gjr_garch(pair: str, run: dict):
    df, res, sigma = _gjr(pair, run)
    nu = float(res.params.get("nu", 8.0))
    oos = df.index >= run["test_start"]
    var = {q: [float(-student_t.ppf(q, df=nu) * s) for s in sigma[oos]] for q in CFG["quantiles"]}
    return forecasts(pair, "GJR-GARCH", df.index[oos], df.loc[oos, "return_fwd"].values, var)


def _filtered(df, run, z, sigma, model, pair):
    """Filtered historical simulation: the quantile of the last 250 standardised returns (up to t-1) times sigma_t."""
    w = CFG["hs"]["window"]
    ts = _out_of_sample(df, run, w)
    var = {q: [float(-float(np.quantile(z[t - w: t], q)) * float(sigma[t])) for t in ts] for q in CFG["quantiles"]}
    return forecasts(pair, model, df.index[ts], df["return_fwd"].values[ts], var)


def fhs_garch(pair: str, run: dict):
    df, _, sigma = _gjr(pair, run)
    z = df["return"].values / np.where(sigma > 1e-8, sigma, 1e-8)
    return _filtered(df, run, z, sigma, "FHS-GARCH", pair)


def fhs_iv(pair: str, run: dict):
    """FHS-IV: returns standardised by the daily ATM implied volatility, ATM / 100 / sqrt(252)."""
    df = master(pair, tenor=run["tenor"], features=["ATM", "RR25", "BF25"], test_end=run["test_end"])
    sigma = df["ATM"].values / 100.0 / np.sqrt(252.0)
    z = df["return"].values / np.where(sigma > 1e-8, sigma, 1e-8)
    return _filtered(df, run, z, sigma, "FHS-IV", pair)
