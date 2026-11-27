"""Section 4.5: breach ratio, Kupiec, Christoffersen, dynamic quantile and Basel traffic light.

These are the tests as coded for the thesis. Three departures from the literature are kept so the
published numbers stay reproducible (tests/test_backtest.py marks them; see README):
Kupiec returns NaN at zero breaches; the Basel 99% thresholds are applied to every quantile;
the DQ Wald test leaves the intercept out and uses chi2(lags + 1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from g7var import CFG


def kupiec(hits: np.ndarray, alpha: float) -> tuple[float, float]:
    """Unconditional coverage (Kupiec, 1995): LR and p-value."""
    t, x = len(hits), int(hits.sum())
    if x in (0, t):
        return np.nan, np.nan
    rate = x / t
    lr = 2.0 * (x * np.log(rate / alpha) + (t - x) * np.log((1.0 - rate) / (1.0 - alpha)))
    return float(lr), float(1.0 - stats.chi2.cdf(lr, df=1))


def christoffersen(hits: np.ndarray, alpha: float) -> tuple[float, float, float, float]:
    """Conditional coverage (Christoffersen, 1998): LR_uc, LR_ind, LR_cc = LR_uc + LR_ind, p-value of LR_cc."""
    lr_uc, _ = kupiec(hits, alpha)
    if np.isnan(lr_uc):
        return (np.nan,) * 4
    prev, cur = hits[:-1], hits[1:]
    n00, n01 = int(np.sum((prev == 0) & (cur == 0))), int(np.sum((prev == 0) & (cur == 1)))
    n10, n11 = int(np.sum((prev == 1) & (cur == 0))), int(np.sum((prev == 1) & (cur == 1)))
    if n00 + n01 == 0 or n10 + n11 == 0:
        return lr_uc, np.nan, np.nan, np.nan
    pi01, pi11, pi = n01 / (n00 + n01), n11 / (n10 + n11), (n01 + n11) / (n00 + n01 + n10 + n11)
    eps = 1e-10
    if pi <= eps or pi >= 1.0 - eps:
        return lr_uc, np.nan, np.nan, np.nan
    log = lambda p: np.log(max(p, eps))
    lr_ind = -2.0 * (n00 * log(1 - pi) + n01 * log(pi) + n10 * log(1 - pi) + n11 * log(pi)
                     - n00 * log(1 - pi01) - n01 * log(pi01) - n10 * log(1 - pi11) - n11 * log(pi11))
    lr_cc = lr_uc + lr_ind
    return float(lr_uc), float(lr_ind), float(lr_cc), float(1.0 - stats.chi2.cdf(lr_cc, df=2))


def dynamic_quantile(hits: np.ndarray, var: np.ndarray, alpha: float, lags: int | None = None) -> tuple[float, float]:
    """Dynamic quantile test (Engle and Manganelli, 2004) on lagged hits and the VaR, as coded for the thesis."""
    lags = CFG["backtest"]["dq_lags"] if lags is None else lags
    h = hits.astype(float) - alpha
    t, k = len(h), lags + 1
    if t < k + 20:
        return np.nan, np.nan
    x = np.ones((t - lags, k + 1))
    for j in range(1, lags + 1):
        x[:, j] = h[lags - j: t - j]
    x[:, -1] = var[lags:]
    y = h[lags:]
    try:
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        resid = y - x @ beta
        sigma2 = float(resid @ resid / (t - lags))
        if sigma2 < 1e-12:
            return np.nan, np.nan
        cov = sigma2 * np.linalg.pinv(x.T @ x)
        r = np.eye(len(beta))[1:]
        wald = float(r @ beta @ np.linalg.pinv(r @ cov @ r.T) @ (r @ beta))
        if np.isnan(wald) or wald < 0:
            return np.nan, np.nan
        return wald, float(1.0 - stats.chi2.cdf(wald, df=k))
    except Exception:
        return np.nan, np.nan


def basel_zone(hits: np.ndarray, window: int | None = None) -> str:
    """Basel traffic light on the last 250 days: 0-4 breaches green, 5-9 yellow, 10 or more red."""
    w = window or CFG["backtest"]["basel_window"]
    n = int(hits[-min(w, len(hits)):].sum())
    return "Green" if n <= 4 else "Yellow" if n <= 9 else "Red"


def _r6(x):
    return np.nan if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 6)


def summary(returns: np.ndarray, var: dict, pair: str, model: str) -> pd.DataFrame:
    """One row per quantile: breaches, sums and every test. `var` maps quantile to the VaR (a positive loss)."""
    rows = []
    for q, v in var.items():
        v = np.asarray(v)
        hits = (returns < -v).astype(int)
        _, _, lr_cc, p_cc = christoffersen(hits, q)
        lr_uc, p_uc = kupiec(hits, q)
        dq, p_dq = dynamic_quantile(hits, v, q)
        rows.append({"Pair": pair, "Model": model, "Quantile": f"{q * 100:.1f}%", "N_OOS": len(hits),
                     "N_Breach": int(hits.sum()), "Breach_Ratio": round(float(hits.mean() * 100), 4),
                     "Sum_if_Breach": round(float(np.sum(hits * (returns + v))), 8),
                     "Sum_if_NoBreach": round(float(np.sum((hits == 0) * (v + returns))), 8),
                     "Kupiec_LR": _r6(lr_uc), "Kupiec_pval": _r6(p_uc), "CC_LR": _r6(lr_cc), "CC_pval": _r6(p_cc),
                     "DQ_stat": _r6(dq), "DQ_pval": _r6(p_dq), "Basel_Zone": basel_zone(hits)})
    return pd.DataFrame(rows)


def forecasts(pair: str, model: str, dates, returns, var: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The forecasts (one row per day and quantile) and their backtest summary."""
    returns = np.asarray(returns, dtype=float)
    pred = pd.concat([pd.DataFrame({"Date": dates, "Pair": pair, "Model": model, "Quantile": f"{q * 100:.1f}%",
                                    "VaR": np.round(np.asarray(v, dtype=float), 8), "Return": np.round(returns, 8),
                                    "Breach": (returns < -np.asarray(v, dtype=float)).astype(int)})
                      for q, v in var.items()], ignore_index=True)
    return pred, summary(returns, var, pair, model)
