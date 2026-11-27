"""HS and RiskMetrics under both run profiles, on simulated returns (the licensed data stay local).

config.yaml is the run of 2026-06-04 behind every table but one: HS on 250 days ending at r_t,
RiskMetrics a 250-day rolling standard deviation. config_2026-05-04.yaml is the run behind
Table 5-7: HS on 500 days ending at r_{t-1}, RiskMetrics an EWMA with lambda 0.94 seeded on 30 returns.
Each is checked against the loop written out by hand.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import g7var.baselines as baselines
from g7var import CFG

Q = [0.01, 0.025, 0.05]
RUN = {"test_start": "2020-01-01", "test_end": "2025-12-31", "train_end": "2019-12-31", "tenor": "1W"}


@pytest.fixture
def returns(monkeypatch):
    rng = np.random.default_rng(20260504)
    n = 1800
    r = rng.standard_t(5, n) * 0.004 * np.exp(np.cumsum(rng.normal(0, 0.05, n)) * 0.2)
    df = pd.DataFrame({"return": r, "return_fwd": np.r_[r[1:], np.nan]}, index=pd.bdate_range("2017-01-02", periods=n)).iloc[:-1]
    monkeypatch.setattr(baselines, "master", lambda pair, **kw: df)
    monkeypatch.setitem(CFG, "quantiles", Q)
    return df


def profile(monkeypatch, may):
    monkeypatch.setitem(CFG, "hs", {"window": 500, "window_end": "exclusive"} if may else {"window": 250, "window_end": "inclusive"})
    monkeypatch.setitem(CFG, "riskmetrics", {"method": "ewma" if may else "rolling", "window": 250, "lambda": 0.94, "seed_window": 30})


def by_hand(df, may, model):
    r, out = df["return"].values, {q: [] for q in Q}
    if model == "RiskMetrics" and may:
        s2 = float(np.var(r[:30]))
        for t in range(1, len(df)):
            if str(df.index[t].date()) >= RUN["test_start"]:
                for q in Q:
                    out[q].append(-norm.ppf(q) * np.sqrt(s2))
            s2 = 0.94 * s2 + (1.0 - 0.94) * r[t - 1] ** 2
        return out
    w = 500 if model == "HS" and may else 250
    for t in range(w, len(df)):
        if str(df.index[t].date()) < RUN["test_start"]:
            continue
        window = r[t - w: t] if may else r[t - w + 1: t + 1]
        for q in Q:
            out[q].append(-np.quantile(window, q) if model == "HS" else -norm.ppf(q) * np.std(window, ddof=1))
    return out


@pytest.mark.parametrize("may", [True, False], ids=["2026-05-04", "2026-06-04"])
@pytest.mark.parametrize("model", ["HS", "RiskMetrics"])
def test_baseline_matches_its_definition(returns, monkeypatch, model, may):
    profile(monkeypatch, may)
    pred, summary = (baselines.hs if model == "HS" else baselines.riskmetrics)("EUR/USD", RUN)
    ref = by_hand(returns, may, model)
    for q in Q:
        got = pred.loc[pred.Quantile == f"{q * 100:.1f}%", "VaR"].to_numpy()
        np.testing.assert_allclose(got, np.round(ref[q], 8), rtol=0, atol=1e-12)
    assert len(summary) == len(Q)
