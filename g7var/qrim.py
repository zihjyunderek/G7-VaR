"""Section 4.2: QR-IM, linear quantile regression of r_{t+1} on the implied-volatility features at t
(Koenker and Bassett, 1978; de Lange, 2022). VaR_q = -(alpha_q + beta_q' x_t).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.regression.quantile_regression import QuantReg

from g7var import CFG
from g7var.backtest import forecasts
from g7var.data import master, split


def design(df: pd.DataFrame, cols: list) -> np.ndarray:
    return np.column_stack([np.ones(len(df))] + [df[c].values for c in cols])


def fit(y: np.ndarray, x: np.ndarray, q: float, p_tol: bool = True):
    kw = {"p_tol": 1e-6} if p_tol else {}
    return QuantReg(y, x).fit(q=q, vcov=CFG["qrim"]["vcov"], max_iter=CFG["qrim"]["max_iter"], **kw)


def qr_im(pair: str, run: dict):
    """Forecasts, backtests and coefficients (with p-values) of QR-IM for one pair; None if the test sample is too short."""
    cols = run["features"] + run["extra"].get(pair, [])
    df = master(pair, tenor=run["tenor"], features=run["features"], test_end=run["test_end"])
    train, test = split(df, run)
    if len(test) < 30:
        return None
    x_train, x_test = design(train, cols), design(test, cols)
    var, coefs = {}, []
    for q in CFG["quantiles"]:
        res = fit(train["return_fwd"].values, x_train, q)
        var[q] = -(x_test @ res.params)
        row = {"Pair": pair, "Quantile": f"{q * 100:.1f}%"}
        for name, b, p in zip(["Const"] + cols, res.params, res.pvalues):
            row[name], row[f"{name}_pval"] = round(float(b), 6), round(float(p), 4)
        coefs.append(row)
    pred, summ = forecasts(pair, "QR-IM", test.index, test["return_fwd"].values, var)
    return pred, summ, pd.DataFrame(coefs)


def fitted_quantiles(train: pd.DataFrame, cols: list) -> dict:
    """QR-IM fitted quantiles on the training sample: the targets the tree ensembles learn (section 4.3)."""
    y, x = train["return_fwd"].values, design(train, cols)
    return {q: (x @ fit(y, x, q, p_tol=False).params).astype(np.float64) for q in CFG["quantiles"]}
