"""Section 4.3: quantile random forest, XGBoost, LightGBM and CatBoost. Each learns the QR-IM fitted
quantile of the training sample from the implied-volatility features and their ATM x RR25 interaction.
"""
from __future__ import annotations

import catboost
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from quantile_forest import RandomForestQuantileRegressor

from g7var import CFG
from g7var.backtest import forecasts
from g7var.data import master, split
from g7var.qrim import fitted_quantiles


def features(df, cols):
    x = np.column_stack([df[c].values.astype(np.float64) for c in cols])
    if "ATM" in cols and "RR25" in cols:
        x = np.column_stack([x, df["ATM"].values * df["RR25"].values])
    return x


def _predict(model, x_train, y_train, x_test, q, cols):
    if model == "RandomForest":
        c = CFG["random_forest"]
        rf = RandomForestQuantileRegressor(n_estimators=c["n_estimators"], random_state=c["random_state"], n_jobs=-1)
        return np.array(rf.fit(x_train, y_train).predict(x_test, quantiles=q))
    if model == "XGBoost":
        m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=float(q), tree_method="hist", device="cpu",
                             verbosity=0, n_jobs=-1, **CFG["xgboost"])
        return m.fit(x_train, y_train).predict(x_test)
    if model == "LightGBM":
        params = {k: v for k, v in CFG["lightgbm"].items() if k != "num_boost_round"} | {"objective": "quantile", "alpha": float(q)}
        names = list(cols) + (["ATM_x_RR25"] if "ATM" in cols and "RR25" in cols else [])
        data = lgb.Dataset(x_train, label=y_train, feature_name=names, free_raw_data=False)
        booster = lgb.train(params, data, num_boost_round=CFG["lightgbm"]["num_boost_round"], callbacks=[lgb.log_evaluation(period=-1)])
        return booster.predict(x_test)
    if model == "CatBoost":
        m = catboost.CatBoostRegressor(loss_function=f"Quantile:alpha={q}", task_type="CPU", verbose=0, **CFG["catboost"])
        return m.fit(x_train, y_train).predict(x_test)
    raise ValueError(model)


def tree_ensemble(model: str, pair: str, run: dict):
    cols = run["features"]
    df = master(pair, tenor=run["tenor"], features=cols, test_end=run["test_end"])
    train, test = split(df, run)
    if len(test) < 30:
        return None
    x_train, x_test, targets = features(train, cols), features(test, cols), fitted_quantiles(train, cols)
    var = {q: -_predict(model, x_train, targets[q], x_test, q, cols).astype(np.float64) for q in CFG["quantiles"]}
    return forecasts(pair, model, test.index, test["return_fwd"].values.astype(np.float64), var)
