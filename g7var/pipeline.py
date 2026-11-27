"""Run one configuration: every model on every pair, then save forecasts, backtests and QR-IM coefficients."""
from __future__ import annotations

import time
from functools import partial
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from g7var import CFG, ROOT, run_settings
from g7var import baselines, deep, ensemble, qrim

FUNCTIONS = {"HS": baselines.hs, "RiskMetrics": baselines.riskmetrics, "GJR-GARCH": baselines.gjr_garch,
             "FHS-GARCH": baselines.fhs_garch, "FHS-IV": baselines.fhs_iv, "QR-IM": qrim.qr_im,
             **{m: partial(ensemble.tree_ensemble, m) for m in ("RandomForest", "XGBoost", "LightGBM", "CatBoost")},
             **{m: partial(deep.network, m) for m in deep.NETS}}
SEQUENTIAL = {"GJR-GARCH", "FHS-GARCH", "CatBoost", "FFNN", "RNN", "LSTM"}   # multi-threaded inside, or on the GPU


def results_dir() -> Path:
    tag = CFG.get("results_tag")
    return ROOT / "results" / (f"runs_{tag}" if tag else "runs")


def run(name: str, models: list | None = None) -> pd.DataFrame:
    """Runs `models` (default: the run's models) and writes results/runs/<name>/; returns the backtests."""
    settings = run_settings(name)
    out = results_dir() / name
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    summaries, coefs = [], []
    for model in models or settings["models"]:
        t0 = time.time()
        jobs = 1 if model in SEQUENTIAL else -1
        res = [r for r in Parallel(n_jobs=jobs)(delayed(FUNCTIONS[model])(p, settings) for p in CFG["pairs"]) if r is not None]
        if not res:
            continue
        pd.concat([r[0] for r in res]).to_csv(out / "predictions" / f"{model}.csv.gz", index=False, date_format="%Y-%m-%d")
        summaries.append(pd.concat([r[1] for r in res]))
        if model == "QR-IM":
            coefs.append(pd.concat([r[2] for r in res]))
        print(f"  {name:<17} {model:<12} {time.time() - t0:6.1f} s")
    backtests = pd.concat(summaries, ignore_index=True)
    backtests.insert(0, "run", name)
    _replace(out / "backtests.csv", backtests, ["run", "Model"])
    if coefs:
        c = pd.concat(coefs, ignore_index=True)
        c.insert(0, "run", name)
        c.to_csv(out / "qrim_coefficients.csv", index=False)
    return backtests


def _replace(path: Path, new: pd.DataFrame, keys: list) -> None:
    """Writes `new`, keeping the rows of `path` whose keys `new` does not have (a partial re-run keeps the rest)."""
    if path.exists():
        old = pd.read_csv(path)
        keep = ~old.set_index(keys).index.isin(new.set_index(keys).index)
        new = pd.concat([old[keep], new], ignore_index=True)
    new.to_csv(path, index=False)
