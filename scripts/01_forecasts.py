"""Chapters 5 and 6: forecast and backtest every configuration, then gather the results.

    python scripts/01_forecasts.py                          # all 19 runs (about an hour on a laptop GPU)
    python scripts/01_forecasts.py --runs main_1W           # chapter 5 only
    python scripts/01_forecasts.py --runs main_1W --models HS QR-IM

Writes results/runs/<run>/ (predictions/ stays local: it is derived day by day from the licensed data)
and, from every run folder, results/backtests.csv and results/qrim_coefficients.csv.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g7var import CFG, MODELS, ROOT  # noqa: E402
from g7var.pipeline import results_dir, run  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--runs", nargs="*", default=list(CFG["runs"]), choices=list(CFG["runs"]))
ap.add_argument("--models", nargs="*", choices=MODELS)
args = ap.parse_args()

for name in args.runs:
    run(name, args.models)

# results/backtests.csv and results/qrim_coefficients.csv: the runs just made replace their rows
tag = f"_{CFG['results_tag']}" if CFG.get("results_tag") else ""
order = {"run": list(CFG["runs"]), "Model": MODELS, "Pair": list(CFG["pairs"])}
for name in ("backtests", "qrim_coefficients"):
    path = ROOT / "results" / f"{name}{tag}.csv"
    new = [pd.read_csv(results_dir() / r / f"{name}.csv") for r in args.runs if (results_dir() / r / f"{name}.csv").exists()]
    if not new:
        continue
    new = pd.concat(new)
    old = pd.read_csv(path) if path.exists() else new.iloc[:0]
    keys = [k for k in ("run", "Model", "Pair") if k in new]
    replaced = old.set_index(keys[:2]).index.isin(new.set_index(keys[:2]).index)   # by run and model (run and pair for coefficients)
    both = pd.concat([old[~replaced], new], ignore_index=True)
    both = both.sort_values(keys + ["Quantile"], key=lambda s: s.map(order[s.name].index) if s.name in order else s, kind="stable")
    both.to_csv(path, index=False)
    print(f"results/{name}{tag}.csv: {both.run.nunique()} runs, {len(both)} rows")
