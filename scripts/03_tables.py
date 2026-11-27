"""Tables 5-1 to 6-10 of the thesis, from results/backtests.csv and results/qrim_coefficients.csv.

One CSV per table in results/tables/, in the thesis layout: a row per model (and quantile), a column
per currency pair. Coefficients are multiplied by 100, as printed; their p-values follow in columns
"<pair> p". Table 5-7 reads the run of 2026-05-04 (results/backtests_2026-05-04.csv), as the thesis did.
Tables 7-1 to 7-3 come from scripts/02_event_studies.py.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g7var import CFG, MODELS, ROOT  # noqa: E402

RES = ROOT / "results"
PAIRS = list(CFG["pairs"])
Q = ["1.0%", "2.5%", "5.0%"]
NINE = MODELS[4:]                     # the models of the robustness runs: FHS-IV to LSTM
BT = pd.read_csv(RES / "backtests.csv")
COEF = pd.read_csv(RES / "qrim_coefficients.csv")


def by_pair(run, metric, models, quantiles=("1.0%",), source=BT, green=False):
    """Rows: model (and quantile); columns: the pairs. `green` adds the count of green Basel zones at 1%."""
    d = source[(source.run == run) & source.Model.isin(models) & source.Quantile.isin(quantiles)]
    t = d.pivot_table(index=["Model", "Quantile"], columns="Pair", values=metric, aggfunc="first")
    if metric == "Basel_Zone":
        t = d.pivot(index=["Model", "Quantile"], columns="Pair", values=metric)
    t = t.reindex(pd.MultiIndex.from_product([models, quantiles], names=["Model", "Quantile"]))[PAIRS]
    if green:
        z = source[(source.run == run) & (source.Quantile == "1.0%")].groupby("Model").Basel_Zone.apply(lambda s: (s == "Green").sum())
        t["Green zones (of 7)"] = [z[m] for m, _ in t.index]
    return t.reset_index().drop(columns=[] if len(quantiles) > 1 else ["Quantile"])


def coefficients(run, variables, pairs=PAIRS):
    """Rows: variable and quantile; columns: the coefficient x 100 per pair, then its p-value per pair."""
    rows = []
    for v in variables:
        for q in Q:
            d = COEF[(COEF.run == run) & (COEF.Quantile == q)].set_index("Pair")
            rows.append({"Variable": v, "Quantile": q, **{p: round(d.loc[p, v] * 100, 6) for p in pairs},
                         **{f"{p} p": d.loc[p, f"{v}_pval"] for p in pairs}})
    return pd.DataFrame(rows)


def tests(run, models, columns):
    return pd.concat([by_pair(run, c, models).assign(Test=c.replace("_pval", "")) for c in columns]).set_index(["Test", "Model"]).reset_index()


TABLES = {
    "5-1": coefficients("main_1W", ["Const", "ATM", "RR25"]),
    "5-2": by_pair("main_1W", "Breach_Ratio", MODELS),
    "5-3": by_pair("main_1W", "Breach_Ratio", MODELS, Q),
    "5-4": by_pair("main_1W", "Breach_Ratio", ["LightGBM", "CatBoost", "QR-IM"], Q),
    "5-5": BT[(BT.run == "main_1W") & (BT.Model == "LSTM") & (BT.Pair == "USD/CAD")][["Quantile", "N_Breach", "Breach_Ratio"]],
    "5-6": by_pair("main_1W", "Sum_if_Breach", MODELS),
    "5-7": by_pair("main_1W", "Sum_if_NoBreach", MODELS, source=pd.read_csv(RES / "backtests_2026-05-04.csv")),
    "5-8": tests("main_1W", ["FHS-GARCH", "FHS-IV", "QR-IM", "RandomForest", "LightGBM", "CatBoost"], ["CC_pval", "DQ_pval", "Kupiec_pval"]),
    "5-9": by_pair("main_1W", "Kupiec_pval", ["FHS-GARCH", "FHS-IV", "QR-IM", "RandomForest", "LightGBM", "CatBoost"], Q),
    "5-10": by_pair("main_1W", "Basel_Zone", MODELS),
    "6-1": coefficients("r1_RR10", ["ATM", "RR10"]),
    "6-2": COEF[COEF.run == "r1_BF25"].pivot(index="Quantile", columns="Pair", values="BF25_pval")[PAIRS].reset_index(),
    "6-3": by_pair("r1_All", "Breach_Ratio", NINE, green=True),
    "6-4": coefficients("r4_OVX_USDCAD", ["ATM", "RR25", "OVX"], ["USD/CAD"]),
    "6-5": coefficients("r5_ImpliedSpread", ["Implied_Bid_Ask_Spread"]),
    "6-6": pd.concat([by_pair(r, "Breach_Ratio", ["FHS-IV", "QR-IM", "LightGBM", "CatBoost", "RandomForest"]).assign(Tenor=t)
                      for t, r in [("1W", "main_1W"), ("6M", "r2_6M"), ("1Y", "r2_1Y"), ("3Y", "r2_3Y")]]).set_index(["Tenor", "Model"]).reset_index(),
    "6-7": by_pair("r3_covid", "Breach_Ratio", NINE, green=True),
    "6-8": tests("r3_covid", ["QR-IM", "LightGBM", "CatBoost", "FFNN", "LSTM"], ["CC_pval", "DQ_pval"]),
    "6-9": by_pair("r3_ratehike", "Breach_Ratio", NINE, green=True),
    "6-10": tests("r3_ratehike", ["QR-IM", "LightGBM", "CatBoost", "FFNN", "LSTM"], ["CC_pval", "DQ_pval"]),
}
(RES / "tables").mkdir(exist_ok=True)
for number, table in TABLES.items():
    table.to_csv(RES / "tables" / f"Table_{number}.csv", index=False)
print(f"results/tables/: Tables {', '.join(TABLES)}")
