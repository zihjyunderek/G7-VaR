"""Chapter 7: the event studies behind Tables 7-1 to 7-3.

Table 7-1: mean ATM, RR25 and BF25 (1W) in the 10 trading days before and after four US tariff events.
Table 7-2: the 5 days before each event as a z-score against the 55 days before them (days -60 to -6).
Table 7-3: does RR25 Granger-cause USD/TWD returns around the 2025 central-bank intervention? F-test p-values, lags 1-5.
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g7var import ROOT  # noqa: E402
from g7var.data import option, spot  # noqa: E402

TARIFFS = {"2018-07-06: US-China tariff $34bn": "2018-07-06", "2019-05-10: US tariff raised to 25%": "2019-05-10",
           "2019-08-23: Tariff escalation (25%→30%)": "2019-08-23", "2025-04-02: Liberation Day universal tariffs": "2025-04-02"}
TARIFF_PAIRS = ["EUR/USD", "USD/JPY", "AUD/USD"]
IV = ["ATM", "RR25", "BF25"]
OUT = ROOT / "results" / "tables"


def frame(pair, features):
    s = spot(pair)
    return pd.concat([np.log(s / s.shift(1)).rename("return")] + [option(pair, f, "1W")[0] for f in features], axis=1).dropna()


shift, zscore = [], []
for event, day in TARIFFS.items():
    for pair in TARIFF_PAIRS:
        df = frame(pair, IV)
        pos = df.index.searchsorted(pd.Timestamp(day))
        row, z = {"Event": event[:30], "Pair": pair}, {"Event": event[:30], "Pair": pair}
        for c in IV:
            pre, post = df[c].iloc[max(0, pos - 10): pos].mean(), df[c].iloc[pos: pos + 10].mean()
            row |= {f"{c}_pre": round(float(pre), 3), f"{c}_post": round(float(post), 3), f"{c}_shift": round(float(post - pre), 3)}
            base = df[c].iloc[pos - 60: pos - 5]
            z[f"{c}_z"] = round(float((df[c].iloc[pos - 5: pos].mean() - base.mean()) / (base.std() + 1e-8)), 2)
        shift.append(row)
        zscore.append(z)
OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(shift).to_csv(OUT / "Table_7-1.csv", index=False)
pd.DataFrame(zscore).to_csv(OUT / "Table_7-2.csv", index=False)

twd = frame("USD/TWD", ["ATM", "RR25"]).loc["2024-10-01":"2025-06-30", ["return", "RR25"]]
with contextlib.redirect_stdout(io.StringIO()):          # statsmodels prints every lag
    gc = grangercausalitytests(twd, maxlag=5)
pd.DataFrame({"Lag": range(1, 6), "p_value": [round(gc[k][0]["ssr_ftest"][1], 4) for k in range(1, 6)]}).to_csv(OUT / "Table_7-3.csv", index=False)
print("results/tables/Table_7-1.csv, Table_7-2.csv, Table_7-3.csv")
