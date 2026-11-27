"""Chapter 3: Bloomberg workbooks to one frame per pair: returns, implied-volatility features, r_{t+1}.

The forecast for day t+1 uses what is known at t, so every forecast is compared with
`return_fwd` = r_{t+1} (de Lange et al., 2022, eq. 4).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from g7var import CFG, ROOT


@lru_cache(maxsize=None)
def read_sheet(path: str, sheet: str) -> pd.DataFrame:
    """One Bloomberg sheet: last price, and bid and ask when the sheet has them (NaN otherwise)."""
    raw = pd.read_excel(ROOT / path, sheet_name=sheet)
    raw["Date"] = pd.to_datetime(raw["Date"])
    find = lambda key: next((c for c in raw.columns if key in c), None)   # Bloomberg adds sort keys such as "(R1)"
    last, bid, ask = find("最新價"), find("買價"), find("賣價")
    if last is None:
        raise KeyError(f"no last-price column in sheet {sheet!r} of {path}")
    cols = {last: "last_price"} | ({bid: "bid", ask: "ask"} if bid and ask else {})
    df = (raw[["Date", *cols]].rename(columns=cols).sort_values("Date").reset_index(drop=True)
          .dropna(subset=["last_price"]).set_index("Date"))
    if "bid" not in df:
        df["bid"] = df["ask"] = np.nan
    return df


def _in_sample(s, end=None):
    return s[(s.index >= CFG["sample"]["train_start"]) & (s.index <= (end or CFG["sample"]["test_end"]))]


def spot(pair: str) -> pd.Series:
    code, sheet = CFG["pairs"][pair]
    return _in_sample(read_sheet(CFG["data"]["spot_file"], sheet)["last_price"])


def option(pair: str, feature: str, tenor: str = "1W") -> tuple[pd.Series, pd.Series]:
    """An implied-volatility series (vol points) and its bid-ask spread (0 where the sheet has no bid and ask)."""
    code = CFG["pairs"][pair][0]
    df = _in_sample(read_sheet(f"{CFG['data']['option_dir']}/{code}.xlsx", f"{code}{CFG['features'][feature]}{tenor} Curncy"))
    return df["last_price"].rename(feature), (df["ask"] - df["bid"]).fillna(0.0).rename("Implied_Bid_Ask_Spread")


def ovx(end: str | None = None) -> pd.Series:
    """CBOE crude-oil volatility (OVX), forward-filled over FX days without an oil quote."""
    return _in_sample(read_sheet(CFG["data"]["global_file"], "OVX Index")["last_price"].rename("OVX"), end).sort_index().ffill()


@lru_cache(maxsize=None)
def _frame(pair: str, tenor: str, features: tuple, test_end: str) -> pd.DataFrame:
    s = spot(pair)
    ret = np.log(s / s.shift(1)).rename("return")
    iv, spreads = zip(*(option(pair, f, tenor) for f in features)) if features else ((), ())
    spread = spreads[0] if spreads else pd.Series(0.0, index=ret.index, name="Implied_Bid_Ask_Spread")
    df = pd.concat([ret, *iv, spread], axis=1).dropna()        # spread of the first feature (ATM): the liquidity proxy
    df["Implied_Bid_Ask_Spread"] = df["Implied_Bid_Ask_Spread"].fillna(0.0)
    df = _in_sample(df, test_end).sort_index()
    if pair == "USD/CAD":
        df = df.join(ovx(test_end), how="left")
        df["OVX"] = df["OVX"].ffill().bfill()
    if pair == "GBP/USD":                                       # the 2022 mini-budget
        df["Mini_Budget_Crisis"] = ((df.index >= "2022-09-01") & (df.index <= "2022-10-31")).astype(float)
        if "BF25" in df:
            df["Crisis_x_BF25"] = df["Mini_Budget_Crisis"] * df["BF25"]
    df["return_fwd"] = df["return"].shift(-1)
    return df.dropna(subset=["return_fwd"])


def master(pair: str, tenor: str = "1W", features=("ATM", "RR25"), test_end: str | None = None) -> pd.DataFrame:
    """Returns r_t, the implied-volatility features, the ATM bid-ask spread, pair extras, and r_{t+1}."""
    feats = tuple(f for f in features if f in CFG["features"])
    return _frame(pair, tenor, feats, test_end or CFG["sample"]["test_end"]).copy()


def split(df: pd.DataFrame, run: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[(df.index >= CFG["sample"]["train_start"]) & (df.index <= run["train_end"])]
    test = df[(df.index >= run["test_start"]) & (df.index <= run["test_end"])]
    return train, test
