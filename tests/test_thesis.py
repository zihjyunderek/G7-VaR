"""Every result table of the thesis (Tables 5-1 to 7-3), checked against results/.

The tables are parsed from the LaTeX of the thesis (thesis/tex/) and each printed value is
compared with the result it reports: a value printed with d decimals passes when the result
lies within half a unit of the d-th decimal (half-up and half-even rounding both pass).
Coefficient tables also check the significance stars. A last test fails if a table of
chapters 5 to 7 has no check. Chapter 3 is computed from the licensed data and is not checked.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "thesis" / "tex"
RESULTS = ROOT / "results"
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CAD", "AUD/USD", "USD/CHF", "USD/TWD"]
MODELS = {"HS", "RiskMetrics", "GJR-GARCH", "FHS-GARCH", "FHS-IV", "QR-IM", "RandomForest",
          "XGBoost", "LightGBM", "CatBoost", "FFNN", "RNN", "LSTM"}
ALIASES = {"RF": "RandomForest"}
COEF_SCALE = 100          # coefficients are printed x 100 (de Oliveira et al. 2022 convention)


# ── LaTeX table parsing ─────────────────────────────────────────────────────
@lru_cache(maxsize=None)
def _tex(chapter: str) -> str:
    return (TEX / chapter).read_text(encoding="utf-8")


def _strip_group(s: str, cmd: str) -> str:
    """Remove \\cmd{...} including its braced argument (nested braces allowed)."""
    out, i = [], 0
    while True:
        j = s.find("\\" + cmd + "{", i)
        if j < 0:
            return "".join(out) + s[i:]
        out.append(s[i:j])
        k, depth = j + len(cmd) + 2, 1
        while depth and k < len(s):
            depth += {"{": 1, "}": -1}.get(s[k], 0)
            k += 1
        i = k


def _unwrap(s: str, cmd: str, nargs: int) -> str:
    """Replace \\cmd{a1}...{an} by its last argument."""
    pat = re.compile(r"\\" + cmd + r"(\{[^{}]*\}){" + str(nargs - 1) + r"}\{((?:[^{}]|\{[^{}]*\})*)\}")
    while True:
        new = pat.sub(lambda m: m.group(2), s)
        if new == s:
            return s
        s = new


def clean(cell: str) -> str:
    """A table cell as plain text: formatting removed, stars kept."""
    s = cell.strip()
    for cmd in ("hline", "cline", "cmidrule", "midrule", "toprule", "bottomrule"):
        s = re.sub(r"\\" + cmd + r"(\([^)]*\))?(\{[^}]*\})?", " ", s)
    s = _strip_group(s, "phantom")
    s = _unwrap(s, "multirow", 3)
    s = _unwrap(s, "multicolumn", 3)
    for cmd in ("textbf", "textit", "textsuperscript", "mathrm"):
        s = _unwrap(s, cmd, 1)
    s = s.replace("$-$", "-").replace("$+$", "+").replace("\\%", "%").replace("\\ ", " ")
    s = s.replace("\\,", " ").replace("\\ddagger", "‡").replace("\\dag", "†")
    s = re.sub(r"[${}^]", "", s)
    return re.sub(r"\s+", " ", s).strip()


# A tabular nested in a cell (two-line event names in chapter 7) would end the table early
NESTED = re.compile(r"(?<=\{)\\begin\{tabular\}(?:\[[^\]]*\])?\{(?:[^{}]|\{[^{}]*\})*\}(.*?)\\end\{tabular\}", re.S)


def table(chapter: str, label: str) -> list[list[str]]:
    """Data rows of the table with this label: cells cleaned, header rows dropped.

    A row that starts a new panel (after \\hline) gets a leading "\\hline" marker cell.
    """
    tex = NESTED.sub(lambda m: m.group(1).replace("\\\\", " "), _tex(chapter))
    start = tex.index("\\label{" + label + "}")
    end = min(i for i in (tex.find("\\end{tabular}", start), tex.find("\\end{longtable}", start)) if i > 0)
    rows = []
    for raw in tex[start:end].split("\\\\"):
        if "&" not in raw:
            continue
        raw = raw.split("\\label{" + label + "}")[-1]
        raw = re.sub(r"(?<!\\)%[^\n]*", "", raw)
        panel = "\\hline" in raw.split("&")[0] and rows
        cells = [clean(c) for c in raw.split("&")]
        if "\\begin{" in raw or sum(p in cells for p in PAIRS) >= 3 or "係數" in cells or "Sig." in cells:
            continue  # header row
        rows.append((["\\hline"] if panel else []) + cells)
    return rows


NUM = re.compile(r"[-+]?(?:\d+\.\d+|\.\d+|\d+)")


def number(cell: str) -> tuple[float, int] | None:
    """The printed number and its decimals; None for a dash or an empty cell."""
    m = NUM.search(cell.replace(",", ""))
    if not m:
        return None
    txt = m.group(0)
    return float(txt), len(txt.split(".")[1]) if "." in txt else 0


def stars(cell: str) -> str:
    return "*" * cell.count("*")


def stars_for(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# ── results ──────────────────────────────────────────────────────────
@lru_cache(maxsize=None)
def result(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name, keep_default_na=True)


MAY_RUN = "backtests_2026-05-04.csv"   # source of Table 5-7 only


def full(run: str, model: str, pair: str, q: str, source: str = "backtests.csv") -> pd.Series:
    m = result(source)
    row = m[(m.run == run) & (m.Model == model) & (m.Pair == pair) & (m.Quantile == q)]
    assert len(row) == 1, f"{run} {model} {pair} {q}: {len(row)} rows in results/backtests.csv"
    return row.iloc[0]


def coef(run: str, pair: str, q: str) -> pd.Series:
    m = result("qrim_coefficients.csv")
    row = m[(m.run == run) & (m.Pair == pair) & (m.Quantile == q)]
    assert len(row) == 1, f"{run} {pair} {q}: {len(row)} rows in results/qrim_coefficients.csv"
    return row.iloc[0]


def quantile(cell: str) -> str:
    v = re.search(r"(\d+(?:\.\d+)?)\s*%", cell)
    assert v, f"no quantile in {cell!r}"
    return f"{float(v.group(1)):.1f}%"


def model_of(cell: str) -> str | None:
    name = ALIASES.get(cell, cell)
    return name if name in MODELS else None


# Printed values that differ from the result, with the reason. Each entry
# must still differ, so the list cannot go stale; see docs/validation.md.
KNOWN = {
    ("tab:r3a_backtest", "r3_covid FFNN GBP/USD CC_pval"):
        "0.001595 printed as .001; rounded it is .002 (significant either way)",
    ("tab:r3a_backtest", "r3_covid CatBoost USD/TWD DQ_pval"):
        "0.999928 printed as .999 instead of 1.000 (a cap, not a rounding)",
    ("tab:sum_if_no_breach", "main_1W RandomForest USD/TWD 1.0% Sum_if_NoBreach"):
        "12.17496 printed as 12.18; rounded it is 12.17",
}


class Check:
    """Collects printed-versus-result comparisons for one table."""

    def __init__(self, label: str):
        self.label, self.n, self.bad, self.known = label, 0, [], set()

    def value(self, where: str, printed: str, actual: float, scale: float = 1.0) -> None:
        got = number(printed)
        if got is None:
            return
        v, d = got
        self.n += 1
        if pd.isna(actual) or abs(actual * scale - v) > 0.5 * 10 ** -d + 1e-9:
            if (self.label, where) in KNOWN:
                self.known.add(where)
            else:
                self.bad.append(f"{where}: thesis {printed!r}, results {actual * scale!r}")

    def text(self, where: str, printed: str, actual: str) -> None:
        self.n += 1
        if printed != actual:
            self.bad.append(f"{where}: thesis {printed!r}, results {actual!r}")

    def done(self, minimum: int) -> None:
        assert self.n >= minimum, f"{self.label}: only {self.n} values parsed"
        stale = {w for (lab, w) in KNOWN if lab == self.label} - self.known
        assert not stale, f"{self.label}: listed in KNOWN but now equal: {sorted(stale)}"
        assert not self.bad, f"{self.label}: {len(self.bad)} of {self.n} values differ\n" + "\n".join(self.bad[:20])


# ── tables of models by pairs ───────────────────────────────────────────────
MODEL_TABLES = [
    # label, chapter, run, quantile (None: from the q column), column, scale, minimum values[, results file]
    ("tab:breach_1pct", "chapter05.tex", "main_1W", "1.0%", "Breach_Ratio", 1, 13 * 7),
    ("tab:breach_all", "chapter05.tex", "main_1W", None, "Breach_Ratio", 1, 13 * 21),
    ("tab:breach_all_q", "chapter05.tex", "main_1W", None, "Breach_Ratio", 1, 3 * 21),
    ("tab:sum_if_breach", "chapter05.tex", "main_1W", "1.0%", "Sum_if_Breach", 1, 13 * 7),
    # Table 5-7 was built from the 2026-05-04 run; every other table from the 2026-06-04 run
    ("tab:sum_if_no_breach", "chapter05.tex", "main_1W", "1.0%", "Sum_if_NoBreach", 1, 13 * 7, MAY_RUN),
    ("tab:backtests_allq", "chapter05.tex", "main_1W", None, "Kupiec_pval", 1, 6 * 21),
    ("tab:all_breach", "chapter06.tex", "r1_All", "1.0%", "Breach_Ratio", 1, 10 * 7),
    ("tab:r3a_breach", "chapter06.tex", "r3_covid", "1.0%", "Breach_Ratio", 1, 10 * 7),
    ("tab:r3b_breach", "chapter06.tex", "r3_ratehike", "1.0%", "Breach_Ratio", 1, 10 * 7),
]


@pytest.mark.parametrize("spec", MODEL_TABLES, ids=[t[0] for t in MODEL_TABLES])
def test_model_by_pair_table(spec):
    label, chapter, run, q, col, scale, minimum, *src = spec
    source = src[0] if src else "backtests.csv"
    chk, model = Check(label), None
    for cells in table(chapter, label):
        cells = [c for c in cells if c != "\\hline"]
        if model_of(cells[0]):
            model = model_of(cells[0])
        elif cells[0] and not re.search(r"\d\s*%", cells[0]):
            continue  # a nominal-level or other label row
        rest = cells[1:]
        qq = q
        if q is None:
            qq, rest = quantile(rest[0] if not re.search(r"\d\s*%", cells[0]) else cells[0]), rest[1:]
        for pair, printed in zip(PAIRS, rest[:7]):
            chk.value(f"{run} {model} {pair} {qq} {col}", printed, full(run, model, pair, qq, source)[col], scale)
        if len(rest) > 7 and "/7" in rest[7]:  # Basel column "k/7 G": pairs in the green zone
            k = int(rest[7].split("/")[0])
            green = sum(full(run, model, p, "1.0%").Basel_Zone == "Green" for p in PAIRS)
            chk.text(f"{run} {model} Basel green count", str(k), str(green))
    chk.done(minimum)


def test_cc_dq_kupiec_at_1pct():
    """Table 5.x (tab:cctest): panels A (CC), B (DQ) and C (Kupiec), main run, 1%."""
    tex = _tex("chapter05.tex")
    blk = tex[tex.index("\\label{tab:cctest}"):tex.index("\\end{tabular}", tex.index("\\label{tab:cctest}"))]
    chk = Check("tab:cctest")
    for col, panel in zip(("CC_pval", "DQ_pval", "Kupiec_pval"), re.split(r"Panel [ABC]", blk)[1:]):
        for raw in panel.split("\\\\"):
            cells = [clean(c) for c in raw.split("&")]
            if len(cells) != 8 or not model_of(cells[0]):
                continue
            for pair, printed in zip(PAIRS, cells[1:]):
                chk.value(f"main_1W {cells[0]} {pair} {col}", printed, full("main_1W", model_of(cells[0]), pair, "1.0%")[col])
    chk.done(3 * 6 * 7)


def test_basel_zones_at_1pct():
    chk = Check("tab:basel")
    for cells in table("chapter05.tex", "tab:basel"):
        cells = [c for c in cells if c != "\\hline"]
        model = model_of(cells[0])
        if not model:
            continue
        for pair, printed in zip(PAIRS, cells[1:8]):
            chk.text(f"main_1W {model} {pair} Basel", printed[:1], full("main_1W", model, pair, "1.0%").Basel_Zone[:1])
    chk.done(13 * 7)


@pytest.mark.parametrize("label,run", [("tab:r3a_backtest", "r3_covid"), ("tab:r3b_backtest", "r3_ratehike")])
def test_subsample_cc_and_dq(label, run):
    """Two panels, CC then DQ, separated by a rule."""
    chk, panel = Check(label), -1
    cols = ("CC_pval", "DQ_pval")
    for cells in table("chapter06.tex", label):
        if cells[0] == "\\hline" or panel < 0:
            panel += 1
        cells = [c for c in cells if c != "\\hline"]
        model = model_of(cells[0])
        for pair, printed in zip(PAIRS, cells[1:8]):
            chk.value(f"{run} {model} {pair} {cols[panel]}", printed, full(run, model, pair, "1.0%")[cols[panel]])
    chk.done(2 * 5 * 7 - 4)


def test_tenor_comparison():
    runs = {"1W": "main_1W", "6M": "r2_6M", "1Y": "r2_1Y", "3Y": "r2_3Y"}
    chk, run = Check("tab:tenor_comparison"), None
    for cells in table("chapter06.tex", "tab:tenor_comparison"):
        cells = [c for c in cells if c != "\\hline"]
        if cells[0]:
            tag = re.match(r"(1W|6M|1Y|3Y)", cells[0])
            if not tag:
                continue  # nominal level
            run = runs[tag.group(1)]
        model = model_of(cells[1])
        for pair, printed in zip(PAIRS, cells[2:9]):
            chk.value(f"{run} {model} {pair} Breach_Ratio", printed, full(run, model, pair, "1.0%").Breach_Ratio)
    chk.done(4 * 5 * 7)


def test_lstm_quantile_collapse():
    chk = Check("tab:lstm_collapse")
    for cells in table("chapter05.tex", "tab:lstm_collapse"):
        cells = [c for c in cells if c != "\\hline"]
        r = full("main_1W", "LSTM", "USD/CAD", quantile(cells[0]))
        chk.value("LSTM USD/CAD N_Breach", cells[1], r.N_Breach)
        chk.value("LSTM USD/CAD Breach_Ratio", cells[2], r.Breach_Ratio)
    chk.done(6)


# ── coefficient tables ──────────────────────────────────────────────────────
def _sig(r: pd.Series, var: str) -> str:
    s = r.get(f"{var}_sig")
    return s if isinstance(s, str) else stars_for(r[f"{var}_pval"])


@pytest.mark.parametrize("label,chapter,run,groups", [
    ("tab:qrim_coef", "chapter05.tex", "main_1W", {"截距": "Const", "ATM IV": "ATM", "RR25": "RR25"}),
    ("tab:rr10_coef", "chapter06.tex", "r1_RR10", {"截距": "Const", "ATM IV": "ATM", "RR10": "RR10"}),
])
def test_coefficients(label, chapter, run, groups):
    chk, var = Check(label), None
    for cells in table(chapter, label):
        cells = [c for c in cells if c != "\\hline"]
        if cells[0]:
            var = groups[cells[0]]
        q = quantile(cells[1])
        for pair, printed in zip(PAIRS, cells[2:9]):
            r = coef(run, pair, q)
            chk.value(f"{run} {pair} {q} {var}", printed, r[var], COEF_SCALE)
            chk.text(f"{run} {pair} {q} {var} stars", stars(printed), _sig(r, var))
    chk.done(2 * 2 * 21)


def test_bf25_wald_pvalues():
    chk = Check("tab:bf25_coef")
    for cells in table("chapter06.tex", "tab:bf25_coef"):
        cells = [c for c in cells if c != "\\hline"]
        q = quantile(cells[0])
        for pair, printed in zip(PAIRS, cells[1:8]):
            chk.value(f"r1_BF25 {pair} {q} BF25_pval", printed, coef("r1_BF25", pair, q).BF25_pval)
    chk.done(21)


def test_ovx_extension():
    chk = Check("tab:r4_coef")
    for cells in table("chapter06.tex", "tab:r4_coef"):
        cells = [c for c in cells if c != "\\hline"]
        r = coef("r4_OVX_USDCAD", "USD/CAD", quantile(cells[0]))
        for var, printed, sig in zip(("ATM", "RR25", "OVX"), cells[1::2], cells[2::2]):
            chk.value(f"USD/CAD {cells[0]} {var}", printed, r[var], COEF_SCALE)
            chk.text(f"USD/CAD {cells[0]} {var} stars", stars(sig), _sig(r, var))
    chk.done(18)


def test_implied_spread_extension():
    chk = Check("tab:r5_coef")
    for cells in table("chapter06.tex", "tab:r5_coef"):
        cells = [c for c in cells if c != "\\hline"]
        if cells[0] not in PAIRS:
            continue
        for q, printed, sig in zip(("1.0%", "2.5%", "5.0%"), cells[1::2], cells[2::2]):
            r = coef("r5_ImpliedSpread", cells[0], q)
            chk.value(f"{cells[0]} {q} spread", printed, r.Implied_Bid_Ask_Spread, COEF_SCALE)
            chk.text(f"{cells[0]} {q} spread stars", stars(sig), _sig(r, "Implied_Bid_Ask_Spread"))
    chk.done(42)


# ── event studies ───────────────────────────────────────────────────────────
EVENT_TABLES = {"event_04_tariff_iv_shift": "tables/Table_7-1.csv", "event_04_tariff_zscore": "tables/Table_7-2.csv",
                "event_05_twd_granger": "tables/Table_7-3.csv"}


def events(source: str) -> pd.DataFrame:
    """One event-study table of chapter 7 (scripts/02_event_studies.py)."""
    return result(EVENT_TABLES[source])


@pytest.mark.parametrize("label,source,cols", [
    ("tab:tariff_ivshift", "event_04_tariff_iv_shift",
     ["ATM_pre", "ATM_post", "ATM_shift", "RR25_pre", "RR25_post", "RR25_shift", "BF25_pre", "BF25_post", "BF25_shift"]),
    ("tab:tariff_zscore", "event_04_tariff_zscore", ["ATM_z", "RR25_z", "BF25_z"]),
])
def test_tariff_events(label, source, cols):
    ev, chk, event = events(source), Check(label), None
    for cells in table("chapter07.tex", label):
        cells = [c for c in cells if c != "\\hline"]
        if cells[0]:
            event = re.match(r"\d{4}-\d{2}-\d{2}", cells[0]).group()
        r = ev[(ev.Event.astype(str).str.startswith(event)) & (ev.Pair == cells[1])]
        assert len(r) == 1, f"{label}: {event} {cells[1]} has {len(r)} rows in results/tables"
        for col, printed in zip(cols, cells[2:]):
            chk.value(f"{event} {cells[1]} {col}", printed, r.iloc[0][col])
    chk.done(12 * len(cols))  # four events by three pairs


def test_twd_granger():
    ev, chk = events("event_05_twd_granger"), Check("tab:twd_granger")
    for cells in table("chapter07.tex", "tab:twd_granger"):
        cells = [c for c in cells if c != "\\hline"]
        r = ev[ev.Lag == int(cells[0])]
        chk.value(f"lag {cells[0]}", cells[1], r.iloc[0].p_value)
    chk.done(5)


# ── every result table is covered ───────────────────────────────────────────
COVERED = {t[0] for t in MODEL_TABLES} | {
    "tab:cctest", "tab:basel", "tab:r3a_backtest", "tab:r3b_backtest", "tab:tenor_comparison",
    "tab:lstm_collapse", "tab:qrim_coef", "tab:rr10_coef", "tab:bf25_coef", "tab:r4_coef",
    "tab:r5_coef", "tab:tariff_ivshift", "tab:tariff_zscore", "tab:twd_granger",
}


def test_every_result_table_is_checked():
    """Chapters 5 to 7 and the appendix: no table without a check."""
    found = set()
    for ch in ("chapter05.tex", "chapter06.tex", "chapter07.tex", "chapter09_appendix.tex"):
        found |= set(re.findall(r"^[^%\n]*\\label\{(tab:[^}]+)\}", _tex(ch), flags=re.M))
    assert found - COVERED == set(), f"tables without a check: {sorted(found - COVERED)}"
    assert COVERED - found == set(), f"checks for tables not in the thesis: {sorted(COVERED - found)}"
