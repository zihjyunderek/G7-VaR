# G7-VaR

Code and results of the master's thesis

> Huang, Zih Jyun (2026). *Forecasting FX Spot Value-at-Risk (VaR) Using Implied Volatility: Evidence from G7 Currency Pairs* [運用選擇權隱含波動率預測現貨風險值(VaR)：以 G7 貨幣對為例]. Master's thesis, Department of Money and Banking, National Chengchi University. Advisor: Ching-Ting Lin. [`thesis/thesis.pdf`](thesis/thesis.pdf)

Option prices carry the market's forecast of risk. The thesis asks whether that forecast, read through quantile regression and machine learning, gives a better one-day spot VaR than models that look back at returns only. It forecasts the 1%, 2.5% and 5% VaR of seven currency pairs (EUR/USD, GBP/USD, USD/JPY, USD/CAD, AUD/USD, USD/CHF, USD/TWD) with 13 models, backtests every forecast, and repeats the exercise in 18 robustness runs.

## Repository

```text
G7-VaR/
├── config.yaml            every setting: data files, sample, models, and the 19 runs
├── g7var/                 the library, one module per section of chapter 4
│   ├── data.py            3    Bloomberg workbooks → returns r_t, features at t, target r_{t+1}
│   ├── baselines.py       4.1  HS, RiskMetrics, GJR-GARCH, FHS-GARCH, FHS-IV
│   ├── qrim.py            4.2  QR-IM: quantile regression on implied volatility
│   ├── ensemble.py        4.3  quantile random forest, XGBoost, LightGBM, CatBoost
│   ├── deep.py            4.4  FFNN, RNN, LSTM with a monotone quantile head
│   ├── backtest.py        4.5  breach ratio, Kupiec, Christoffersen, DQ, Basel traffic light
│   └── pipeline.py             one run: every model on every pair
├── scripts/
│   ├── 01_forecasts.py    chapters 5 and 6: forecasts and backtests of every run
│   ├── 02_event_studies.py chapter 7: Tables 7-1 to 7-3
│   └── 03_tables.py       Tables 5-1 to 6-10
├── results/
│   ├── backtests.csv      every run × model × pair × quantile: breaches, sums, test statistics, Basel zone
│   ├── qrim_coefficients.csv   QR-IM coefficients and p-values of every run
│   ├── backtests_2026-05-04.csv the earlier main run behind Table 5-7 (see below)
│   └── tables/            Table_5-1.csv … Table_7-3.csv, one file per table of the thesis
├── tests/                 the backtests against their definitions; every printed thesis number against results/
├── thesis/                the thesis (NCCU edition) and the LaTeX of its result chapters, read by the tests
└── data/README.md         the Bloomberg data the code expects (the data are not in the repository)
```

## Data availability

All market data come from Bloomberg and are licensed; they cannot be redistributed and are not in this repository. Anyone with a Bloomberg Terminal can download them: daily `PX_LAST`, `PX_BID` and `PX_ASK` from 2009-01-01 to 2025-12-31 for

| Series | Bloomberg tickers |
| --- | --- |
| Spot rates | `EURUSD BGN Curncy`, `GBPUSD Curncy`, `USDJPY Curncy`, `USDCAD Curncy`, `AUDUSD Curncy`, `USDCHF Curncy`, `USDTWD Curncy` |
| Implied volatility | `<PAIR><metric><tenor> Curncy`: metric `V` (ATM), `25R`, `10R` (risk reversals), `25B`, `10B` (butterflies); tenor `ON`, `1W`, `2W`, `1M`, `3M`, `6M`, `1Y`, `3Y` |
| Oil volatility | `OVX Index` (CBOE crude oil volatility, for USD/CAD) |

[`data/README.md`](data/README.md) gives the workbook layout the code reads. Nothing derived day by day from these data is committed: `results/` holds aggregated backtests and coefficients only, and the figures of the thesis that plot the series are left out.

## Reproduce

Python 3.11. The pinned versions are those of the thesis run; PyTorch is the one for your hardware (the CUDA build for a GPU).

```bash
pip install -r requirements.txt
python scripts/01_forecasts.py          # all 19 runs: about an hour on a laptop GPU (RTX 4050)
python scripts/02_event_studies.py      # Tables 7-1 to 7-3
python scripts/03_tables.py             # Tables 5-1 to 6-10
pytest                                  # 40 tests; see below
```

`01_forecasts.py --runs main_1W` runs chapter 5 alone; `--models HS QR-IM` limits a run to some models. Forecasts are written to `results/runs/<run>/` (`predictions/` stays local, being derived from the licensed data), and the runs replace their rows in `results/backtests.csv` and `results/qrim_coefficients.csv`. Table 5-7 comes from the earlier specification of HS and RiskMetrics: `G7VAR_CONFIG=config_2026-05-04.yaml python scripts/01_forecasts.py --runs main_1W`.

## Tables

| Table | Content | Run | Script |
| --- | --- | --- | --- |
| 5-1 | QR-IM coefficients | `main_1W` | 01, 03 |
| 5-2 to 5-4 | breach ratios, 1% and all quantiles | `main_1W` | 01, 03 |
| 5-5 | LSTM quantile collapse, USD/CAD | `main_1W` | 01, 03 |
| 5-6, 5-7 | sum if breach, sum if no breach | `main_1W`; 5-7 the run of 2026-05-04 | 01, 03 |
| 5-8, 5-9 | conditional coverage, DQ and Kupiec p-values | `main_1W` | 01, 03 |
| 5-10 | Basel traffic light | `main_1W` | 01, 03 |
| 6-1, 6-2 | QR-IM with RR10; with BF25 | `r1_RR10`, `r1_BF25` | 01, 03 |
| 6-3 | five features | `r1_All` | 01, 03 |
| 6-4, 6-5 | oil volatility for USD/CAD; option bid-ask spread | `r4_OVX_USDCAD`, `r5_ImpliedSpread` | 01, 03 |
| 6-6 | option tenor 1W, 6M, 1Y, 3Y | `main_1W`, `r2_*` | 01, 03 |
| 6-7 to 6-10 | COVID (2020–2021) and rate-hike (2022–2025) subsamples | `r3_covid`, `r3_ratehike` | 01, 03 |
| 7-1 to 7-3 | 2018–2025 tariff events; 2025 USD/TWD intervention | — | 02 |

The runs (`config.yaml`): `main_1W` (1-week options, ATM and 25-delta risk reversal, out of sample 2020–2025); `r1_*` six feature sets; `r2_*` seven tenors; `r3_*` two subsamples; `r4` and `r5` two added features; `r0_replication` the sample of de Lange et al. (2022) and Blom et al. (2023), out of sample 2018 to September 2020.

## What the results are, and how exactly they reproduce

`results/` holds the run behind the thesis (2026-06-04), made with the earlier code of this repository (v0.2.3). `tests/test_thesis.py` reads every result table of the thesis from its LaTeX and checks each printed number against `results/`: 23 tables, 1,892 values, all within rounding. Three printed values are known exceptions, none changing a conclusion: Table 6-8 prints CC p = 0.001595 as .001 and DQ p = 0.999928 as .999, and Table 5-7 prints 12.17496 as 12.18. Table 5-7 was taken from an earlier run (2026-05-04) whose HS used 500 days ending at r_{t-1} and whose RiskMetrics was an EWMA recursion; `config_2026-05-04.yaml` reproduces it and `results/backtests_2026-05-04.csv` keeps it.

The code here is that v0.2.3 code rewritten for reading. On one machine with one set of libraries both give identical forecasts and backtests, checked model by model in `main_1W`: HS, RiskMetrics, GJR-GARCH, FHS-GARCH, FHS-IV, QR-IM, XGBoost and LightGBM on all seven pairs; random forest and CatBoost on three; FFNN on two; RNN and LSTM on one. The same check passed in `r0_replication`, `r1_All`, `r2_ON`, `r3_covid`, `r4_OVX_USDCAD`, `r5_ImpliedSpread` and the 2026-05-04 specification, for a selection of models and pairs. Across machines and library versions: HS, RiskMetrics, FHS-IV and the event studies reproduce to the last digit; GARCH and QR-IM rest on numerical optimisation, so their statistics can move in the fourth decimal and, in a few cells, a breach count by one; the tree ensembles and networks are seeded, but GPU training is not bit-reproducible from run to run.

## Known departures of the backtests

The backtests are kept as they produced the published numbers; `tests/test_backtest.py` marks three departures from the literature as strict expected failures.

| Test | Literature | As coded |
| --- | --- | --- |
| Kupiec | with no breach, LR = −2T ln(1 − α), a rejection | NaN (7 cells, robustness runs only, at 1%) |
| Basel traffic light | defined for 99% VaR | the 99% thresholds are applied at 97.5% and 95% too, so a well-calibrated 95% VaR turns red |
| Dynamic quantile | Wald test that every coefficient, the intercept included, is zero, χ²(K + 2) (Engle and Manganelli, 2004) | the intercept is left out, χ²(K + 1): DQ tests whether breaches cluster or follow the VaR level, not unconditional coverage |

## References

- Blom, H. M., de Lange, P. E. and Risstad, M. (2023). Estimating value-at-risk in the EURUSD currency cross from implied volatilities using machine learning methods and quantile regression. *Journal of Risk and Financial Management*, 16(7), 312.
- Christoffersen, P. F. (1998). Evaluating interval forecasts. *International Economic Review*, 39(4), 841–862.
- de Lange, P. E., Risstad, M. and Westgaard, S. (2022). Estimating value-at-risk using quantile regression and implied moments. *Journal of Risk Model Validation*, 16(4), 53–76.
- Engle, R. F. and Manganelli, S. (2004). CAViaR: Conditional autoregressive value at risk by regression quantiles. *Journal of Business and Economic Statistics*, 22(4), 367–381.
- Koenker, R. and Bassett, G. (1978). Regression quantiles. *Econometrica*, 46(1), 33–50.
- Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*, 3(2), 73–84.

## License

Code: BSD 3-Clause (`LICENSE.txt`). The thesis in `thesis/` is © 2026 Zih Jyun Huang. Market data: Bloomberg, not included.
