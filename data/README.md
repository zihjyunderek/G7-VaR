# data/

The Bloomberg data are licensed and stay out of git (`.gitignore`). Put the workbooks here; `config.yaml` (`data:`) sets the paths.

| File | Sheets | Columns read |
| --- | --- | --- |
| `raw/old/G7_Spot.xlsx` | one per pair, named by its spot ticker: `EURUSD BGN Curncy`, `GBPUSD Curncy`, `USDJPY Curncy`, `USDCAD Curncy`, `AUDUSD Curncy`, `USDCHF Curncy`, `USDTWD Curncy` | `Date`, last price |
| `raw/new/<PAIR>.xlsx`, e.g. `EURUSD.xlsx` | one per series, named by its ticker `<PAIR><metric><tenor> Curncy`, e.g. `EURUSDV1W Curncy` (metric `V`, `25R`, `25B`, `10R`, `10B`; tenor `ON`, `1W`, `2W`, `1M`, `3M`, `6M`, `1Y`, `3Y`) | `Date`, last price, bid, ask |
| `raw/new/Global.xlsx` | `OVX Index` | `Date`, last price |

The workbooks are Bloomberg exports with the column headers of a Chinese-language terminal: last price `最新價`, bid `買價`, ask `賣價` (`PX_LAST`, `PX_BID`, `PX_ASK`), each with a sort-key suffix such as `(R1)`; `g7var/data.py` finds them by substring. A sheet without bid and ask is read with a bid-ask spread of zero. Implied volatilities are in volatility points.
