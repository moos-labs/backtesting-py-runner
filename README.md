# backtesting-py-runner

Runs [backtesting.py](https://kernc.github.io/backtesting.py/) analyses
via GitHub Actions — commands supplied through the `COMMANDS` repository
variable. Part of the one-public-repo-per-tool runner pattern (see also
`vectorbt-runner`, `finbert-runner`): the workflow and scripts stay
generic and market-agnostic; the ticker and parameters are inputs, never
hardcoded EGX specifics in the workflow.

## How it works

1. Edit the **COMMANDS** repository variable
   (Settings → Secrets and variables → Actions → Variables), e.g.:

   ```sh
   mkdir -p results
   python scripts/backtest.py 2>&1 | tee results/run.log
   ```

2. Trigger the **Run analysis** workflow manually
   (Actions tab → Run workflow, or a `workflow_dispatch` API call).

3. Download the **`results`** artifact from the workflow run —
   it contains `results/stats.json`, `results/report.md` and
   `results/run.log`.

## What the script does

`scripts/backtest.py`:

- Fetches daily OHLC data **directly from Yahoo's chart API** with
  `requests` (never yfinance — yfinance crashes on Africa/Cairo DST
  timestamps), parsed into backtesting.py's required format:
  capitalised `Open/High/Low/Close/Volume` columns and a `DatetimeIndex`.
- Runs two strategies (cash and commission are CLI args):
  - **SmaCross** — buy when SMA10 crosses above SMA20, exit on cross below.
  - **DividendCapture** — events passed as a JSON arg
    (`--events` or `--events-file`): buy at the close of the last session
    on/before each event's entitlement-end date, sell at the ex-day close.
    backtesting.py does not credit dividends, so the dividend credit per
    trade is computed manually and reported next to the library's
    price-only stats.

### Pilot

Defaults reproduce the pilot: `POUL.CA` (Cairo Poultry), data from
2019-01-01 onwards (the pre-2019 Yahoo segment for this ticker is
corrupted and is dropped via `--drop-before 2019-01-01`), cash 100000,
commission 0.001, and four dividend events (2019, 2021, 2024, 2026).

```sh
python scripts/backtest.py --ticker POUL.CA
python scripts/backtest.py --ticker <OTHER> --drop-before "" --events "[]"
```
