#!/usr/bin/env python3
"""Generic backtesting.py runner.

Fetches daily OHLC data DIRECTLY from Yahoo's chart API with requests
(never yfinance -- yfinance crashes on Africa/Cairo DST timestamps),
parses it into the capitalised OHLC + DatetimeIndex format that
backtesting.py requires, and runs strategies on it.

Ticker, dates, cash, commission and dividend events are all CLI args /
an events JSON -- nothing is hardcoded beyond the pilot defaults below.

Pilot defaults: POUL.CA (Cairo Poultry, EGX), post-2019 data only
(the pre-2019 Yahoo segment for POUL.CA is corrupted: stale flat prices
at a x0.4115 scale), cash=100000, commission=0.001.

Strategies:
  1. SmaCross        -- buy when SMA10 crosses above SMA20, exit on cross below.
  2. DividendCapture -- buy at the close of the last session on/before each
                        event's "end" (entitlement end) date and sell at the
                        ex-day close. backtesting.py does NOT credit dividends,
                        so a dividend credit per trade is computed manually and
                        reported alongside the library's price-only stats.

Outputs: results/stats.json and results/report.md
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import requests
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

DEFAULT_EVENTS = (
    '[{"end":"2019-07-14","ex":"2019-07-15","div":0.25},'
    '{"end":"2021-05-17","ex":"2021-05-18","div":0.16},'
    '{"end":"2024-10-21","ex":"2024-10-22","div":0.847875},'
    '{"end":"2026-04-26","ex":"2026-04-27","div":0.33}]'
)


def fetch_yahoo_chart(ticker, start, end):
    """Fetch daily OHLC from Yahoo's chart API directly (no yfinance)."""
    period1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    period2 = int(pd.Timestamp(end, tz="UTC").timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={period1}&period2={period2}&interval=1d"
    )
    resp = requests.get(
        url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
                                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                                  "Chrome/124.0 Safari/537.36"},
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json()["chart"]["result"][0]
    ts = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    df = pd.DataFrame({
        "Open": quote["open"],
        "High": quote["high"],
        "Low": quote["low"],
        "Close": quote["close"],
        "Volume": quote["volume"],
    }, index=pd.to_datetime(ts, unit="s", utc=True).tz_convert(None).normalize())
    df.index.name = "Date"
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _sma(array, n):
    return pd.Series(array).rolling(n).mean()


class SmaCross(Strategy):
    n1 = 10
    n2 = 20

    def init(self):
        close = self.data.Close
        self.sma1 = self.I(_sma, close, self.n1)
        self.sma2 = self.I(_sma, close, self.n2)

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.position.close()


class DividendCapture(Strategy):
    # Resolved trading dates (strings 'YYYY-MM-DD'), passed via bt.run(...)
    buy_dates = ()
    sell_dates = ()

    def init(self):
        self._buy = {pd.Timestamp(d).date() for d in self.buy_dates}
        self._sell = {pd.Timestamp(d).date() for d in self.sell_dates}

    def next(self):
        today = self.data.index[-1].date()
        if today in self._sell and self.position:
            self.position.close()
        elif today in self._buy and not self.position:
            self.buy()


def _f(value):
    """JSON-safe float or None."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if (np.isnan(v) or np.isinf(v)) else round(v, 6)


def extract_stats(stats):
    return {
        "return_pct": _f(stats["Return [%]"]),
        "buy_and_hold_return_pct": _f(stats["Buy & Hold Return [%]"]),
        "win_rate_pct": _f(stats["Win Rate [%]"]),
        "num_trades": int(stats["# Trades"]),
        "max_drawdown_pct": _f(stats["Max. Drawdown [%]"]),
        "equity_final": _f(stats["Equity Final [$]"]),
    }


def resolve_event_dates(data, events):
    """For each event: buy = last session on/before 'end';
    sell = the ex-day session (first session on/after 'ex')."""
    idx = data.index
    resolved = []
    for ev in events:
        end = pd.Timestamp(ev["end"])
        ex = pd.Timestamp(ev["ex"])
        before = idx[idx <= end]
        after = idx[idx >= ex]
        if len(before) == 0 or len(after) == 0:
            resolved.append({**ev, "buy_date": None, "sell_date": None,
                             "entry_close": None, "exit_close": None})
            continue
        buy_date, sell_date = before[-1], after[0]
        resolved.append({
            **ev,
            "buy_date": buy_date.strftime("%Y-%m-%d"),
            "sell_date": sell_date.strftime("%Y-%m-%d"),
            "entry_close": _f(data.loc[buy_date, "Close"]),
            "exit_close": _f(data.loc[sell_date, "Close"]),
        })
    return resolved


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticker", default="POUL.CA")
    ap.add_argument("--start", default="2017-01-01",
                    help="fetch start (before the drop-before filter)")
    ap.add_argument("--end", default=None,
                    help="fetch end (default: today)")
    ap.add_argument("--drop-before", default="2019-01-01",
                    help="drop rows before this date (corrupted segment "
                         "for POUL.CA); pass empty string to disable")
    ap.add_argument("--cash", type=float, default=100000)
    ap.add_argument("--commission", type=float, default=0.001)
    ap.add_argument("--events", default=DEFAULT_EVENTS,
                    help="dividend events as a JSON list string")
    ap.add_argument("--events-file", default=None,
                    help="path to a JSON file with the dividend events "
                         "(overrides --events)")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    end = args.end or pd.Timestamp.now().strftime("%Y-%m-%d")
    events_raw = (open(args.events_file).read() if args.events_file
                  else args.events)
    events = json.loads(events_raw)

    print(f"Fetching {args.ticker} {args.start} -> {end} from Yahoo chart API...")
    data = fetch_yahoo_chart(args.ticker, args.start, end)
    rows_fetched = len(data)
    if args.drop_before:
        data = data[data.index >= pd.Timestamp(args.drop_before)]
    print(f"Rows fetched: {rows_fetched}; rows used: {len(data)} "
          f"({data.index[0].date()} -> {data.index[-1].date()})")

    os.makedirs(args.outdir, exist_ok=True)
    out = {
        "ticker": args.ticker,
        "data_source": "Yahoo chart API (direct requests, no yfinance)",
        "data_start": data.index[0].strftime("%Y-%m-%d"),
        "data_end": data.index[-1].strftime("%Y-%m-%d"),
        "rows_fetched": rows_fetched,
        "rows_used": len(data),
        "drop_before": args.drop_before or None,
        "cash": args.cash,
        "commission": args.commission,
        "strategies": {},
    }

    # ---- Strategy 1: SMA cross -------------------------------------------
    bt = Backtest(data, SmaCross, cash=args.cash,
                  commission=args.commission, exclusive_orders=True)
    stats = bt.run()
    sma_stats = extract_stats(stats)
    out["strategies"]["SmaCross"] = sma_stats
    print("SmaCross:", json.dumps(sma_stats))

    # ---- Strategy 2: Dividend capture ------------------------------------
    resolved = resolve_event_dates(data, events)
    usable = [r for r in resolved if r["buy_date"]]
    buy_dates = [r["buy_date"] for r in usable]
    sell_dates = [r["sell_date"] for r in usable]

    div_out = {"events": resolved, "feasible": False}
    try:
        bt2 = Backtest(data, DividendCapture, cash=args.cash,
                       commission=args.commission, exclusive_orders=True,
                       trade_on_close=True)
        stats2 = bt2.run(buy_dates=buy_dates, sell_dates=sell_dates)
        price_only = extract_stats(stats2)
        trades = stats2["_trades"]

        # Manual dividend credit: match each closed trade to its event by
        # entry/exit date, credit div * size, add to the trade's PnL.
        by_dates = {(r["buy_date"], r["sell_date"]): r for r in usable}
        total_div_credit = 0.0
        total_price_pnl = 0.0
        per_share_adj_sum = 0.0
        matched = 0
        trade_rows = []
        for _, tr in trades.iterrows():
            entry_t = pd.Timestamp(tr["EntryTime"]).strftime("%Y-%m-%d")
            exit_t = pd.Timestamp(tr["ExitTime"]).strftime("%Y-%m-%d")
            ev = by_dates.get((entry_t, exit_t))
            size = float(tr["Size"])
            pnl = float(tr["PnL"])
            total_price_pnl += pnl
            row = {"entry_time": entry_t, "exit_time": exit_t,
                   "entry_price": _f(tr["EntryPrice"]),
                   "exit_price": _f(tr["ExitPrice"]),
                   "size": _f(size), "price_pnl": _f(pnl)}
            if ev is not None:
                credit = float(ev["div"]) * size
                total_div_credit += credit
                per_share_adj_sum += (float(tr["ExitPrice"])
                                      - float(tr["EntryPrice"])
                                      + float(ev["div"]))
                matched += 1
                row["div_per_share"] = float(ev["div"])
                row["dividend_credit"] = _f(credit)
                row["adjusted_pnl"] = _f(pnl + credit)
            trade_rows.append(row)

        # Independent per-event check straight from the price series:
        manual_per_share = [
            {"buy_date": r["buy_date"], "sell_date": r["sell_date"],
             "div": r["div"], "entry_close": r["entry_close"],
             "exit_close": r["exit_close"],
             "per_share_exit_minus_entry_plus_div":
                 _f(r["exit_close"] - r["entry_close"] + r["div"])}
            for r in usable
        ]
        div_out.update({
            "feasible": True,
            "price_only_stats": price_only,
            "trades": trade_rows,
            "trades_matched_to_events": matched,
            "total_price_pnl_from_trades": _f(total_price_pnl),
            "total_dividend_credit": _f(total_div_credit),
            "dividend_adjusted_total_pnl": _f(total_price_pnl
                                              + total_div_credit),
            "dividend_adjusted_equity_final":
                _f(args.cash + total_price_pnl + total_div_credit),
            "per_share_sum_exit_minus_entry_plus_div": _f(per_share_adj_sum),
            "manual_per_event_per_share": manual_per_share,
            "manual_per_share_sum": _f(sum(
                r["per_share_exit_minus_entry_plus_div"]
                for r in manual_per_share)),
            "note": "backtesting.py does not credit dividends; the dividend "
                    "credit above is computed manually (div x shares per "
                    "matched trade) and added to the price-only PnL.",
        })
        print("DividendCapture price-only:", json.dumps(price_only))
        print(f"DividendCapture: {matched} trades matched, "
              f"dividend credit={total_div_credit:.2f}, "
              f"adjusted PnL={total_price_pnl + total_div_credit:.2f}")
    except Exception as exc:  # report plainly, SMA numbers still delivered
        div_out["reason"] = f"{type(exc).__name__}: {exc}"
        print("DividendCapture FAILED:", div_out["reason"])

    out["strategies"]["DividendCapture"] = div_out

    with open(os.path.join(args.outdir, "stats.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    lines = [
        f"# Backtest report -- {args.ticker}",
        "",
        f"- Data: Yahoo chart API (direct), {out['data_start']} -> "
        f"{out['data_end']} ({out['rows_used']} rows used, "
        f"{rows_fetched} fetched, dropped before {args.drop_before})",
        f"- Cash: {args.cash:,.0f}, commission: {args.commission}",
        "",
        "## SmaCross (SMA10 x SMA20)",
        "",
        f"- Return [%]: {sma_stats['return_pct']}",
        f"- Buy & Hold Return [%]: {sma_stats['buy_and_hold_return_pct']}",
        f"- Win Rate [%]: {sma_stats['win_rate_pct']}",
        f"- # Trades: {sma_stats['num_trades']}",
        f"- Max. Drawdown [%]: {sma_stats['max_drawdown_pct']}",
        f"- Equity Final: {sma_stats['equity_final']}",
        "",
        "## DividendCapture (buy at entitlement-end close, sell at ex close)",
        "",
    ]
    if div_out.get("feasible"):
        po = div_out["price_only_stats"]
        lines += [
            f"- Price-only Return [%] (library): {po['return_pct']}",
            f"- Price-only # Trades: {po['num_trades']} "
            f"(matched to events: {div_out['trades_matched_to_events']})",
            f"- Total dividend credit (manual): "
            f"{div_out['total_dividend_credit']}",
            f"- Dividend-adjusted total PnL (manual): "
            f"{div_out['dividend_adjusted_total_pnl']}",
            f"- Dividend-adjusted equity final (manual): "
            f"{div_out['dividend_adjusted_equity_final']}",
            f"- Per-share sum (exit - entry + div, matched trades): "
            f"{div_out['per_share_sum_exit_minus_entry_plus_div']}",
            "",
            "| Buy (end) | Sell (ex) | Div | Entry close | Exit close | "
            "Exit-Entry+Div / share |",
            "|---|---|---|---|---|---|",
        ]
        for r in div_out["manual_per_event_per_share"]:
            lines.append(
                f"| {r['buy_date']} | {r['sell_date']} | {r['div']} | "
                f"{r['entry_close']} | {r['exit_close']} | "
                f"{r['per_share_exit_minus_entry_plus_div']} |")
    else:
        lines.append(f"Not feasible in this run: "
                     f"{div_out.get('reason', 'unknown reason')}")
    lines.append("")
    with open(os.path.join(args.outdir, "report.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("Wrote", os.path.join(args.outdir, "stats.json"), "and report.md")


if __name__ == "__main__":
    main()
