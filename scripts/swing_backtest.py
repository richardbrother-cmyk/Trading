"""Barrido swing (H1/H4, 1-3 dias) sobre data/intraday/*_M15.csv -> docs/swing_report.json.

Uso: python scripts/swing_backtest.py [--equity 10000]
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autotrader.swing import SwingParams, backtest_symbol, metrics  # noqa: E402


def label(p: SwingParams) -> str:
    base = f"{p.strategy} {p.timeframe} stop{p.stop_atr}"
    if p.strategy == "pullback":
        base += f" tp{p.tp_atr} rsi{p.rsi_entry:.0f}"
    if p.strategy == "breakout":
        base += f" n{p.breakout_bars}"
    return base + (" L+S" if p.allow_short else " solo L")


def configs() -> list[SwingParams]:
    out = []
    for tf in ["H1", "H4"]:
        for sa, ta, sh, rs in itertools.product([1.5, 2.0], [2.0, 3.0], [True, False], [35.0, 40.0]):
            out.append(SwingParams("pullback", tf, stop_atr=sa, tp_atr=ta, allow_short=sh, rsi_entry=rs))
        for sa, nb, sh in itertools.product([1.5, 2.0], [20, 30], [True, False]):
            out.append(SwingParams("breakout", tf, stop_atr=sa, tp_atr=99.0, allow_short=sh, breakout_bars=nb))
        for sa, sh in itertools.product([1.5, 2.0], [True, False]):
            out.append(SwingParams("bands", tf, stop_atr=sa, allow_short=sh))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/intraday")
    ap.add_argument("--equity", type=float, default=10_000)
    ap.add_argument("--out", default="docs/swing_report.json")
    args = ap.parse_args()
    data = {}
    for path in sorted(glob.glob(os.path.join(args.data, "*_M15.csv"))):
        df = pd.read_csv(path, parse_dates=["time"]).set_index("time")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        data[os.path.basename(path).split("_")[0]] = df
    any_df = next(iter(data.values()))
    days = (any_df.index[-1] - any_df.index[0]).days
    edges = pd.date_range(any_df.index[0].normalize(), any_df.index[-1], periods=5, tz="UTC")
    rows = []
    for p in configs():
        trades = []
        per = {}
        for sym, df in data.items():
            tr = backtest_symbol(df, sym, p, args.equity)
            per[sym] = metrics(tr, args.equity, days)
            trades += tr
        trades.sort(key=lambda t: t.entry_time)
        # Cartera: una sola cuenta de `equity` que arriesga el 1 % en cada operacion de cualquier simbolo
        port = metrics(trades, args.equity, days)
        quarters = []
        for k in range(4):
            sub = [t for t in trades if edges[k] <= t.entry_time < edges[k + 1]]
            quarters.append({"from": str(edges[k].date()), "trades": len(sub), "pnl": round(sum(t.pnl for t in sub), 2),
                             "profit_factor": metrics(sub, args.equity, 90).get("profit_factor") if sub else None})
        monthly = {}
        for t in trades:
            key = t.exit_time.strftime("%Y-%m")
            monthly[key] = round(monthly.get(key, 0.0) + t.pnl, 2)
        rows.append({"config": label(p), "params": p.__dict__, "portfolio": port, "by_symbol": per, "quarters": quarters, "monthly": monthly})
    rows.sort(key=lambda r: -(r["portfolio"].get("profit_factor", 0) if r["portfolio"].get("profit_factor", 0) != float("inf") else 0))
    report = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "equity": args.equity, "days": days,
              "data": {s: [str(d.index[0].date()), str(d.index[-1].date()), len(d)] for s, d in data.items()}, "runs": rows}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1, default=str)
    print(f"{'configuración':<40} {'ops':>4} {'acierto':>7} {'PF':>5} {'R medio':>7} {'retorno':>8} {'DD':>7} {'días':>5}")
    for r in rows[:12]:
        m = r["portfolio"]
        print(f"{r['config']:<40} {m['trades']:>4} {m['win_rate']:>7.0%} {m['profit_factor']:>5} {m['avg_r']:>7} {m['return']:>+8.1%} {m['max_drawdown']:>7.1%} {m['avg_hold_days']:>5}")
    print(f"Informe: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
