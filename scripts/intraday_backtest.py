"""Backtest intradia sobre los CSV de data/intraday y resumen en docs/intraday_report.json.

Uso: python scripts/intraday_backtest.py [--equity 200,10000] [--strategies orb,ema]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autotrader.intraday import SPECS, IntradayParams, backtest_intraday, load_bars  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/intraday")
    ap.add_argument("--equity", default="200,10000")
    ap.add_argument("--strategies", default="orb,ema")
    ap.add_argument("--out", default="docs/intraday_report.json")
    args = ap.parse_args()
    data = {}
    for path in sorted(glob.glob(os.path.join(args.data, "*_M15.csv"))):
        sym = os.path.basename(path).split("_")[0]
        if sym in SPECS:
            data[sym] = load_bars(path)
    if not data:
        print("Sin datos intradia en", args.data)
        return 1
    span = {sym: [str(df.index[0].date()), str(df.index[-1].date()), len(df)] for sym, df in data.items()}
    report = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "data": span, "runs": []}
    for strat in args.strategies.split(","):
        for eq in [float(x) for x in args.equity.split(",")]:
            p = IntradayParams(strategy=strat)
            res = backtest_intraday(data, p, initial=eq)
            m = res.metrics()
            by_sym = defaultdict(lambda: {"trades": 0, "pnl": 0.0, "wins": 0})
            for t in res.trades:
                b = by_sym[t.symbol]
                b["trades"] += 1
                b["pnl"] = round(b["pnl"] + t.pnl, 2)
                b["wins"] += 1 if t.pnl > 0 else 0
            reasons = defaultdict(int)
            for t in res.trades:
                reasons[t.reason] += 1
            run = {"strategy": strat, "equity": eq, "params": p.__dict__, "metrics": m, "by_symbol": dict(by_sym),
                   "exit_reasons": dict(reasons), "equity_curve": [[d.strftime("%Y-%m-%d"), round(float(v), 2)] for d, v in res.equity.items()]}
            report["runs"].append(run)
            print(f"\n=== {strat.upper()} con {eq:,.0f} USD ===")
            for k, v in m.items():
                print(f"  {k:>18}: {v}")
            for sym, b in sorted(by_sym.items(), key=lambda x: -x[1]["pnl"]):
                print(f"    {sym:>7}: {b['trades']:>3} ops  P&L {b['pnl']:>+9.2f}  acierto {b['wins'] / b['trades']:.0%}")
            print("  salidas:", dict(reasons))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print(f"\nInforme: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
