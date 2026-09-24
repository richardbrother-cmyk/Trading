"""Investigacion: cerrar posiciones "en resistencia" y reevaluar despues.

Pregunta: en el bot tendencial diario (cruce SMA 20/50 con filtro RSI, stop 3 %), conviene cerrar un largo cuando el
precio llega al maximo de las ultimas N barras (una resistencia) y volver a entrar mas tarde? Se prueba sobre dos
universos: el de la cuenta demo de cTrader (oro, metales, energia, agricolas e indices; datos diarios de Yahoo con
los futuros equivalentes) y el de la cuenta paper de Alpaca (acciones, metales y agricolas en ETF).

Variantes: resistencia = maximo de 20, 60 o 120 barras anteriores, con tolerancia 0 o 1 %, cierre solo si la
posicion gana; reentrada (a) solo con un cruce alcista nuevo, (b) tras 5 dias si la tendencia sigue, (c) cuando un
cierre supera la resistencia. Referencia: la estrategia tal cual. Metricas en todo el periodo y por ano.

Uso: python scripts/resistance_exit.py [--out docs/resistance_exit.json]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from autotrader.backtest import run_backtest  # noqa: E402
from autotrader.data import yahoo_bars  # noqa: E402
from autotrader.risk import RiskParams  # noqa: E402
from autotrader.strategy import StrategyParams  # noqa: E402

# Cuenta demo de cTrader: futuros de Yahoo equivalentes a cada CFD
CTRADER_PROXIES = {"XAUUSD": "GC=F", "XAGUSD": "SI=F", "XPTUSD": "PL=F", "XTIUSD": "CL=F", "XBRUSD": "BZ=F", "XNGUSD": "NG=F",
                   "COFARA": "KC=F", "COTTON": "CT=F", "WHEAT": "ZW=F", "CORN": "ZC=F", "SUGAR": "SB=F", "US500": "ES=F", "NAS100": "NQ=F"}
CTRADER_CACHE = os.path.join(ROOT, "data/research/ctrader_daily_3y.csv")
ALPACA_CACHE = os.path.join(ROOT, "data/research/alpaca_daily_2y.csv")

UNIVERSES = {
    "ctrader_demo": {"initial": 10_000.0, "risk": RiskParams(0.02, 8, 0.12, 0.03, 0.03, 5.0), "label": "Cuenta demo cTrader (13 CFD, 10.000 USD, 8 posiciones, 12 % x5)"},
    "alpaca_paper": {"initial": 100_000.0, "risk": RiskParams(0.02, 10, 0.25, 0.03, 0.05, 1.0), "label": "Cuenta paper Alpaca (13 ETF y acciones, 100.000 USD, stop 5 %)"},
}
STRATEGY = StrategyParams(20, 50, 14, 70.0)


def load_ctrader() -> dict[str, pd.DataFrame]:
    if os.path.exists(CTRADER_CACHE):
        df = pd.read_csv(CTRADER_CACHE, parse_dates=["date"])
    else:
        frames = []
        for sym, y in CTRADER_PROXIES.items():
            d = yahoo_bars(y, range_="3y").reset_index()
            d["symbol"] = sym
            frames.append(d)
        df = pd.concat(frames)
        os.makedirs(os.path.dirname(CTRADER_CACHE), exist_ok=True)
        df.to_csv(CTRADER_CACHE, index=False)
    return {s: g.drop(columns="symbol").set_index("date").sort_index() for s, g in df.groupby("symbol")}


def load_alpaca() -> dict[str, pd.DataFrame]:
    df = pd.read_csv(ALPACA_CACHE, parse_dates=["date"])
    return {s: g.drop(columns="symbol").set_index("date").sort_index() for s, g in df.groupby("symbol")}


def by_year(res) -> dict:
    eq = res.equity
    out = {}
    for y, g in eq.groupby(eq.index.year):
        start = eq[eq.index < g.index[0]]
        base = float(start.iloc[-1]) if len(start) else res.initial_cash
        out[str(y)] = round(float(g.iloc[-1]) / base - 1, 4)
    return out


def summarize(res, name: str, params: dict) -> dict:
    m = res.metrics()
    if m.get("profit_factor") == float("inf"):
        m["profit_factor"] = None
    reasons = {}
    for t in res.trades:
        reasons[t.reason] = reasons.get(t.reason, 0) + 1
    return {"name": name, "params": params, **m, "by_year": by_year(res), "exits": reasons}


def variants() -> list[tuple[str, dict]]:
    out = [("referencia", {})]
    names = {"signal": "reentrada con cruce nuevo", "cooldown": "reentrada tras 5 dias", "breakout": "reentrada al superar la resistencia"}
    for lb, tol, re in itertools.product([20, 60, 120], [0.0, 0.01], ["signal", "cooldown", "breakout"]):
        out.append((f"resistencia {lb} d, tolerancia {tol:.0%}, {names[re]}",
                    {"resistance_lookback": lb, "resistance_tol": tol, "resistance_reentry": re, "reentry_cooldown_days": 5 if re == "cooldown" else 0}))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/resistance_exit.json")
    args = ap.parse_args()
    data = {"ctrader_demo": load_ctrader(), "alpaca_paper": load_alpaca()}
    out = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "strategy": "SMA 20/50, RSI < 70, ejecucion en la apertura siguiente",
           "universes": {}}
    for uni, cfg in UNIVERSES.items():
        d = data[uni]
        rows = []
        for name, kw in variants():
            res = run_backtest(d, STRATEGY, cfg["risk"], initial_cash=cfg["initial"], **kw)
            rows.append(summarize(res, name, kw))
        base = rows[0]
        for r in rows[1:]:
            r["vs_reference"] = {"return": round(r["total_return"] - base["total_return"], 4), "max_drawdown": round(r["max_drawdown"] - base["max_drawdown"], 4),
                                 "years_better": sum(1 for y in r["by_year"] if r["by_year"][y] > base["by_year"].get(y, 0))}
        better = [r for r in rows[1:] if r["vs_reference"]["return"] > 0 and r["vs_reference"]["max_drawdown"] >= 0]
        out["universes"][uni] = {"label": cfg["label"], "symbols": sorted(d), "period": [str(min(x.index[0] for x in d.values()).date()), str(max(x.index[-1] for x in d.values()).date())],
                                 "reference": base, "variants": rows[1:], "better_return_and_drawdown": [r["name"] for r in better]}
        print(f"\n== {cfg['label']} ==")
        print(f"{'variante':70s} {'ops':>4s} {'retorno':>8s} {'DD':>7s} {'PF':>6s} {'por ano':>40s}")
        for r in rows:
            print(f"{r['name'][:70]:70s} {r['trades']:4d} {r['total_return']:+8.1%} {r['max_drawdown']:+7.1%} {str(r['profit_factor']):>6s} "
                  + " ".join(f"{y[2:]}:{v:+.0%}" for y, v in r["by_year"].items()))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    print("->", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
