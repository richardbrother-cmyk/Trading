"""Investigacion: variantes del bot de tendencia de Alpaca sobre 2 anos de barras diarias.

Ideas tomadas de "151 Trading Strategies" (Kakushadze & Serur, 2018):
1. Filtro de regimen (4.1.2, dual momentum): solo se abren largos si SPY cierra sobre su SMA de N dias.
2. Tamano por volatilidad (4.6 / 6.5 / 10.4): peso de cada posicion = objetivo de volatilidad diaria / volatilidad del
   activo (desviacion tipica de los retornos diarios de 20 sesiones), con el tope habitual del 12 % por posicion.
3. Salida por caida diaria (3.12, ec. 323): cerrar un largo si el cierre cae mas de X % respecto al cierre anterior.

Uso: python scripts/alpaca_variants.py [--data data/research/alpaca_daily_2y.csv] [--out docs/alpaca_variants.json]
Los datos se cachean en data/research/ (Yahoo, 2 anos) para que el estudio sea reproducible.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autotrader.backtest import run_backtest  # noqa: E402
from autotrader.risk import RiskParams  # noqa: E402
from autotrader.strategy import StrategyParams  # noqa: E402

SYMBOLS = "SPY,QQQ,AAPL,MSFT,NVDA,GLD,SLV,PPLT,PALL,USO,WEAT,CORN,DBA".split(",")


def load(path: str) -> dict[str, pd.DataFrame]:
    if not os.path.exists(path):
        from autotrader.data import yahoo_bars
        frames = {s: yahoo_bars(s, range_="2y") for s in SYMBOLS}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.concat(frames, names=["symbol", "date"]).to_csv(path)
    raw = pd.read_csv(path, parse_dates=["date"])
    return {s: g.drop(columns="symbol").set_index("date").sort_index() for s, g in raw.groupby("symbol")}


def regime_series(spy: pd.DataFrame, sma: int) -> pd.Series:
    return (spy["close"] > spy["close"].rolling(sma).mean()).fillna(False)


def vol_sizer(data: dict[str, pd.DataFrame], target_daily: float, cap_pct: float, window: int = 20):
    """qty = equity * min(cap, objetivo / sigma) / precio, acotado al efectivo. sigma usa datos hasta el cierre anterior."""
    sigma = {s: df["close"].pct_change().rolling(window).std().shift(1) for s, df in data.items()}

    def size(symbol, date, equity, cash, price):
        sg = sigma[symbol].get(date)
        if sg is None or not (sg > 0) or price <= 0 or cash <= 0:
            return 0
        pct = min(cap_pct, target_daily / float(sg))
        return int(max(min(equity * pct, cash) // price, 0))

    return size


def metrics_split(equity: pd.Series, initial: float, trades) -> dict:
    """Metricas por mitades del periodo (consistencia)."""
    mid = equity.index[len(equity) // 2]
    out = {}
    for name, eq in (("primera_mitad", equity[:mid]), ("segunda_mitad", equity[mid:])):
        r = eq.iloc[-1] / eq.iloc[0] - 1
        dd = (eq / eq.cummax() - 1).min()
        out[name] = {"return": round(float(r), 4), "max_drawdown": round(float(dd), 4)}
    return out


def buy_and_hold(spy: pd.DataFrame, initial: float, start, end) -> dict:
    px = spy["close"][(spy.index >= start) & (spy.index <= end)]
    eq = initial * px / px.iloc[0]
    rets = eq.pct_change().dropna()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    return {"total_return": round(float(eq.iloc[-1] / initial - 1), 4), "cagr": round(float((eq.iloc[-1] / initial) ** (1 / years) - 1), 4),
            "sharpe": round(float(math.sqrt(252) * rets.mean() / rets.std()), 2), "max_drawdown": round(float((eq / eq.cummax() - 1).min()), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/research/alpaca_daily_2y.csv")
    ap.add_argument("--out", default="docs/alpaca_variants.json")
    ap.add_argument("--equity", type=float, default=100_000)
    args = ap.parse_args()
    data = load(args.data)
    strat = StrategyParams(20, 50, 14, 70)
    risk = RiskParams(0.02, 10, 0.12, 0.03, 0.05, 1.0)
    spy = data["SPY"]

    variants = [("base", "Configuración actual (cruce SMA 20/50, RSI < 70, stop 5 %, 12 % por posición)", {})]
    for n in (200, 100):
        reg = regime_series(spy, n)
        variants.append((f"regimen_spy_sma{n}", f"Filtro de régimen: solo entra si SPY > SMA {n}", {"entry_allowed": lambda d, r=reg: bool(r.get(d, False))}))
    reg200 = regime_series(spy, 200)
    variants.append(("regimen_spy_sma200_salida", "Régimen SPY > SMA 200 con salida de todo cuando SPY cae bajo la media",
                     {"entry_allowed": lambda d: bool(reg200.get(d, False)), "regime_exit": True}))
    for t in (0.0010, 0.0012, 0.0015):
        variants.append((f"vol_{int(t*10000)}pb", f"Tamaño por volatilidad: objetivo {t*100:.2f} % diario por posición (tope 12 %)",
                         {"size_fn": vol_sizer(data, t, 0.12)}))
    for x in (0.02, 0.03):
        variants.append((f"caida_{int(x*100)}pct", f"Salida si el cierre cae más de {x*100:.0f} % en el día", {"drop_exit_pct": x}))
        variants.append((f"caida_{int(x*100)}pct_con_ganancia", f"Salida por caída de {x*100:.0f} % solo si la posición gana",
                         {"drop_exit_pct": x, "drop_exit_only_in_profit": True}))
    for cd in (5, 10):
        variants.append((f"caida_3pct_espera{cd}d", f"Salida por caída de 3 % y sin reentrar en {cd} días (salvo cruce nuevo)",
                         {"drop_exit_pct": 0.03, "reentry_cooldown_days": cd}))
        variants.append((f"caida_3pct_con_ganancia_espera{cd}d", f"Salida por caída de 3 % solo con ganancia, sin reentrar en {cd} días",
                         {"drop_exit_pct": 0.03, "drop_exit_only_in_profit": True, "reentry_cooldown_days": cd}))
    variants.append(("stop_espera5d", "Solo espera de 5 días tras un stop antes de reentrar (sin regla de caída)", {"reentry_cooldown_days": 5}))
    variants.append(("regimen200_vol12", "Régimen SPY > SMA 200 + tamaño por volatilidad 0,12 %",
                     {"entry_allowed": lambda d: bool(reg200.get(d, False)), "size_fn": vol_sizer(data, 0.0012, 0.12)}))
    variants.append(("regimen200_vol12_caida3g", "Régimen + volatilidad 0,12 % + salida por caída 3 % con ganancia",
                     {"entry_allowed": lambda d: bool(reg200.get(d, False)), "size_fn": vol_sizer(data, 0.0012, 0.12),
                      "drop_exit_pct": 0.03, "drop_exit_only_in_profit": True}))

    rows = []
    for key, desc, kw in variants:
        res = run_backtest(data, strat, risk, initial_cash=args.equity, **kw)
        m = res.metrics()
        reasons = pd.Series([t.reason for t in res.trades]).value_counts().to_dict()
        rows.append({"key": key, "description": desc, **m, "exits": reasons, "halves": metrics_split(res.equity, args.equity, res.trades)})
        print(f"{key:28s} ret {m['total_return']*100:6.1f}%  cagr {m['cagr']*100:5.1f}%  sharpe {m['sharpe']:4.2f}  dd {m['max_drawdown']*100:6.1f}%  "
              f"ops {m['trades']:3d}  pf {m['profit_factor']:5.2f}  mitades {rows[-1]['halves']['primera_mitad']['return']*100:5.1f}% / {rows[-1]['halves']['segunda_mitad']['return']*100:5.1f}%")
    start, end = rows[0] and run_backtest(data, strat, risk, initial_cash=args.equity).equity.index[[0, -1]]
    bh = buy_and_hold(spy, args.equity, start, end)
    print("SPY buy&hold", bh)
    out = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "period": [str(start.date()), str(end.date())],
           "symbols": SYMBOLS, "equity": args.equity, "benchmark_spy": bh, "variants": rows,
           "source": "Kakushadze & Serur (2018), 151 Trading Strategies, secciones 3.12, 4.1.2, 4.6, 6.5 y 10.4"}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
