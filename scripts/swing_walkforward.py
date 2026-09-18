"""Walk-forward de la estrategia swing de bandas (H4, solo largos) -> docs/swing_walkforward.json.

Ventanas rodantes: se eligen los parametros en `train` meses y se miden en los `test` meses siguientes,
avanzando `step` meses. Los tramos de prueba encadenados forman el resultado fuera de muestra (OOS).
Se compara con la configuracion desplegada (stop 2 ATR, bandas 20/2, RSI < 30, 3 dias) medida en las
mismas ventanas de prueba, y se calcula la eficiencia OOS/IS del factor de beneficio.

Uso: python scripts/swing_walkforward.py [--train 4 --test 2 --step 2 --equity 10000]
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autotrader.swing import SwingParams, backtest_symbol, metrics  # noqa: E402

WARMUP_DAYS = 60  # historial previo para que EMA200/bandas esten formadas al empezar cada tramo
MIN_TRADES = 8

DEPLOYED = SwingParams("bands", "H4", stop_atr=2.0, allow_short=False, bb_std=2.0, bands_rsi=30.0, max_hold_days=3.0)


def grid() -> list[SwingParams]:
    out = []
    for sa, std, rsi, hold in itertools.product([1.5, 2.0, 2.5, 3.0], [2.0, 2.5], [30.0, 35.0], [2.0, 3.0, 4.0]):
        out.append(replace(DEPLOYED, stop_atr=sa, bb_std=std, bands_rsi=rsi, max_hold_days=hold))
    return out


def label(p: SwingParams) -> str:
    return f"stop {p.stop_atr} ATR · bandas {p.bb_period}/{p.bb_std} · RSI<{p.bands_rsi:.0f} · {p.max_hold_days:.0f} d"


def load(data_dir: str) -> dict[str, pd.DataFrame]:
    data = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*_M15.csv"))):
        df = pd.read_csv(path, parse_dates=["time"]).set_index("time")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        data[os.path.basename(path).split("_")[0]] = df
    return data


def run_window(data: dict[str, pd.DataFrame], p: SwingParams, start: pd.Timestamp, end: pd.Timestamp, equity: float) -> list:
    """Operaciones cuya entrada cae en [start, end); el backtest arranca WARMUP_DAYS antes para formar indicadores."""
    trades = []
    for sym, df in data.items():
        sub = df[(df.index >= start - pd.Timedelta(days=WARMUP_DAYS)) & (df.index < end)]
        if len(sub) < 500:
            continue
        trades += [t for t in backtest_symbol(sub, sym, p, equity) if start <= t.entry_time < end]
    trades.sort(key=lambda t: t.entry_time)
    return trades


def score(m: dict) -> float:
    """Criterio de seleccion en entrenamiento: factor de beneficio con un minimo de operaciones."""
    if m.get("trades", 0) < MIN_TRADES:
        return -1.0
    pf = m["profit_factor"]
    return (10.0 if pf == float("inf") else pf) + m["return"]


def summarize(trades: list, equity: float, days: float) -> dict:
    m = metrics(trades, equity, max(days, 1))
    if m.get("profit_factor") == float("inf"):
        m["profit_factor"] = None
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/intraday")
    ap.add_argument("--out", default="docs/swing_walkforward.json")
    ap.add_argument("--equity", type=float, default=10_000)
    ap.add_argument("--train", type=int, default=4, help="meses de entrenamiento")
    ap.add_argument("--test", type=int, default=2, help="meses de prueba")
    ap.add_argument("--step", type=int, default=2, help="avance entre ventanas, en meses")
    args = ap.parse_args()
    data = load(args.data)
    any_df = next(iter(data.values()))
    first = (any_df.index[0] + pd.Timedelta(days=WARMUP_DAYS)).normalize()
    last = any_df.index[-1]
    folds = []
    t0 = first
    while True:
        tr_end = t0 + pd.DateOffset(months=args.train)
        te_end = tr_end + pd.DateOffset(months=args.test)
        if te_end > last + pd.Timedelta(days=1):
            break
        folds.append((t0, tr_end, te_end))
        t0 = t0 + pd.DateOffset(months=args.step)
    if not folds:
        print("Datos insuficientes para al menos una ventana", file=sys.stderr)
        return 1
    configs = grid()
    results, oos_trades, base_trades = [], [], []
    for k, (a, b, c) in enumerate(folds, 1):
        train_days, test_days = (b - a).days, (c - b).days
        best, best_m, best_s = None, None, -9.0
        for p in configs:
            m = summarize(run_window(data, p, a, b, args.equity), args.equity, train_days)
            sc = score(m)
            if sc > best_s:
                best, best_m, best_s = p, m, sc
        test = run_window(data, best, b, c, args.equity)
        base = run_window(data, DEPLOYED, b, c, args.equity)
        oos_trades += test
        base_trades += base
        results.append({"fold": k, "train_window": [str(a.date()), str(b.date())], "test_window": [str(b.date()), str(c.date())],
                        "chosen": label(best), "chosen_params": {"stop_atr": best.stop_atr, "bb_std": best.bb_std, "bands_rsi": best.bands_rsi,
                                                                 "max_hold_days": best.max_hold_days},
                        "train": {**{"from": str(a.date()), "to": str(b.date())}, **best_m},
                        "test": {**{"from": str(b.date()), "to": str(c.date())}, **summarize(test, args.equity, test_days)},
                        "deployed_test": summarize(base, args.equity, test_days)})
        print(f"tramo {k}: entrena {a.date()}..{b.date()} -> {label(best)} PF {best_m.get('profit_factor')} | "
              f"prueba {b.date()}..{c.date()} PF {results[-1]['test'].get('profit_factor')} ret {results[-1]['test'].get('return')}")
    oos_days = (folds[-1][2] - folds[0][1]).days
    oos = summarize(sorted(oos_trades, key=lambda t: t.entry_time), args.equity, oos_days)
    base = summarize(sorted(base_trades, key=lambda t: t.entry_time), args.equity, oos_days)
    # in-sample de referencia: la configuracion desplegada sobre todo el periodo, como en la investigacion
    full = summarize(run_window(data, DEPLOYED, first, last, args.equity), args.equity, (last - first).days)
    pf_is, pf_oos = full.get("profit_factor"), base.get("profit_factor")
    efficiency = round(pf_oos / pf_is, 2) if pf_is and pf_oos else None
    chosen_stable = len({r["chosen"] for r in results}) == 1
    verdict = []
    if oos.get("trades", 0) < MIN_TRADES * len(folds) / 2:
        verdict.append("pocas operaciones fuera de muestra para concluir")
    if pf_oos and pf_oos > 1.2 and oos.get("profit_factor") and oos["profit_factor"] > 1.2:
        verdict.append("el borde sobrevive fuera de muestra")
    elif (pf_oos or 0) <= 1.0:
        verdict.append("la configuracion desplegada no gana fuera de muestra")
    else:
        verdict.append("borde debil fuera de muestra")
    if not chosen_stable:
        verdict.append("los parametros elegidos cambian entre tramos: sensibilidad alta")
    out = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "equity": args.equity, "risk_pct": DEPLOYED.risk_pct,
           "symbols": sorted(data), "windows": {"train_months": args.train, "test_months": args.test, "step_months": args.step, "folds": len(folds)},
           "grid_size": len(configs), "deployed": label(DEPLOYED), "folds": results,
           "oos_walkforward": oos, "oos_deployed": base, "insample_deployed": full, "efficiency_pf": efficiency,
           "chosen_stable": chosen_stable, "verdict": "; ".join(verdict)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    print(f"OOS walk-forward: {oos} | OOS desplegada: {base} | IS desplegada: {full} | eficiencia PF {efficiency} | {out['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
