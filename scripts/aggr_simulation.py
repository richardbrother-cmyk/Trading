"""Simulacion de la cuenta agresiva (500 USD) con los parametros que operan hoy, sobre todo el historico M15 disponible.

1. Se generan las operaciones de la estrategia (ruptura H4, stop 0,75 ATR, objetivo 6R, break even tras 2 R, 7 dias) por
   simbolo con una cuenta grande, para obtener la secuencia historica de resultados en R (incluye spread, comision y swap).
2. Se simula una sola cuenta de 500 USD que recorre esa secuencia en orden: riesgo 6 % del equity por operacion, lote
   minimo del simbolo (se descarta la operacion si el lote minimo arriesga mas del 9 %), maximo 3 posiciones a la vez y
   freno del 30 % de caida desde el maximo (cuando salta, deja de abrir posiciones hasta que alguien lo rearme; aqui se
   supone que no se rearma).
3. Monte Carlo: se repite la misma agenda de operaciones (fechas, solapes, distancias al stop) barajando los resultados
   en R con reemplazo, para ver el abanico de trayectorias que la misma estrategia y los mismos parametros pueden dar.

Uso: python scripts/aggr_simulation.py [--paths 2000] [--out docs/aggr_simulation.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autotrader.intraday import SPECS, SymbolSpec  # noqa: E402
from autotrader.swing import SwingParams, backtest_symbol  # noqa: E402

# Spread tipico estimado (en unidades de precio) para simbolos sin especificacion fija; el lote minimo, el paso y los
# decimales se leen de data/intraday/symbols.json (volcado del broker por scripts/fetch_intraday.py).
SPREAD_ESTIMATES = {"CORN": 1.0, "WHEAT": 1.0, "COFARA": 0.4, "USCOCOA": 15.0, "COTTON": 0.10, "SUGAR": 0.03,
                    "XAGUSD": 0.03, "XPTUSD": 2.0, "XBRUSD": 0.03, "XNGUSD": 0.006}


def ensure_specs(symbols: list[str], data_dir: str, spread_mult: float = 1.0) -> None:
    """Registra en SPECS los simbolos que falten usando symbols.json del broker y el spread estimado."""
    path = os.path.join(data_dir, "symbols.json")
    broker = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    for sym in symbols:
        if sym in SPECS and spread_mult == 1.0:
            continue
        base = SPECS.get(sym)
        info = broker.get(sym)
        if base is None and info is None:
            raise SystemExit(f"{sym}: sin especificacion (ni en SPECS ni en {path})")
        spread = (base.spread if base else SPREAD_ESTIMATES.get(sym, 0.0)) * spread_mult
        SPECS[sym] = SymbolSpec(sym, "00:00", "24:00", spread, info["step_units"] if info else base.step,
                                info["min_units"] if info else base.min_units, info["digits"] if info else base.digits)

PARAMS = dict(strategy="breakout", timeframe="H4", stop_atr=0.75, tp_atr=4.5, pure_rr=True, allow_short=False, max_hold_days=7.0,
              breakout_bars=20, breakeven_r=2.0, breakeven_lock_r=0.1)
ACCOUNT = dict(initial=500.0, risk_pct=0.06, max_risk_pct=0.09, max_positions=3, max_drawdown_pct=0.30)


def load(data_dir: str, symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
    out = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*_M15.csv"))):
        sym = os.path.basename(path).split("_")[0]
        if symbols and sym not in symbols:
            continue
        df = pd.read_csv(path, parse_dates=["time"]).set_index("time")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        out[sym] = df
    return out


def historical_trades(data: dict[str, pd.DataFrame]) -> list[dict]:
    p = SwingParams(**PARAMS, risk_pct=0.01, max_risk_pct=0.5)
    rows = []
    for sym, df in data.items():
        for t in backtest_symbol(df, sym, p, 10_000.0):
            rows.append({"symbol": sym, "entry_time": t.entry_time, "exit_time": t.exit_time, "entry": t.entry,
                         "dist": abs(t.entry - t.stop), "r": t.r, "reason": t.reason})
    rows.sort(key=lambda r: r["entry_time"])
    return rows


def simulate(trades: list[dict], r_values: np.ndarray, grid: pd.DatetimeIndex, brake: bool = True, account: dict | None = None) -> dict:
    """Recorre las operaciones en orden con una cuenta de 500 USD. `r_values[i]` es el resultado en R de la operacion i."""
    a = account or ACCOUNT
    equity, peak = a["initial"], a["initial"]
    open_pos: list[tuple[pd.Timestamp, float]] = []  # (salida, pnl)
    events: list[tuple[pd.Timestamp, float]] = []  # (momento, equity tras cerrar)
    taken = skipped_lot = skipped_full = 0
    brake_at = None
    max_dd = 0.0
    for i, t in enumerate(trades):
        # cierra lo que vence antes de esta entrada
        still = []
        for exit_time, pnl in sorted(open_pos):
            if exit_time <= t["entry_time"]:
                equity += pnl
                peak = max(peak, equity)
                max_dd = min(max_dd, equity / peak - 1)
                events.append((exit_time, equity))
            else:
                still.append((exit_time, pnl))
        open_pos = still
        if brake_at is not None or equity <= 0:
            continue
        if equity <= peak * (1 - a["max_drawdown_pct"]):
            if brake:
                brake_at = t["entry_time"]
                continue
            brake_at_soft = True  # noqa: F841 - sin freno: se sigue operando
        if len(open_pos) >= a["max_positions"]:
            skipped_full += 1
            continue
        spec = SPECS[t["symbol"]]
        units = np.floor(equity * a["risk_pct"] / t["dist"] / spec.step) * spec.step
        if units < spec.min_units:
            if spec.min_units * t["dist"] <= equity * a["max_risk_pct"]:
                units = spec.min_units
            else:
                skipped_lot += 1
                continue
        pnl = float(r_values[i]) * units * t["dist"]
        open_pos.append((t["exit_time"], pnl))
        taken += 1
    for exit_time, pnl in sorted(open_pos):
        equity += pnl
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / peak - 1)
        events.append((exit_time, equity))
    # curva sobre la rejilla temporal (equity cerrado, sin resultado abierto)
    ser = pd.Series({k: v for k, v in events})
    ser = ser[~ser.index.duplicated(keep="last")].sort_index()
    curve = ser.reindex(ser.index.union(grid)).ffill().reindex(grid).fillna(a["initial"]).to_numpy()
    return {"final": float(equity), "max_dd": float(max_dd), "taken": taken, "skipped_lot": skipped_lot, "skipped_full": skipped_full,
            "brake_at": None if brake_at is None else str(brake_at)[:10], "curve": curve}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/intraday")
    ap.add_argument("--paths", type=int, default=2000)
    ap.add_argument("--out", default="docs/aggr_simulation.json")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tp-r", type=float, default=6.0, help="objetivo en multiplos del stop (6 = configuracion actual)")
    ap.add_argument("--breakeven-r", type=float, default=2.0)
    ap.add_argument("--symbols", default="US500,NAS100,XAUUSD,XTIUSD,EURUSD,GBPUSD", help="universo (coma)")
    ap.add_argument("--spread-mult", type=float, default=1.0, help="multiplicador del spread (sensibilidad a costes)")
    args = ap.parse_args()
    PARAMS["tp_atr"] = round(PARAMS["stop_atr"] * args.tp_r, 4)
    PARAMS["breakeven_r"] = args.breakeven_r
    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    ensure_specs(symbols, args.data, args.spread_mult)
    data = load(args.data, symbols)
    missing = [x for x in symbols if x not in data]
    if missing:
        print("sin datos para", missing)
    trades = historical_trades(data)
    if not trades:
        print("sin operaciones"); return 1
    start = min(df.index[0] for df in data.values()).normalize()
    end = max(df.index[-1] for df in data.values()).normalize()
    grid = pd.date_range(start, end, freq="W-FRI", tz="UTC")
    if len(grid) == 0 or grid[-1] < end:
        grid = grid.append(pd.DatetimeIndex([end]))
    R = np.array([t["r"] for t in trades])
    years = {}
    for t in trades:
        y = str(t["entry_time"].year)
        years.setdefault(y, []).append(t["r"])
    by_year = {y: {"trades": len(v), "sum_r": round(float(np.sum(v)), 1), "win": round(float(np.mean(np.array(v) > 0)), 2)} for y, v in years.items()}
    months = [d.strftime("%Y-%m-%d") for d in grid]
    scenarios = {}
    for key, brake in (("freno", True), ("sin_freno", False)):
        hist = simulate(trades, R, grid, brake=brake)
        rng = np.random.default_rng(args.seed)
        curves, finals, dds, brakes, brake_months = [], [], [], 0, []
        for _ in range(args.paths):
            res = simulate(trades, rng.choice(R, size=len(R), replace=True), grid, brake=brake)
            curves.append(res["curve"]); finals.append(res["final"]); dds.append(res["max_dd"])
            if res["brake_at"] is not None:
                brakes += 1; brake_months.append((pd.Timestamp(res["brake_at"], tz="UTC") - start).days / 30.44)
        C = np.array(curves); finals = np.array(finals)
        pct = lambda q: np.percentile(C, q, axis=0)  # noqa: E731
        scenarios[key] = {
            "brake": brake,
            "historical": {"final": round(hist["final"], 2), "return": round(hist["final"] / ACCOUNT["initial"] - 1, 4), "max_dd": round(hist["max_dd"], 4),
                           "taken": hist["taken"], "skipped_lot": hist["skipped_lot"], "skipped_full": hist["skipped_full"], "brake_at": hist["brake_at"],
                           "curve": [round(float(x), 2) for x in hist["curve"]]},
            "fan": {"p5": pct(5).round(2).tolist(), "p25": pct(25).round(2).tolist(), "p50": pct(50).round(2).tolist(),
                    "p75": pct(75).round(2).tolist(), "p95": pct(95).round(2).tolist()},
            "samples": [C[i].round(2).tolist() for i in rng.choice(len(C), size=12, replace=False)],
            "stats": {"final_p10": round(float(np.percentile(finals, 10)), 0), "final_p50": round(float(np.median(finals)), 0),
                      "final_p90": round(float(np.percentile(finals, 90)), 0), "p_loss": round(float((finals < ACCOUNT["initial"]).mean()), 3),
                      "p_half": round(float((finals < ACCOUNT["initial"] / 2).mean()), 3), "p_brake": round(brakes / args.paths, 3),
                      "brake_month_p50": round(float(np.median(brake_months)), 1) if brake_months else None,
                      "p_double": round(float((finals >= 2 * ACCOUNT["initial"]).mean()), 3), "p_x5": round(float((finals >= 5 * ACCOUNT["initial"]).mean()), 3),
                      "max_dd_p50": round(float(np.median(dds)), 4), "max_dd_p90": round(float(np.percentile(dds, 10)), 4)}}
    # Sensibilidad al riesgo por operacion (mismas senales, mismo freno): que cambia si se arriesga menos
    sweep = []
    for risk in (0.02, 0.03, 0.04, 0.06):
        acc = dict(ACCOUNT, risk_pct=risk, max_risk_pct=max(0.03, 1.5 * risk))
        for brake in (True, False):
            rng = np.random.default_rng(args.seed)
            finals, dds, brakes = [], [], 0
            for _ in range(min(args.paths, 600)):
                res = simulate(trades, rng.choice(R, size=len(R), replace=True), grid, brake=brake, account=acc)
                finals.append(res["final"]); dds.append(res["max_dd"]); brakes += res["brake_at"] is not None
            finals = np.array(finals); h = simulate(trades, R, grid, brake=brake, account=acc)
            sweep.append({"risk_pct": risk, "brake": brake, "hist_final": round(h["final"], 0), "hist_max_dd": round(h["max_dd"], 3),
                          "final_p10": round(float(np.percentile(finals, 10)), 0), "final_p50": round(float(np.median(finals)), 0),
                          "final_p90": round(float(np.percentile(finals, 90)), 0), "p_loss": round(float((finals < ACCOUNT["initial"]).mean()), 3),
                          "p_half": round(float((finals < ACCOUNT["initial"] / 2).mean()), 3), "p_brake": round(brakes / len(finals), 3),
                          "max_dd_p50": round(float(np.median(dds)), 3)})
    out = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "period": [str(start.date()), str(end.date())],
           "years": round((end - start).days / 365.25, 2), "symbols": list(data), "params": PARAMS, "tp_r": args.tp_r, "account": ACCOUNT, "paths": args.paths,
           "spreads": {k: SPECS[k].spread for k in data}, "spread_mult": args.spread_mult,
           "dates": months,
           "signals": {"count": len(trades), "win": round(float((R > 0).mean()), 3), "avg_r": round(float(R.mean()), 3), "sum_r": round(float(R.sum()), 1),
                       "profit_factor": round(float(R[R > 0].sum() / -R[R <= 0].sum()), 2) if (R <= 0).any() else None,
                       "reasons": pd.Series([t["reason"] for t in trades]).value_counts().to_dict(), "by_year": by_year},
           "scenarios": scenarios, "risk_sweep": sweep}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False)
    s = out["signals"]
    print(f"periodo {out['period']} ({out['years']} anos), {s['count']} senales, acierto {s['win']:.0%}, R medio {s['avg_r']}, PF {s['profit_factor']}, por ano {by_year}")
    for key, sc in scenarios.items():
        h, st = sc["historical"], sc["stats"]
        print(f"[{key}] historico: final {h['final']} ({h['return']:+.0%}), DD {h['max_dd']:.0%}, tomadas {h['taken']}, sin lote {h['skipped_lot']}, llenas {h['skipped_full']}, freno {h['brake_at']}")
        print(f"[{key}] montecarlo: mediana {st['final_p50']}, p10 {st['final_p10']}, p90 {st['final_p90']}, P(perder) {st['p_loss']:.0%}, P(mitad) {st['p_half']:.0%}, "
              f"P(freno) {st['p_brake']:.0%} (mes mediano {st['brake_month_p50']}), P(x2) {st['p_double']:.0%}, P(x5) {st['p_x5']:.0%}, DD mediana {st['max_dd_p50']:.0%}")
    for r in sweep:
        print(f"riesgo {r['risk_pct']:.0%} {'con' if r['brake'] else 'sin'} freno: hist {r['hist_final']} (DD {r['hist_max_dd']:.0%}), mediana {r['final_p50']}, p10 {r['final_p10']}, "
              f"P(perder) {r['p_loss']:.0%}, P(mitad) {r['p_half']:.0%}, P(freno) {r['p_brake']:.0%}, DD mediana {r['max_dd_p50']:.0%}")
    print("->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
