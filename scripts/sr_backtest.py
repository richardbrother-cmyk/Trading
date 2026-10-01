"""Backtest de la estrategia de soportes y resistencias (autotrader/sr.py) sobre barras H4 de 3 anos, con costes.

Metodologia (para no engañarse):
- Cada variante se corre sobre los seis simbolos que opera la cuenta agresiva con una cuenta grande (10.000 USD, riesgo 1 %),
  para medir la distribucion de resultados en R sin que el lote minimo la distorsione. El efecto del lote minimo y de los 500 USD
  se mide aparte (`account`), con el mismo simulador que la cuenta agresiva (scripts/aggr_simulation.py).
- Los niveles solo usan pivotes ya confirmados (sin mirar al futuro; lo comprueba tests/test_sr.py). Entrada en la apertura de la
  barra siguiente, con spread, comision y swap.
- El periodo se parte en AJUSTE (hasta --split) y CIEGO (despues). Se mira el ajuste y se confirma en el ciego.
- Tres bloques en la salida:
    grid          36 variantes: modo (rebote / retest / ambos) x objetivo (2R, 3R, siguiente nivel) x tendencia x cortos.
    neighborhood  vecindad de la familia que sobrevive (rebote, largos, objetivo en el siguiente nivel): toques, R minimo, memoria.
                  Sirve para ver si hay una meseta estable o un pico casual.
    selected      configuracion que opera el bot, con desglose por simbolo y por ano y la simulacion de la cuenta de 500 USD.
- Aviso: elegir una variante entre ~70 mirando resultados infla lo que se ve. La eleccion se hizo en la zona central de la meseta,
  no en el maximo, y aun asi la evidencia es modesta (pocas operaciones, concentrada en GBPUSD). Se evalua en demo.

Uso: python scripts/sr_backtest.py [--out docs/sr_backtest.json] [--split 2025-09-15] [--paths 800]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aggr_simulation as ag  # noqa: E402
from autotrader.stats import benjamini_hochberg, summarize_variant  # noqa: E402
from autotrader.swing import SwingParams, backtest_symbol  # noqa: E402

SYMBOLS = ["US500", "NAS100", "EURUSD", "GBPUSD", "XAUUSD", "XTIUSD"]
DATA = "data/intraday"
BASE = dict(strategy="sr", timeframe="H4", max_hold_days=7.0, breakeven_r=0.0)
# Configuracion que opera el bot (.github/workflows/ctrader-sr.yml)
SELECTED = dict(BASE, sr_mode="bounce", allow_short=False, sr_target="level", sr_min_rr=2.0, sr_min_touches=2, sr_lookback=300, sr_trend=True)


def load(sym: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA, f"{sym}_M15.csv"), parse_dates=["time"]).set_index("time")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def stats(rows: list[dict]) -> dict:
    if not rows:
        return {"trades": 0}
    r = np.array([x["r"] for x in rows])
    wins, losses = r[r > 0], r[r <= 0]
    return {"trades": int(len(r)), "win_rate": round(float((r > 0).mean()), 3), "avg_r": round(float(r.mean()), 3),
            "sum_r": round(float(r.sum()), 1), "profit_factor": round(float(wins.sum() / -losses.sum()), 2) if losses.sum() < 0 else None,
            "median_r": round(float(np.median(r)), 2), "worst_r": round(float(r.min()), 2)}


def trade_rows(params: dict, symbols: list[str]) -> list[dict]:
    p = SwingParams(**params, risk_pct=0.01, max_risk_pct=0.5)
    rows = []
    for sym in symbols:
        for t in backtest_symbol(load(sym), sym, p, 10_000.0):
            rows.append({"symbol": sym, "side": t.side, "t": t.entry_time, "x": t.exit_time, "entry": t.entry, "dist": abs(t.entry - t.stop),
                         "r": t.r, "reason": t.reason})
    rows.sort(key=lambda x: x["t"])
    return rows


def run_variant(args: tuple) -> dict:
    name, params, symbols, split = args
    rows = trade_rows(params, symbols)
    cut = pd.Timestamp(split, tz="UTC")
    return {"name": name, "params": params, "all": stats(rows), "fit": stats([x for x in rows if x["t"] < cut]),
            "blind": stats([x for x in rows if x["t"] >= cut]), "test": summarize_variant([x["r"] for x in rows]),
            "by_symbol": {s: stats([x for x in rows if x["symbol"] == s]) for s in symbols},
            "by_side": {"long": stats([x for x in rows if x["side"] == 1]), "short": stats([x for x in rows if x["side"] == -1])},
            "exits": pd.Series([x["reason"] for x in rows]).value_counts().to_dict() if rows else {}}


def grid() -> list[tuple[str, dict]]:
    out = []
    for mode, target, trend, short in itertools.product(["bounce", "retest", "both"], ["rr2", "rr3", "level"], [True, False], [False, True]):
        params = dict(BASE, sr_mode=mode, sr_trend=trend, allow_short=short)
        params.update(sr_target="level", sr_min_rr=1.5) if target == "level" else params.update(sr_target="rr", sr_rr=float(target[-1]))
        out.append((f"{mode}|{target}|{'tendencia' if trend else 'libre'}|{'largo+corto' if short else 'solo largos'}", params))
    return out


def neighborhood() -> list[tuple[str, dict]]:
    out = []
    for touches, min_rr, lookback, trend in itertools.product([2, 3], [1.5, 2.0, 3.0], [160, 240, 360], [False, True]):
        params = dict(BASE, sr_mode="bounce", allow_short=False, sr_target="level", sr_min_rr=min_rr, sr_min_touches=touches,
                      sr_lookback=lookback, sr_trend=trend)
        out.append((f"toques {touches} | R min {min_rr:g} | memoria {lookback} | {'tendencia' if trend else 'libre'}", params))
    return out


def selected_detail(split: str, paths: int, seed: int = 7) -> dict:
    rows = trade_rows(SELECTED, SYMBOLS)
    cut = pd.Timestamp(split, tz="UTC")
    df = pd.DataFrame(rows)
    by_year = {str(y): stats([x for x in rows if x["t"].year == y]) for y in sorted({x["t"].year for x in rows})}
    ag.ensure_specs(SYMBOLS, DATA)
    sim_rows = [{"symbol": x["symbol"], "entry_time": x["t"], "exit_time": x["x"], "entry": x["entry"], "dist": x["dist"], "r": x["r"],
                 "reason": x["reason"]} for x in rows]
    R = np.array([x["r"] for x in rows])
    start = min(x["t"] for x in rows).normalize()
    end = max(df["x"]).normalize()
    grid_dates = pd.date_range(start, end, freq="W-FRI", tz="UTC")
    account = []
    for risk, maxpos in ((0.03, 2), (0.06, 2)):
        acc = dict(ag.ACCOUNT, risk_pct=risk, max_risk_pct=max(0.03, 1.5 * risk), max_positions=maxpos)
        h = ag.simulate(sim_rows, R, grid_dates, brake=True, account=acc)
        rng = np.random.default_rng(seed)
        finals, dds, brakes = [], [], 0
        for _ in range(paths):
            res = ag.simulate(sim_rows, rng.choice(R, size=len(R), replace=True), grid_dates, brake=True, account=acc)
            finals.append(res["final"]); dds.append(res["max_dd"]); brakes += res["brake_at"] is not None
        f = np.array(finals)
        account.append({"risk_pct": risk, "max_positions": maxpos, "historical": {"final": round(h["final"], 0), "max_dd": round(h["max_dd"], 3),
                        "taken": h["taken"], "skipped_min_lot": h["skipped_lot"], "skipped_full": h["skipped_full"]},
                        "montecarlo": {"paths": paths, "final_p10": round(float(np.percentile(f, 10)), 0), "final_p50": round(float(np.median(f)), 0),
                                       "final_p90": round(float(np.percentile(f, 90)), 0), "p_loss": round(float((f < 500).mean()), 3),
                                       "p_half": round(float((f < 250).mean()), 3), "p_brake": round(brakes / paths, 3),
                                       "max_dd_p50": round(float(np.median(dds)), 3),
                                       "caveat": "el remuestreo supone que la ventaja historica (R medio) es real; si no lo es, el riesgo de perder es mayor"}})
    return {"params": SELECTED, "all": stats(rows), "fit": stats([x for x in rows if x["t"] < cut]), "blind": stats([x for x in rows if x["t"] >= cut]),
            "by_symbol": {s: stats([x for x in rows if x["symbol"] == s]) for s in SYMBOLS}, "by_year": by_year,
            "exits": pd.Series([x["reason"] for x in rows]).value_counts().to_dict(),
            "min_lot_risk_usd": {s: {"median": round(float(np.median([ag.SPECS[s].min_units * x["dist"] for x in rows if x["symbol"] == s])), 1),
                                     "max": round(float(np.max([ag.SPECS[s].min_units * x["dist"] for x in rows if x["symbol"] == s])), 1)}
                                 for s in SYMBOLS if any(x["symbol"] == s for x in rows)},
            "account": account}


def print_table(title: str, results: list[dict]) -> None:
    print(f"\n{title}\n{'variante':52} | {'AJUSTE n':>8} {'avgR':>6} {'PF':>5} | {'CIEGO n':>7} {'avgR':>6} {'PF':>5} {'sumR':>6}")
    for r in results:
        f, b = r["fit"], r["blind"]
        print(f"{r['name']:52} | {f.get('trades', 0):>8} {f.get('avg_r', 0):>6} {str(f.get('profit_factor')):>5} | "
              f"{b.get('trades', 0):>7} {b.get('avg_r', 0):>6} {str(b.get('profit_factor')):>5} {b.get('sum_r', 0):>6}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/sr_backtest.json")
    ap.add_argument("--split", default="2025-09-15")
    ap.add_argument("--paths", type=int, default=800)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    args = ap.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        g = list(ex.map(run_variant, [(n, p, SYMBOLS, args.split) for n, p in grid()]))
        nb = list(ex.map(run_variant, [(n, p, SYMBOLS, args.split) for n, p in neighborhood()]))
    g.sort(key=lambda r: -(r["fit"].get("avg_r") or -9))
    nb.sort(key=lambda r: r["name"])
    print_table("REJILLA (ordenada por R medio en el ajuste)", g)
    print_table("VECINDAD de rebote / largos / objetivo en el siguiente nivel", nb)
    both_pos = sum(1 for r in nb if (r["fit"].get("avg_r") or 0) > 0 and (r["blind"].get("avg_r") or 0) > 0)
    print(f"\nvecindad: positivas en ajuste Y ciego: {both_pos} de {len(nb)}")
    sel = selected_detail(args.split, args.paths)
    # Significancia: la elegida se compara con TODAS las variantes distintas probadas (la rejilla y la vecindad se solapan en algunas)
    unique: dict[str, dict] = {}
    for r in g + nb:
        unique.setdefault(json.dumps(r["params"], sort_keys=True), r)
    rows_sel = trade_rows(SELECTED, SYMBOLS)
    cut = pd.Timestamp(args.split, tz="UTC")
    sel_key = json.dumps(SELECTED, sort_keys=True)
    unique.setdefault(sel_key, {"test": summarize_variant([x["r"] for x in rows_sel])})  # la elegida (memoria 300) no estaba en la vecindad
    names = list(unique)
    qs = benjamini_hochberg([unique[k]["test"].get("p_value", 1.0) for k in names])
    q_by_key = dict(zip(names, qs))
    sel["significance"] = {"all": summarize_variant([x["r"] for x in rows_sel]), "blind": summarize_variant([x["r"] for x in rows_sel if x["t"] >= cut]),
                           "variants_tested": len(names), "q_value_all_variants": round(q_by_key[sel_key], 3),
                           "variants_with_q_below_0_10": int(sum(1 for q in qs if q < 0.10)),
                           "note": "p unilateral (media de R > 0) por bootstrap; q = Benjamini-Hochberg sobre todas las variantes distintas probadas"}
    print("significancia:", json.dumps(sel["significance"], ensure_ascii=False))
    print("\nELEGIDA:", json.dumps({k: sel[k] for k in ("all", "fit", "blind", "by_year", "exits")}, ensure_ascii=False))
    print("por simbolo:", {s: (v.get("trades"), v.get("avg_r")) for s, v in sel["by_symbol"].items()})
    for a in sel["account"]:
        print(f"cuenta 500 USD, riesgo {a['risk_pct']:.0%}:", a["historical"], a["montecarlo"])
    out = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "split": args.split, "symbols": SYMBOLS,
           "period": ["2023-09-15", "2026-09-18"], "grid": g, "neighborhood": nb, "selected": sel,
           "reading": ("No hay ventaja robusta en la version ingenua: los cortos pierden en todas las variantes y los objetivos fijos de 2R/3R ganan algo en el "
                       "ajuste y pierden en el periodo ciego. Sobrevive una familia: rebote en soporte, solo largos, objetivo en la siguiente resistencia con "
                       "recorrido minimo; con memoria de 240 barras o mas y 2 toques es positiva en ajuste y ciego en todas las combinaciones probadas, pero "
                       "son pocas operaciones, concentradas en GBPUSD, y la eleccion se hizo entre ~70 variantes.")}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    print("->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
