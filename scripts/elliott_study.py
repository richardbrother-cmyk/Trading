"""Estudio de la estrategia de ondas de Elliott (autotrader/elliott.py) sobre el oro en 4 h y diario.

Para cada marco temporal se recorre una rejilla de parametros (umbral del ZigZag, onda operada, tipo de entrada,
objetivo, stop, lado) y para cada variante se calcula:
- resultado dentro de muestra en todo el historico: operaciones, acierto, factor de beneficio, R medio, retorno,
  drawdown, y resultado por ano natural (consistencia);
- estadistica de la media de R por operacion: intervalo de confianza bootstrap, p-valor y q-valor de
  Benjamini-Hochberg dentro del marco temporal y en el conjunto de todas las variantes probadas;
- referencia de azar: entradas en barras aleatorias con la misma proporcion de largos, la misma distancia media
  de stop y objetivo (en ATR) y el mismo tiempo maximo, repetidas `--random` veces; sirve para separar lo que
  aporta el recuento de ondas de lo que regala la tendencia del oro.
Ademas, un walk-forward por marco temporal: se elige la variante en la ventana de entrenamiento y se mide en la
siguiente; los tramos de prueba encadenados forman el resultado fuera de muestra.

Uso: python scripts/elliott_study.py [--symbol XAUUSD] [--out docs/elliott_study.json] [--random 100]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from autotrader.elliott import ElliottParams, atr, backtest_symbol, resample  # noqa: E402
from autotrader.intraday import SPECS, load_bars  # noqa: E402
from autotrader.stats import benjamini_hochberg, summarize_variant  # noqa: E402
from autotrader.swing import _size, metrics  # noqa: E402

WARMUP_DAYS = 60
MIN_TRADES = {"H4": 10, "D1": 5}
WINDOWS = {"H4": (8, 4, 4), "D1": (12, 6, 6)}  # meses de entrenamiento, prueba y paso
ZZ = {"H4": [2.0, 3.0, 4.0], "D1": [1.5, 2.0, 3.0]}
HOLD = {"H4": 10.0, "D1": 30.0}


def grid(tf: str) -> list[ElliottParams]:
    out = []
    for zz, waves, entry, ext, stop, short in itertools.product(ZZ[tf], [(2,), (4,), (2, 4)], ["ruptura", "pivote"], [1.0, 1.618],
                                                                 ["onda", "inicio"], [True, False]):
        out.append(ElliottParams(timeframe=tf, zz_atr=zz, waves=waves, entry=entry, target_ext=ext, stop=stop, allow_short=short,
                                 max_hold_days=HOLD[tf]))
    return out


def label(p: ElliottParams) -> str:
    waves = "onda 2" if p.waves == (2,) else "onda 4" if p.waves == (4,) else "ondas 2 y 4"
    return (f"{p.timeframe} · ZigZag {p.zz_atr:g} ATR · {waves} · entrada {p.entry} · objetivo {p.target_ext:g}×onda 1 · "
            f"stop {p.stop} · {'largos y cortos' if p.allow_short else 'solo largos'}")


def key(p: ElliottParams) -> str:
    return f"{p.timeframe}_zz{p.zz_atr:g}_w{''.join(map(str, p.waves))}_{p.entry}_t{p.target_ext:g}_{p.stop}_{'ls' if p.allow_short else 'l'}"


def clean(m: dict) -> dict:
    if m.get("profit_factor") == float("inf"):
        m["profit_factor"] = None
    return m


def by_year(trades: list, equity: float) -> dict:
    out = {}
    for y, group in itertools.groupby(sorted(trades, key=lambda t: t.entry_time), key=lambda t: t.entry_time.year):
        g = list(group)
        r = [t.r for t in g]
        out[str(y)] = {"trades": len(g), "sum_r": round(float(sum(r)), 1), "win_rate": round(float(np.mean([x > 0 for x in r])), 2),
                       "pnl": round(float(sum(t.pnl for t in g)), 2)}
    return out


def random_reference(df15: pd.DataFrame, sym: str, p: ElliottParams, trades: list, equity: float, draws: int, seed: int = 0) -> dict | None:
    """Entradas en barras al azar con la misma estructura de salida que la variante (stop y objetivo medianos en ATR,
    misma proporcion de largos, mismo tiempo maximo y mismo numero de operaciones)."""
    if len(trades) < 3:
        return None
    spec = SPECS[sym]
    d = resample(df15, p.timeframe)
    a = atr(d, p.atr_period).to_numpy()
    o, h, l, c = d["open"].to_numpy(), d["high"].to_numpy(), d["low"].to_numpy(), d["close"].to_numpy()
    pos = {ts: i for i, ts in enumerate(d.index)}
    stop_atr = float(np.median([abs(t.entry - t.stop) / a[pos[t.entry_time] - 1] for t in trades]))
    tp_atr = float(np.median([abs(t.target - t.entry) / a[pos[t.entry_time] - 1] for t in trades]))
    long_share = float(np.mean([t.side == 1 for t in trades]))
    n_tr, hold = len(trades), p.max_hold_bars()
    sp = p.swing_like()
    rng = np.random.default_rng(seed)
    pfs, rets, rs = [], [], []
    valid = np.arange(p.atr_period + 2, len(d) - 2)
    for _ in range(draws):
        bars = np.sort(rng.choice(valid, size=n_tr, replace=False))
        eq, pnl_sum, r_list, gw, gl = equity, 0.0, [], 0.0, 0.0
        last_exit = -1
        for i in bars:
            if i <= last_exit:
                continue
            s = 1 if rng.random() < long_share else -1
            entry = o[i + 1] + s * spec.spread / 2
            stop = entry - s * stop_atr * a[i]
            tp = entry + s * tp_atr * a[i]
            units = _size(eq, entry, stop, spec, sp)
            if units <= 0:
                continue
            j_end = min(len(d) - 1, i + 1 + hold)
            exit_px, j_exit = c[j_end], j_end
            for j in range(i + 1, j_end + 1):
                if (s == 1 and l[j] <= stop) or (s == -1 and h[j] >= stop):
                    exit_px, j_exit = stop, j; break
                if (s == 1 and h[j] >= tp) or (s == -1 and l[j] <= tp):
                    exit_px, j_exit = tp, j; break
            exit_px -= s * spec.spread / 2
            days = max((d.index[j_exit] - d.index[i + 1]).total_seconds() / 86400, 0)
            costs = (entry + exit_px) * units * p.commission_side + entry * units * p.swap_daily * days
            pnl = (exit_px - entry) * s * units - costs
            eq += pnl
            r_list.append(pnl / (abs(entry - stop) * units))
            if pnl > 0:
                gw += pnl
            else:
                gl -= pnl
            last_exit = j_exit
        if not r_list:
            continue
        pfs.append(min(gw / gl, 10.0) if gl > 0 else 10.0)
        rets.append(eq / equity - 1)
        rs.append(float(np.mean(r_list)))
    if not pfs:
        return None
    real_r = float(np.mean([t.r for t in trades]))
    return {"draws": len(pfs), "stop_atr": round(stop_atr, 2), "tp_atr": round(tp_atr, 2), "long_share": round(long_share, 2),
            "profit_factor_mean": round(float(np.mean(pfs)), 2), "profit_factor_p10_p90": [round(float(np.percentile(pfs, 10)), 2), round(float(np.percentile(pfs, 90)), 2)],
            "return_mean": round(float(np.mean(rets)), 4), "avg_r_mean": round(float(np.mean(rs)), 3),
            "share_random_avg_r_at_least_real": round(float(np.mean([x >= real_r for x in rs])), 3)}


def run_window(df15: pd.DataFrame, sym: str, p: ElliottParams, start: pd.Timestamp, end: pd.Timestamp, equity: float) -> list:
    sub = df15[(df15.index >= start - pd.Timedelta(days=WARMUP_DAYS)) & (df15.index < end)]
    if len(sub) < 500:
        return []
    return [t for t in backtest_symbol(sub, sym, p, equity) if start <= t.entry_time < end]


def score(m: dict, tf: str) -> float:
    if m.get("trades", 0) < MIN_TRADES[tf]:
        return -1.0
    pf = m["profit_factor"]
    return (10.0 if pf in (None, float("inf")) else pf) + m["return"]


def walkforward(df15: pd.DataFrame, sym: str, tf: str, configs: list[ElliottParams], equity: float) -> dict:
    train, test, step = WINDOWS[tf]
    first = (df15.index[0] + pd.Timedelta(days=WARMUP_DAYS)).normalize()
    last = df15.index[-1]
    folds, t0 = [], first
    while True:
        a, b = t0 + pd.DateOffset(months=train), t0 + pd.DateOffset(months=train + test)
        if b > last + pd.Timedelta(days=1):
            break
        folds.append((t0, a, b)); t0 = t0 + pd.DateOffset(months=step)
    results, oos = [], []
    for k, (a, b, c) in enumerate(folds, 1):
        best, best_m, best_s = None, None, -9.0
        for p in configs:
            m = clean(metrics(run_window(df15, sym, p, a, b, equity), equity, max((b - a).days, 1)))
            sc = score(m, tf)
            if sc > best_s:
                best, best_m, best_s = p, m, sc
        tr = run_window(df15, sym, best, b, c, equity)
        oos += tr
        tm = clean(metrics(tr, equity, max((c - b).days, 1)))
        results.append({"fold": k, "train_window": [str(a.date()), str(b.date())], "test_window": [str(b.date()), str(c.date())],
                        "chosen": label(best), "chosen_key": key(best), "train": best_m, "test": tm})
        print(f"  {tf} tramo {k}: {label(best)} | entrena PF {best_m.get('profit_factor')} ops {best_m.get('trades')} | "
              f"prueba PF {tm.get('profit_factor')} ops {tm.get('trades')} ret {tm.get('return')}")
    oos.sort(key=lambda t: t.entry_time)
    oos_days = (folds[-1][2] - folds[0][1]).days if folds else 1
    om = clean(metrics(oos, equity, max(oos_days, 1)))
    stats = summarize_variant([t.r for t in oos]) if oos else {"n": 0}
    return {"windows": {"train_months": train, "test_months": test, "step_months": step, "folds": len(folds)}, "folds": results,
            "oos": om, "oos_stats": stats, "chosen_stable": len({r["chosen_key"] for r in results}) <= 1}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--data", default="data/intraday")
    ap.add_argument("--out", default="docs/elliott_study.json")
    ap.add_argument("--equity", type=float, default=10_000)
    ap.add_argument("--random", type=int, default=100, help="repeticiones de la referencia de azar por variante")
    ap.add_argument("--q", type=float, default=0.10)
    args = ap.parse_args()
    df15 = load_bars(os.path.join(args.data, f"{args.symbol}_M15.csv"))
    days = (df15.index[-1] - df15.index[0]).days
    rows, wf = [], {}
    for tf in ("H4", "D1"):
        configs = grid(tf)
        print(f"{tf}: {len(configs)} variantes, {len(resample(df15, tf))} barras")
        for p in configs:
            tr = backtest_symbol(df15, args.symbol, p, args.equity)
            m = clean(metrics(tr, args.equity, days))
            R = [t.r for t in tr]
            row = {"key": key(p), "timeframe": tf, "label": label(p), "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(p).items()},
                   "metrics": m, "stats": summarize_variant(R), "by_year": by_year(tr, args.equity),
                   "long_share": round(float(np.mean([t.side == 1 for t in tr])), 2) if tr else None,
                   "exits": {r: sum(t.reason == r for t in tr) for r in ("objetivo", "stop", "tiempo maximo")} if tr else {},
                   "random": random_reference(df15, args.symbol, p, tr, args.equity, args.random)}
            rows.append(row)
        wf[tf] = walkforward(df15, args.symbol, tf, configs, args.equity)
    # q-valores por marco temporal y en el conjunto
    for tf in ("H4", "D1"):
        idx = [i for i, r in enumerate(rows) if r["timeframe"] == tf and r["stats"].get("n", 0) > 1]
        for i, qv in zip(idx, benjamini_hochberg([rows[i]["stats"]["p_value"] for i in idx])):
            rows[i]["q_family"] = round(qv, 4)
    idx = [i for i, r in enumerate(rows) if r["stats"].get("n", 0) > 1]
    for i, qv in zip(idx, benjamini_hochberg([rows[i]["stats"]["p_value"] for i in idx])):
        rows[i]["q_all"] = round(qv, 4)
        rows[i]["significant_all"] = bool(qv <= args.q)
        rows[i]["significant_family"] = bool(rows[i].get("q_family", 1.0) <= args.q)
    rows.sort(key=lambda r: (r["stats"].get("p_value", 1.0), -r["stats"].get("n", 0)))
    tested = len(idx)
    sig = [r for r in rows if r.get("significant_all")]
    beat_random = [r for r in rows if r.get("random") and r["random"]["share_random_avg_r_at_least_real"] <= 0.05 and r["stats"].get("n", 0) >= 10]
    years_pos = [r for r in rows if r["by_year"] and all(v["sum_r"] > 0 for v in r["by_year"].values()) and r["stats"].get("n", 0) >= 10]
    verdict = []
    verdict.append(f"{tested} variantes probadas; {len(sig)} superan la correccion por multiples pruebas (q <= {args.q:g})")
    verdict.append(f"{len(beat_random)} baten a las entradas al azar con la misma estructura de salida (p <= 0,05)")
    verdict.append(f"{len(years_pos)} son positivas todos los anos con al menos 10 operaciones")
    for tf in ("H4", "D1"):
        o = wf[tf]["oos"]
        verdict.append(f"walk-forward {tf}: {o.get('trades', 0)} operaciones fuera de muestra, PF {o.get('profit_factor')}, retorno {o.get('return')}"
                       + ("" if wf[tf]["chosen_stable"] else ", parametros elegidos cambian entre tramos"))
    out = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "symbol": args.symbol, "equity": args.equity,
           "risk_pct": 0.01, "period": [str(df15.index[0].date()), str(df15.index[-1].date())], "grid_size": {tf: len(grid(tf)) for tf in ("H4", "D1")},
           "q_level": args.q, "tests": tested, "random_draws": args.random,
           "method": "media de R por operacion; IC 95 % bootstrap; p unilateral bootstrap centrado; q Benjamini-Hochberg; referencia de azar con misma estructura de salida",
           "variants": rows, "walkforward": wf, "verdict": "; ".join(verdict)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    print(f"{'variante':95s} {'n':>4s} {'R medio':>8s} {'p':>7s} {'q todo':>7s} {'PF':>6s} {'azar>=real':>10s}")
    for r in rows[:25]:
        s, rd = r["stats"], r.get("random") or {}
        print(f"{r['label'][:95]:95s} {s.get('n', 0):4d} {s.get('mean_r', 0):+8.3f} {s.get('p_value', 1):7.4f} {r.get('q_all', 1):7.3f} "
              f"{str(r['metrics'].get('profit_factor')):>6s} {str(rd.get('share_random_avg_r_at_least_real', '—')):>10s}"
              + ("  *" if r.get("significant_all") else ""))
    print("->", args.out, "|", out["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
