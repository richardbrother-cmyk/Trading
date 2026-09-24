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

Con varios simbolos (oro e indices) se anade la comparacion entre activos: para cada combinacion de parametros se mira
en cuantos activos bate al azar; si el borde es del metodo y no del activo, deberia repetirse.

Uso: python scripts/elliott_study.py [--symbols XAUUSD,US500,NAS100] [--out docs/elliott_study.json] [--random 100]
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


def label(p: ElliottParams, sym: str = "") -> str:
    waves = "onda 2" if p.waves == (2,) else "onda 4" if p.waves == (4,) else "ondas 2 y 4"
    return ((f"{sym} · " if sym else "") + f"{p.timeframe} · ZigZag {p.zz_atr:g} ATR · {waves} · entrada {p.entry} · objetivo {p.target_ext:g}×onda 1 · "
            f"stop {p.stop} · {'largos y cortos' if p.allow_short else 'solo largos'}")


def param_key(p: ElliottParams) -> str:
    """Clave de la combinacion de parametros, sin el simbolo (sirve para comparar activos)."""
    return f"{p.timeframe}_zz{p.zz_atr:g}_w{''.join(map(str, p.waves))}_{p.entry}_t{p.target_ext:g}_{p.stop}_{'ls' if p.allow_short else 'l'}"


def key(p: ElliottParams, sym: str = "") -> str:
    return (f"{sym}_" if sym else "") + param_key(p)


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
                        "chosen": label(best), "chosen_key": param_key(best), "train": best_m, "test": tm})
        print(f"  {sym} {tf} tramo {k}: {label(best)} | entrena PF {best_m.get('profit_factor')} ops {best_m.get('trades')} | "
              f"prueba PF {tm.get('profit_factor')} ops {tm.get('trades')} ret {tm.get('return')}")
    oos.sort(key=lambda t: t.entry_time)
    oos_days = (folds[-1][2] - folds[0][1]).days if folds else 1
    om = clean(metrics(oos, equity, max(oos_days, 1)))
    stats = summarize_variant([t.r for t in oos]) if oos else {"n": 0}
    return {"windows": {"train_months": train, "test_months": test, "step_months": step, "folds": len(folds)}, "folds": results,
            "oos": om, "oos_stats": stats, "chosen_stable": len({r["chosen_key"] for r in results}) <= 1}


def study_symbol(df15: pd.DataFrame, sym: str, equity: float, draws: int) -> tuple[list[dict], dict]:
    days = (df15.index[-1] - df15.index[0]).days
    rows, wf = [], {}
    for tf in ("H4", "D1"):
        configs = grid(tf)
        print(f"{sym} {tf}: {len(configs)} variantes, {len(resample(df15, tf))} barras")
        for p in configs:
            tr = backtest_symbol(df15, sym, p, equity)
            m = clean(metrics(tr, equity, days))
            R = [t.r for t in tr]
            rows.append({"key": key(p, sym), "param_key": param_key(p), "symbol": sym, "timeframe": tf, "label": label(p, sym),
                         "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(p).items()},
                         "metrics": m, "stats": summarize_variant(R), "by_year": by_year(tr, equity),
                         "long_share": round(float(np.mean([t.side == 1 for t in tr])), 2) if tr else None,
                         "exits": {r: sum(t.reason == r for t in tr) for r in ("objetivo", "stop", "tiempo maximo")} if tr else {},
                         "random": random_reference(df15, sym, p, tr, equity, draws)})
        wf[tf] = walkforward(df15, sym, tf, configs, equity)
    return rows, wf


def beats_random(r: dict, min_trades: int = 10) -> bool:
    return bool(r.get("random")) and r["stats"].get("n", 0) >= min_trades and r["random"]["share_random_avg_r_at_least_real"] <= 0.05


def cross_symbol(rows: list[dict], symbols: list[str]) -> list[dict]:
    """Para cada combinacion de parametros: en cuantos activos bate al azar (con >= 10 operaciones) y resumen por activo."""
    by_pk: dict[str, dict] = {}
    for r in rows:
        by_pk.setdefault(r["param_key"], {})[r["symbol"]] = r
    out = []
    for pk, per in by_pk.items():
        beats = [s for s in symbols if s in per and beats_random(per[s])]
        positive = [s for s in symbols if s in per and per[s]["stats"].get("n", 0) >= 10 and per[s]["stats"]["mean_r"] > 0]
        sample = next(iter(per.values()))
        out.append({"param_key": pk, "timeframe": sample["timeframe"], "label": sample["label"].split(" · ", 1)[1],
                    "beats_random_in": beats, "positive_in": positive, "n_beats": len(beats),
                    "per_symbol": {s: {"n": per[s]["stats"].get("n", 0), "mean_r": per[s]["stats"].get("mean_r"),
                                       "profit_factor": per[s]["metrics"].get("profit_factor"),
                                       "p_random": (per[s].get("random") or {}).get("share_random_avg_r_at_least_real")}
                                   for s in symbols if s in per}})
    out.sort(key=lambda x: (-x["n_beats"], -len(x["positive_in"]), -min((v["n"] for v in x["per_symbol"].values()), default=0)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="XAUUSD,US500,NAS100")
    ap.add_argument("--data", default="data/intraday")
    ap.add_argument("--out", default="docs/elliott_study.json")
    ap.add_argument("--equity", type=float, default=10_000)
    ap.add_argument("--random", type=int, default=100, help="repeticiones de la referencia de azar por variante")
    ap.add_argument("--q", type=float, default=0.10)
    args = ap.parse_args()
    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    rows, wf, periods = [], {}, {}
    for sym in symbols:
        df15 = load_bars(os.path.join(args.data, f"{sym}_M15.csv"))
        periods[sym] = [str(df15.index[0].date()), str(df15.index[-1].date())]
        r, w = study_symbol(df15, sym, args.equity, args.random)
        rows += r
        wf[sym] = w
    # q-valores por familia (activo y marco temporal) y en el conjunto de todo lo probado
    for sym in symbols:
        for tf in ("H4", "D1"):
            idx = [i for i, r in enumerate(rows) if r["symbol"] == sym and r["timeframe"] == tf and r["stats"].get("n", 0) > 1]
            for i, qv in zip(idx, benjamini_hochberg([rows[i]["stats"]["p_value"] for i in idx])):
                rows[i]["q_family"] = round(qv, 4)
    idx = [i for i, r in enumerate(rows) if r["stats"].get("n", 0) > 1]
    for i, qv in zip(idx, benjamini_hochberg([rows[i]["stats"]["p_value"] for i in idx])):
        rows[i]["q_all"] = round(qv, 4)
        rows[i]["significant_all"] = bool(qv <= args.q)
        rows[i]["significant_family"] = bool(rows[i].get("q_family", 1.0) <= args.q)
    rows.sort(key=lambda r: (r["stats"].get("p_value", 1.0), -r["stats"].get("n", 0)))
    cross = cross_symbol(rows, symbols)
    per_symbol = {}
    verdict = [f"{len(idx)} variantes probadas en {', '.join(symbols)}"]
    for sym in symbols:
        rs = [r for r in rows if r["symbol"] == sym]
        beat = [r for r in rs if beats_random(r)]
        per_symbol[sym] = {"tested": sum(1 for r in rs if r["stats"].get("n", 0) > 1), "significant": sum(1 for r in rs if r.get("significant_all")),
                           "with_10_trades": sum(1 for r in rs if r["stats"].get("n", 0) >= 10), "beat_random": len(beat),
                           "beat_random_by_tf": {tf: sum(1 for r in beat if r["timeframe"] == tf) for tf in ("H4", "D1")},
                           "walkforward": {tf: {"trades": wf[sym][tf]["oos"].get("trades", 0), "profit_factor": wf[sym][tf]["oos"].get("profit_factor"),
                                                "return": wf[sym][tf]["oos"].get("return"), "chosen_stable": wf[sym][tf]["chosen_stable"]} for tf in ("H4", "D1")}}
        verdict.append(f"{sym}: {len(beat)} baten al azar (H4 {per_symbol[sym]['beat_random_by_tf']['H4']}, D1 {per_symbol[sym]['beat_random_by_tf']['D1']}); "
                       f"walk-forward H4 {wf[sym]['H4']['oos'].get('trades', 0)} ops PF {wf[sym]['H4']['oos'].get('profit_factor')}")
    multi = [c for c in cross if c["n_beats"] >= 2]
    verdict.append(f"{len(multi)} combinaciones de parametros baten al azar en al menos dos activos")
    out = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "symbols": symbols, "symbol": symbols[0], "equity": args.equity,
           "risk_pct": 0.01, "period": periods[symbols[0]], "periods": periods, "grid_size": {tf: len(grid(tf)) for tf in ("H4", "D1")},
           "q_level": args.q, "tests": len(idx), "random_draws": args.random,
           "method": "media de R por operacion; IC 95 % bootstrap; p unilateral bootstrap centrado; q Benjamini-Hochberg; referencia de azar con misma estructura de salida",
           "variants": rows, "walkforward": wf, "per_symbol": per_symbol, "cross_symbol": cross[:40], "verdict": "; ".join(verdict)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    print(f"{'variante':100s} {'n':>4s} {'R medio':>8s} {'PF':>6s} {'azar>=real':>10s}")
    for r in sorted(rows, key=lambda r: ((r.get('random') or {}).get('share_random_avg_r_at_least_real', 1.0), -r['stats'].get('n', 0)))[:30]:
        s, rd = r["stats"], r.get("random") or {}
        print(f"{r['label'][:100]:100s} {s.get('n', 0):4d} {s.get('mean_r', 0):+8.3f} {str(r['metrics'].get('profit_factor')):>6s} "
              f"{str(rd.get('share_random_avg_r_at_least_real', '—')):>10s}" + ("  *" if beats_random(r) else ""))
    print("-- combinaciones que baten al azar en mas de un activo --")
    for c in multi[:15]:
        print(f"  {c['label'][:90]:90s} {', '.join(c['beats_random_in'])} | " + " ".join(f"{s}: n{v['n']} R{v['mean_r']:+.2f} p{v['p_random']}" for s, v in c["per_symbol"].items()))
    print("->", args.out, "|", out["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
