"""Familia diaria de Elliott en indices con historia larga (Yahoo, desde 2000).

En el estudio de 3 anos (scripts/elliott_study.py) las unicas combinaciones que batian al azar en mas de un activo eran
diarias, con ZigZag de 1,5 ATR, entrada en el pivote de la onda 2 y stop bajo la correccion, en US500 y NAS100. Aqui
se prueba esa familia (y sus vecinas: ZigZag 2 ATR, onda 4, objetivo 1 o 1,618) sobre siete indices de contado desde
2000, separando el tramo ANTERIOR al 15-09-2023 (datos que el estudio corto no vio) del posterior.

Para cada indice y variante: operaciones, R medio, PF, resultado por bloques de 5 anos, referencia de azar con la misma
estructura de salida (en el tramo anterior a 2023) y p-valor de la media de R. Por variante: cuantos indices son
positivos y baten al azar en el tramo anterior a 2023, y el R agrupado de todos los indices.

Uso: python scripts/elliott_daily_indices.py [--out docs/elliott_daily_indices.json] [--random 100]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from autotrader.elliott import ElliottParams, atr, backtest_symbol  # noqa: E402
from autotrader.intraday import SPECS, SymbolSpec  # noqa: E402
from autotrader.stats import benjamini_hochberg, summarize_variant  # noqa: E402
from autotrader.swing import _size, metrics  # noqa: E402

CACHE = os.path.join(ROOT, "data/research/indices_daily_2000.csv")
SPLIT = pd.Timestamp("2023-09-15")
# indice de Yahoo -> (nombre corto, spread tipico del CFD en puntos)
INDICES = {"^GSPC": ("SP500", 0.4), "^NDX": ("NDX100", 1.0), "^DJI": ("DOW30", 2.0), "^GDAXI": ("DAX40", 1.0),
           "^N225": ("NIKKEI", 8.0), "^FTSE": ("FTSE100", 1.0), "^STOXX50E": ("STOXX50", 1.0)}
FAMILY = {"D1_zz1.5_w2_pivote_t1.618_onda_ls", "D1_zz1.5_w24_pivote_t1_onda_l", "D1_zz1.5_w24_pivote_t1.618_onda_l"}
HOLD_DAYS = 30.0


def fetch_daily(sym: str, start: str = "2000-01-01") -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
    params = {"period1": int(pd.Timestamp(start).timestamp()), "period2": int(time.time()), "interval": "1d", "events": "div,splits"}
    r = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0 (autotrader)"}, timeout=40)
    r.raise_for_status()
    node = r.json()["chart"]["result"][0]
    q = node["indicators"]["quote"][0]
    df = pd.DataFrame({"open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"], "volume": q["volume"]},
                      index=pd.to_datetime(node["timestamp"], unit="s", utc=True).tz_convert(None).normalize()).dropna()
    df.index.name = "date"
    return df[(df["high"] >= df["low"]) & (df["close"] > 0)]


def load() -> dict[str, pd.DataFrame]:
    if os.path.exists(CACHE):
        df = pd.read_csv(CACHE, parse_dates=["date"])
    else:
        frames = []
        for y, (name, _sp) in INDICES.items():
            d = fetch_daily(y).reset_index()
            d["symbol"] = name
            frames.append(d)
        df = pd.concat(frames)
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        df.to_csv(CACHE, index=False)
    out = {}
    for s, g in df.groupby("symbol"):
        d = g.drop(columns="symbol").set_index("date").sort_index()
        d.index = d.index.tz_localize("UTC")
        out[s] = d
    return out


def register_specs() -> None:
    for _y, (name, spread) in INDICES.items():
        SPECS.setdefault(name, SymbolSpec(name, "00:00", "24:00", spread, 0.01, 0.01, 2))


def grid() -> list[ElliottParams]:
    out = []
    for zz, waves, ext, short in itertools.product([1.5, 2.0], [(2,), (2, 4)], [1.0, 1.618], [True, False]):
        out.append(ElliottParams(timeframe="D1", zz_atr=zz, waves=waves, entry="pivote", target_ext=ext, stop="onda", allow_short=short,
                                 max_hold_days=HOLD_DAYS))
    return out


def key(p: ElliottParams) -> str:
    return f"D1_zz{p.zz_atr:g}_w{''.join(map(str, p.waves))}_{p.entry}_t{p.target_ext:g}_{p.stop}_{'ls' if p.allow_short else 'l'}"


def label(p: ElliottParams) -> str:
    waves = "onda 2" if p.waves == (2,) else "ondas 2 y 4"
    return f"ZigZag {p.zz_atr:g} ATR · {waves} · entrada pivote · objetivo {p.target_ext:g}×onda 1 · stop onda · {'largos y cortos' if p.allow_short else 'solo largos'}"


def clean(m: dict) -> dict:
    if m.get("profit_factor") == float("inf"):
        m["profit_factor"] = None
    return m


def blocks(trades: list) -> dict:
    out = {}
    for t in trades:
        b = f"{(t.entry_time.year // 5) * 5}-{(t.entry_time.year // 5) * 5 + 4}"
        cur = out.setdefault(b, {"trades": 0, "sum_r": 0.0})
        cur["trades"] += 1
        cur["sum_r"] += t.r
    return {k: {"trades": v["trades"], "sum_r": round(v["sum_r"], 1)} for k, v in sorted(out.items())}


def random_reference(d: pd.DataFrame, sym: str, p: ElliottParams, trades: list, equity: float, draws: int, seed: int = 0) -> dict | None:
    """Entradas al azar con la misma estructura de salida (stop y objetivo medianos en ATR, misma proporcion de largos,
    mismo tiempo maximo y numero de operaciones) sobre las barras `d`."""
    if len(trades) < 5:
        return None
    spec = SPECS[sym]
    a = atr(d, p.atr_period).to_numpy()
    o, h, l, c = d["open"].to_numpy(), d["high"].to_numpy(), d["low"].to_numpy(), d["close"].to_numpy()
    pos = {ts: i for i, ts in enumerate(d.index)}
    stop_atr = float(np.median([abs(t.entry - t.stop) / a[pos[t.entry_time] - 1] for t in trades]))
    tp_atr = float(np.median([abs(t.target - t.entry) / a[pos[t.entry_time] - 1] for t in trades]))
    long_share = float(np.mean([t.side == 1 for t in trades]))
    n_tr, hold = len(trades), p.max_hold_bars()
    sp = p.swing_like()
    rng = np.random.default_rng(seed)
    rs = []
    valid = np.arange(p.atr_period + 2, len(d) - 2)
    for _ in range(draws):
        bars_i = np.sort(rng.choice(valid, size=n_tr, replace=False))
        eq, r_list, last_exit = equity, [], -1
        for i in bars_i:
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
            last_exit = j_exit
        if r_list:
            rs.append(float(np.mean(r_list)))
    if not rs:
        return None
    real_r = float(np.mean([t.r for t in trades]))
    return {"draws": len(rs), "stop_atr": round(stop_atr, 2), "tp_atr": round(tp_atr, 2), "long_share": round(long_share, 2),
            "avg_r_mean": round(float(np.mean(rs)), 3), "share_random_avg_r_at_least_real": round(float(np.mean([x >= real_r for x in rs])), 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/elliott_daily_indices.json")
    ap.add_argument("--equity", type=float, default=10_000)
    ap.add_argument("--random", type=int, default=100)
    args = ap.parse_args()
    register_specs()
    data = load()
    rows = []
    for p in grid():
        k = key(p)
        for sym, d in data.items():
            tr = backtest_symbol(d, sym, p, args.equity, bars=d)
            pre = [t for t in tr if t.entry_time < SPLIT.tz_localize("UTC")]
            post = [t for t in tr if t.entry_time >= SPLIT.tz_localize("UTC")]
            d_pre = d[d.index < SPLIT.tz_localize("UTC")]
            days = (d.index[-1] - d.index[0]).days
            rows.append({"key": k, "symbol": sym, "label": label(p), "family": k in FAMILY,
                         "params": {kk: (list(v) if isinstance(v, tuple) else v) for kk, v in asdict(p).items()},
                         "period": [str(d.index[0].date()), str(d.index[-1].date())],
                         "full": {**clean(metrics(tr, args.equity, days)), "stats": summarize_variant([t.r for t in tr])},
                         "pre2023": {**clean(metrics(pre, args.equity, max((SPLIT.tz_localize('UTC') - d.index[0]).days, 1))), "stats": summarize_variant([t.r for t in pre]),
                                     "random": random_reference(d_pre, sym, p, pre, args.equity, args.random)},
                         "post2023": {**clean(metrics(post, args.equity, max((d.index[-1] - SPLIT.tz_localize('UTC')).days, 1))), "stats": summarize_variant([t.r for t in post])},
                         "blocks": blocks(tr)})
            r = rows[-1]
            print(f"{sym:8s} {k:42s} n {r['full']['trades']:4d} R {r['full']['stats'].get('mean_r', 0):+.2f} | pre n {len(pre):4d} R {r['pre2023']['stats'].get('mean_r', 0):+.2f} "
                  f"p azar {(r['pre2023']['random'] or {}).get('share_random_avg_r_at_least_real', '—')} | post n {len(post):3d} R {r['post2023']['stats'].get('mean_r', 0):+.2f}")
    # resumen por variante: consistencia entre indices en el tramo anterior a 2023 y R agrupado
    by_var = {}
    for p in grid():
        k = key(p)
        rs = [r for r in rows if r["key"] == k]
        pooled_pre = []
        for r in rs:
            pooled_pre += []  # las R individuales no se guardan en las filas; se recalculan abajo
        by_var[k] = {"key": k, "label": label(p), "family": k in FAMILY,
                     "indices_positive_pre": [r["symbol"] for r in rs if r["pre2023"]["stats"].get("n", 0) >= 10 and r["pre2023"]["stats"]["mean_r"] > 0],
                     "indices_beat_random_pre": [r["symbol"] for r in rs if r["pre2023"]["stats"].get("n", 0) >= 10 and r["pre2023"].get("random")
                                                and r["pre2023"]["random"]["share_random_avg_r_at_least_real"] <= 0.05],
                     "indices_positive_post": [r["symbol"] for r in rs if r["post2023"]["stats"].get("n", 0) >= 5 and r["post2023"]["stats"]["mean_r"] > 0],
                     "per_symbol": {r["symbol"]: {"pre_n": r["pre2023"]["stats"].get("n", 0), "pre_mean_r": r["pre2023"]["stats"].get("mean_r"),
                                                  "pre_pf": r["pre2023"].get("profit_factor"), "pre_p_random": (r["pre2023"].get("random") or {}).get("share_random_avg_r_at_least_real"),
                                                  "post_n": r["post2023"]["stats"].get("n", 0), "post_mean_r": r["post2023"]["stats"].get("mean_r"),
                                                  "blocks": r["blocks"]} for r in rs}}
    # R agrupado (todas las operaciones de todos los indices) en el tramo anterior a 2023, con p y q
    pooled = {}
    for p in grid():
        k = key(p)
        R = []
        for sym, d in data.items():
            R += [t.r for t in backtest_symbol(d, sym, p, args.equity, bars=d) if t.entry_time < SPLIT.tz_localize("UTC")]
        pooled[k] = summarize_variant(R)
    keys = list(pooled)
    q = benjamini_hochberg([pooled[k].get("p_value", 1.0) for k in keys])
    for k, qv in zip(keys, q):
        by_var[k]["pooled_pre2023"] = {**pooled[k], "q_value": round(qv, 4)}
        bl = {}
        for sym in by_var[k]["per_symbol"]:
            for b, v in by_var[k]["per_symbol"][sym]["blocks"].items():
                cur = bl.setdefault(b, {"trades": 0, "sum_r": 0.0}); cur["trades"] += v["trades"]; cur["sum_r"] += v["sum_r"]
        by_var[k]["blocks_all"] = {b: {"trades": v["trades"], "sum_r": round(v["sum_r"], 1)} for b, v in sorted(bl.items())}
        by_var[k]["blocks_positive"] = sum(1 for v in bl.values() if v["sum_r"] > 0)
        by_var[k]["blocks_total"] = len(bl)
    variants = sorted(by_var.values(), key=lambda v: (-len(v["indices_beat_random_pre"]), -len(v["indices_positive_pre"]), v["pooled_pre2023"].get("p_value", 1)))
    fam = [v for v in variants if v["family"]]
    verdict = []
    for v in fam:
        pp = v["pooled_pre2023"]
        verdict.append(f"{v['label'].split(' · entrada')[0]} ({'largos y cortos' if v['key'].endswith('_ls') else 'solo largos'}, objetivo {v['key'].split('_t')[1].split('_')[0]}): "
                       f"antes de 2023, {len(v['indices_positive_pre'])} de {len(data)} indices positivos, {len(v['indices_beat_random_pre'])} baten al azar, "
                       f"R agrupado {pp.get('mean_r')} (p {pp.get('p_value')}, q {pp.get('q_value')}), {v['blocks_positive']}/{v['blocks_total']} bloques de 5 anos positivos")
    out = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "indices": {name: y for y, (name, _s) in INDICES.items()},
           "split": str(SPLIT.date()), "equity": args.equity, "risk_pct": 0.01, "random_draws": args.random, "grid_size": len(grid()),
           "periods": {s: [str(d.index[0].date()), str(d.index[-1].date())] for s, d in data.items()},
           "variants": variants, "rows": rows, "verdict": "; ".join(verdict)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    print("\n== resumen por variante (tramo anterior al 15-09-2023) ==")
    for v in variants:
        pp = v["pooled_pre2023"]
        print(f"{'*' if v['family'] else ' '} {v['label'][:80]:80s} pos {len(v['indices_positive_pre'])}/{len(data)} azar {len(v['indices_beat_random_pre'])} | "
              f"agrupado n {pp.get('n')} R {pp.get('mean_r')} p {pp.get('p_value')} q {pp.get('q_value')} | bloques + {v['blocks_positive']}/{v['blocks_total']} | post pos {len(v['indices_positive_post'])}")
    print("->", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
