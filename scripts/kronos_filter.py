"""Investigacion: Kronos como filtro de confirmacion de las senales swing (bandas H4, solo largos).

Para cada senal de compra de la configuracion desplegada se pide a Kronos-small la prediccion de las
siguientes `--horizon` barras de 4 h a partir de las ultimas 400 barras, y se guarda en
data/research/kronos_predictions.json (cache: las ejecuciones siguientes no vuelven a predecir).
Con eso se mide (1) si la prediccion tiene informacion (acierto direccional e IC frente al retorno
real a ese horizonte) y (2) que pasa con el backtest si solo se entra cuando Kronos preve subida,
comparado con el inverso y con un filtro aleatorio de la misma tasa de paso.

Uso: python scripts/kronos_filter.py [--horizon 6 --samples 10 --kronos-dir <repo clonado>]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autotrader.swing import SwingParams, backtest_symbol, indicators, metrics, resample, signal  # noqa: E402

DEPLOYED = SwingParams("bands", "H4", stop_atr=2.0, allow_short=False, bb_std=2.0, bands_rsi=30.0, max_hold_days=3.0)
LOOKBACK = 400
CACHE = "data/research/kronos_predictions.json"


def load(data_dir: str) -> dict[str, pd.DataFrame]:
    data = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*_M15.csv"))):
        df = pd.read_csv(path, parse_dates=["time"]).set_index("time")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        data[os.path.basename(path).split("_")[0]] = df
    return data


def signal_bars(h4: pd.DataFrame, p: SwingParams) -> list[int]:
    d = indicators(h4, p)
    return [i for i in range(max(LOOKBACK, 210), len(d) - 1) if signal(d, i, p) == 1]


def build_predictor(kronos_dir: str, cache_dir: str):
    sys.path.insert(0, kronos_dir)
    from model import Kronos, KronosPredictor, KronosTokenizer  # type: ignore
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base", cache_dir=cache_dir)
    mdl = Kronos.from_pretrained("NeoQuasar/Kronos-small", cache_dir=cache_dir)
    return KronosPredictor(mdl, tok, device="cpu", max_context=512)


def predict_at(predictor, h4: pd.DataFrame, i: int, horizon: int, samples: int) -> list[float]:
    x = h4.iloc[i - LOOKBACK + 1:i + 1][["open", "high", "low", "close"]].reset_index(drop=True)
    xt = pd.Series(h4.index[i - LOOKBACK + 1:i + 1].tz_convert(None))
    last = h4.index[i]
    yt = pd.Series(pd.date_range(last + pd.Timedelta(hours=4), periods=horizon, freq="4h").tz_convert(None))
    out = predictor.predict(df=x, x_timestamp=xt, y_timestamp=yt, pred_len=horizon, T=1.0, top_p=0.9, sample_count=samples, verbose=False)
    return [float(v) for v in out["close"]]


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 5:
        return None
    ra, rb = pd.Series(a).rank(), pd.Series(b).rank()
    return round(float(np.corrcoef(ra, rb)[0, 1]), 3)


def run_variant(data, filt, equity: float, days: float) -> dict:
    trades = []
    for sym, df in data.items():
        trades += backtest_symbol(df, sym, DEPLOYED, equity, entry_filter=(lambda t, s=sym: filt(s, t)) if filt else None)
    trades.sort(key=lambda t: t.entry_time)
    m = metrics(trades, equity, days)
    if m.get("profit_factor") == float("inf"):
        m["profit_factor"] = None
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/intraday")
    ap.add_argument("--out", default="docs/kronos_report.json")
    ap.add_argument("--horizon", type=int, default=6, help="barras de 4 h a predecir")
    ap.add_argument("--samples", type=int, default=10, help="muestras por prediccion (se promedian)")
    ap.add_argument("--equity", type=float, default=10_000)
    ap.add_argument("--kronos-dir", default=os.environ.get("KRONOS_DIR", ""))
    ap.add_argument("--hf-cache", default=os.environ.get("HF_HOME", ""))
    args = ap.parse_args()
    data = load(args.data)
    h4s = {sym: resample(df, "H4") for sym, df in data.items()}
    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as fh:
            cache = json.load(fh)
    key = f"h{args.horizon}"
    cache.setdefault(key, {})
    todo = [(sym, i) for sym, h4 in h4s.items() for i in signal_bars(h4, DEPLOYED) if f"{sym}|{h4.index[i].isoformat()}" not in cache[key]]
    if todo:
        if not args.kronos_dir:
            print("Faltan predicciones y no se indico --kronos-dir", file=sys.stderr)
            return 1
        predictor = build_predictor(args.kronos_dir, args.hf_cache or None)
        for n, (sym, i) in enumerate(todo, 1):
            h4 = h4s[sym]
            closes = predict_at(predictor, h4, i, args.horizon, args.samples)
            cache[key][f"{sym}|{h4.index[i].isoformat()}"] = {"close": float(h4["close"].iloc[i]), "pred": closes}
            if n % 10 == 0 or n == len(todo):
                print(f"predicciones {n}/{len(todo)}", flush=True)
                os.makedirs(os.path.dirname(CACHE), exist_ok=True)
                with open(CACHE, "w", encoding="utf-8") as fh:
                    json.dump(cache, fh)
    # 1) informacion de la prediccion: direccion y correlacion con el retorno real al horizonte
    rows = []
    for sym, h4 in h4s.items():
        d = indicators(h4, DEPLOYED)
        for i in signal_bars(h4, DEPLOYED):
            rec = cache[key].get(f"{sym}|{h4.index[i].isoformat()}")
            if not rec or i + args.horizon >= len(h4):
                continue
            c0 = rec["close"]
            pred_ret = rec["pred"][-1] / c0 - 1
            real_ret = float(h4["close"].iloc[i + args.horizon]) / c0 - 1
            atr = float(d["atr"].iloc[i])
            rows.append({"symbol": sym, "bar": h4.index[i].isoformat(), "pred_ret": pred_ret, "real_ret": real_ret,
                         "pred_atr": (rec["pred"][-1] - c0) / atr if atr else 0.0, "real_atr": (float(h4["close"].iloc[i + args.horizon]) - c0) / atr if atr else 0.0})
    n = len(rows)
    up = [r for r in rows if r["pred_ret"] > 0]
    hit = sum(1 for r in rows if (r["pred_ret"] > 0) == (r["real_ret"] > 0)) / n if n else None
    base_up = sum(1 for r in rows if r["real_ret"] > 0) / n if n else None
    info = {"signals": n, "pred_up_rate": round(len(up) / n, 3) if n else None, "real_up_rate": round(base_up, 3) if n else None,
            "directional_accuracy": round(hit, 3) if n else None, "ic_spearman": spearman([r["pred_ret"] for r in rows], [r["real_ret"] for r in rows]),
            "mean_real_ret_when_pred_up": round(float(np.mean([r["real_ret"] for r in up])), 5) if up else None,
            "mean_real_ret_when_pred_down": round(float(np.mean([r["real_ret"] for r in rows if r["pred_ret"] <= 0])), 5) if n - len(up) else None,
            "by_symbol": {}}
    for sym in h4s:
        sub = [r for r in rows if r["symbol"] == sym]
        if sub:
            info["by_symbol"][sym] = {"signals": len(sub), "directional_accuracy": round(sum(1 for r in sub if (r["pred_ret"] > 0) == (r["real_ret"] > 0)) / len(sub), 3),
                                     "ic_spearman": spearman([r["pred_ret"] for r in sub], [r["real_ret"] for r in sub])}
    # 2) backtests: todas las variantes se restringen a las barras con prediccion (las primeras 400 barras no la tienen)
    pred_by = {(r["symbol"], r["bar"]): r for r in rows}
    any_df = next(iter(data.values()))
    days = (any_df.index[-1] - any_df.index[0]).days
    mags = sorted(r["pred_atr"] for r in rows)
    q = lambda frac: mags[int(len(mags) * frac)] if mags else 0.0  # noqa: E731

    def has_pred(sym, t):
        return (sym, t.isoformat()) in pred_by

    def above(th):
        return lambda sym, t: (sym, t.isoformat()) in pred_by and pred_by[(sym, t.isoformat())]["pred_atr"] > th

    def below(th):
        return lambda sym, t: (sym, t.isoformat()) in pred_by and pred_by[(sym, t.isoformat())]["pred_atr"] <= th

    # 2b) informacion por magnitud: retorno real medio por tercil de la prediccion (en ATR)
    terc = [q(1 / 3), q(2 / 3)]
    info["magnitude_terciles_atr"] = [round(terc[0], 3), round(terc[1], 3)]
    info["real_atr_by_pred_tercile"] = [
        round(float(np.mean([r["real_atr"] for r in rows if r["pred_atr"] <= terc[0]])), 3),
        round(float(np.mean([r["real_atr"] for r in rows if terc[0] < r["pred_atr"] <= terc[1]])), 3),
        round(float(np.mean([r["real_atr"] for r in rows if r["pred_atr"] > terc[1]])), 3)] if rows else None

    variants = {"todas_las_senales_con_prediccion": run_variant(data, has_pred, args.equity, days),
                "kronos_sube_mas_de_0.25_atr": run_variant(data, above(0.25), args.equity, days),
                "kronos_sube_mas_de_0.5_atr": run_variant(data, above(0.5), args.equity, days),
                "kronos_sube_mas_de_1_atr": run_variant(data, above(1.0), args.equity, days),
                "kronos_tercil_superior": run_variant(data, above(terc[1]), args.equity, days),
                "kronos_tercil_inferior": run_variant(data, below(terc[0]), args.equity, days)}
    for name, th in (("kronos_sube_mas_de_0.5_atr", 0.5), ("kronos_tercil_superior", terc[1])):
        pass_rate = sum(1 for r in rows if r["pred_atr"] > th) / n if n else 0.0
        rand_runs = []
        for seed in range(30):
            rng = random.Random(seed)
            keep = {k for k in pred_by if rng.random() < pass_rate}
            rand_runs.append(run_variant(data, lambda sym, t, keep=keep: (sym, t.isoformat()) in keep, args.equity, days))
        pfs = [r["profit_factor"] for r in rand_runs if r.get("profit_factor") is not None]
        rets = [r["return"] for r in rand_runs if r.get("return") is not None]
        variants[name]["random_same_pass_rate"] = {"pass_rate": round(pass_rate, 3), "runs": len(rand_runs),
                                                   "profit_factor_mean": round(float(np.mean(pfs)), 2) if pfs else None,
                                                   "profit_factor_p10_p90": [round(float(np.percentile(pfs, 10)), 2), round(float(np.percentile(pfs, 90)), 2)] if pfs else None,
                                                   "return_mean": round(float(np.mean(rets)), 4) if rets else None,
                                                   "return_p10_p90": [round(float(np.percentile(rets, 10)), 4), round(float(np.percentile(rets, 90)), 4)] if rets else None}
    b = variants["todas_las_senales_con_prediccion"]
    verdict = []
    if info["pred_up_rate"] is not None and info["pred_up_rate"] > 0.95:
        verdict.append("Kronos preve subida en casi todas las senales (el precio esta bajo la media de su ventana), asi que el signo no filtra nada")
    if info["ic_spearman"] is not None and info["ic_spearman"] >= 0.15:
        verdict.append(f"la magnitud prevista si correlaciona con el rebote real (IC {info['ic_spearman']})")
    elif info["ic_spearman"] is not None:
        verdict.append(f"la magnitud prevista apenas correlaciona con el rebote real (IC {info['ic_spearman']})")
    wins = []
    for name in ("kronos_sube_mas_de_0.5_atr", "kronos_tercil_superior"):
        v = variants[name]; rr = v.get("random_same_pass_rate", {})
        if v.get("profit_factor") and rr.get("profit_factor_p10_p90") and v["profit_factor"] > rr["profit_factor_p10_p90"][1] and v.get("return", 0) > (rr.get("return_p10_p90") or [0, 0])[1]:
            wins.append(name)
    if wins:
        verdict.append("el filtro por magnitud supera al azar con la misma tasa de paso en: " + ", ".join(wins))
    else:
        verdict.append("ningun umbral de magnitud supera de forma clara a un filtro aleatorio con la misma tasa de paso")
    # 3) robustez del filtro del tercil superior: mitades del periodo y por simbolo
    first = any_df.index[0] + pd.Timedelta(days=LOOKBACK * 4 / 24)
    mid = first + (any_df.index[-1] - first) / 2

    def run_window(filt, lo, hi):
        tr = []
        for sym, df in data.items():
            tr += [t for t in backtest_symbol(df, sym, DEPLOYED, args.equity, entry_filter=lambda t, s=sym: filt(s, t)) if lo <= t.entry_time < hi]
        m = metrics(tr, args.equity, max((hi - lo).days, 1))
        if m.get("profit_factor") == float("inf"):
            m["profit_factor"] = None
        return {k: m.get(k) for k in ("trades", "profit_factor", "return", "max_drawdown")}

    robustness = {"halves": [], "by_symbol": {}}
    for name, lo, hi in (("primera mitad", first, mid), ("segunda mitad", mid, any_df.index[-1] + pd.Timedelta(days=1))):
        robustness["halves"].append({"name": name, "from": str(lo.date()), "to": str(hi.date()), "all": run_window(has_pred, lo, hi),
                                     "top_tercile": run_window(above(terc[1]), lo, hi)})
    for sym, df in data.items():
        a = backtest_symbol(df, sym, DEPLOYED, args.equity, entry_filter=lambda t, s=sym: has_pred(s, t))
        b_ = backtest_symbol(df, sym, DEPLOYED, args.equity, entry_filter=lambda t, s=sym: above(terc[1])(s, t))
        robustness["by_symbol"][sym] = {"all_trades": len(a), "all_pnl": round(sum(t.pnl for t in a), 2), "top_trades": len(b_), "top_pnl": round(sum(t.pnl for t in b_), 2)}
    halves_ok = all(h["top_tercile"].get("return", 0) > 0 for h in robustness["halves"])
    symbols_ok = sum(1 for v in robustness["by_symbol"].values() if v["top_pnl"] > 0)
    verdict.append(f"tercil superior: {'positivo en las dos mitades' if halves_ok else 'no es positivo en ambas mitades'} y en {symbols_ok} de {len(robustness['by_symbol'])} simbolos")
    out = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "model": "NeoQuasar/Kronos-small (24.7M) + Kronos-Tokenizer-base, CPU",
           "horizon_bars": args.horizon, "samples": args.samples, "lookback_bars": LOOKBACK, "deployed": "bandas H4 20/2, RSI<30, stop 2 ATR, 3 dias, solo largos",
           "equity": args.equity, "risk_pct": DEPLOYED.risk_pct, "symbols": sorted(data), "information": info, "variants": variants, "robustness": robustness, "verdict": "; ".join(verdict) or "sin conclusion clara"}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"information": info, "variants": variants, "robustness": robustness, "verdict": out["verdict"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
