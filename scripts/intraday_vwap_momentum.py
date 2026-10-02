"""Estudio intradia en horario de Nueva York: reversion al VWAP y momentum de la primera hora.

Datos: barras M15 de cTrader (data/intraday/, septiembre 2023 a septiembre 2026), seis simbolos (US500, NAS100, XAUUSD, XTIUSD,
EURUSD, GBPUSD). Todo se define en hora de Nueva York (con su horario de verano real, no horas UTC fijas): sesion 09:30-16:00,
una sola operacion por simbolo y dia, siempre cerrada antes de las 16:00.

Familias:
- VR (reversion al VWAP): VWAP anclado a las 09:30 con el volumen de ticks. Si el cierre se aleja del VWAP `k` ATR(M15), se entra en
  contra en la apertura de la barra siguiente, stop a `s` ATR del precio de entrada, objetivo en el VWAP (o a mitad de camino).
  Entradas entre 10:00 y 14:30. Opcion de saltar los dias de tendencia (precio del mismo lado del VWAP en >=85 % de las barras).
- FH (momentum de la primera hora): sentido de la primera hora (09:30-10:30); entrada a las 10:30 a favor (mom) o en contra (fade),
  stop a `s` x U (U = 2 ATR M15, la escala de una hora), umbral de movimiento minimo `t` x U, salida a las 12:00, 14:00 o 16:00.
- GAO (primera media hora -> ultima media hora): sentido de 09:30-10:00; entrada a las 15:30 hasta el cierre.
- Controles (no entran en la correccion por multiplicidad): largo y corto fijo a las 10:30 hasta el cierre, para separar la deriva
  del mercado del efecto de la regla.

Costes: la mitad del spread en cada lado y comision proporcional por lado, igual que `autotrader.intraday`. Un stop que se salta por
hueco se ejecuta a la apertura de esa barra. Si en una barra caben stop y objetivo, se cuenta el stop (conservador).
Medida: R por operacion (ganancia neta / distancia al stop). p-valor y IC por bootstrap de DIAS (las operaciones del mismo dia en
varios simbolos estan correlacionadas), q-valor de Benjamini-Hochberg sobre las variantes de VR, FH y GAO. Ademas: R medio por ano
y seleccion en la primera mitad de la muestra evaluada en la segunda.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autotrader.intraday import SPECS, load_bars  # noqa: E402
from autotrader.stats import benjamini_hochberg  # noqa: E402

SYMBOLS = ["US500", "NAS100", "XAUUSD", "XTIUSD", "EURUSD", "GBPUSD"]
COMMISSION = 0.000025
OPEN_MIN, CLOSE_MIN = 9 * 60 + 30, 16 * 60  # 09:30-16:00 NY


# ---------------------------------------------------------------------------------------------------------------- datos
def load_days(sym: str) -> list[dict]:
    d = load_bars(str(ROOT / "data" / "intraday" / f"{sym}_M15.csv"))
    tr = pd.concat([(d.high - d.low), (d.high - d.close.shift()).abs(), (d.low - d.close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    ny = d.index.tz_convert("America/New_York")
    d = d.copy()
    d["atr"] = atr.to_numpy()
    d.index = ny
    mins = d.index.hour * 60 + d.index.minute
    d = d[(mins >= OPEN_MIN) & (mins < CLOSE_MIN)]
    days = []
    for day, g in d.groupby(d.index.date):
        if pd.Timestamp(day).weekday() >= 5 or len(g) < 24:
            continue
        m = g.index.hour * 60 + g.index.minute
        if m[0] != OPEN_MIN or m[-1] != CLOSE_MIN - 15:  # sesion completa o dia descartado
            continue
        c, h, l, v = g.close.to_numpy(), g.high.to_numpy(), g.low.to_numpy(), g.volume.to_numpy()
        v = np.where(v > 0, v, 1.0)
        tp = (h + l + c) / 3
        days.append({"day": day, "m": np.asarray(m), "o": g.open.to_numpy(), "h": h, "l": l, "c": c,
                     "atr": g.atr.to_numpy(), "vwap": np.cumsum(tp * v) / np.cumsum(v)})
    return days


# ------------------------------------------------------------------------------------------------------------ simulacion
def _fill_stop(side: int, stop: float, o: float, h: float, l: float) -> float | None:
    if side == 1 and l <= stop:
        return min(stop, o)
    if side == -1 and h >= stop:
        return max(stop, o)
    return None


def _walk(D: dict, side: int, i_entry: int, i_exit: int, stop: float, target: float | None, spec, exit_at_open: bool):
    """Entra en la apertura de la barra i_entry; recorre hasta i_exit. Devuelve (R neto, bps netos, motivo)."""
    o, h, l, c = D["o"], D["h"], D["l"], D["c"]
    entry = o[i_entry] + side * spec.spread / 2
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    exit_px, why = None, ""
    last = i_exit - 1 if exit_at_open else i_exit
    for j in range(i_entry, last + 1):
        fs = _fill_stop(side, stop, o[j], h[j], l[j])
        if fs is not None:
            exit_px, why = fs, "stop"
            break
        if target is not None and ((side == 1 and h[j] >= target) or (side == -1 and l[j] <= target)):
            exit_px, why = target, "objetivo"
            break
    if exit_px is None:
        exit_px, why = (o[i_exit] if exit_at_open else c[last]), "horario"
    exit_net = exit_px - side * spec.spread / 2
    pnl = (exit_net - entry) * side - COMMISSION * (entry + exit_net)
    return pnl / risk, pnl / entry * 1e4, why


def sim_vr(D, spec, k, s, tgt, trend_skip):
    c, vw, atr, m = D["c"], D["vwap"], D["atr"], D["m"]
    n = len(c)
    for i in range(2, n - 1):
        if m[i] < 10 * 60 or m[i] > 14 * 60 + 30:
            continue
        a = atr[i]
        dev = (c[i] - vw[i]) / a if a > 0 else 0.0
        side = 1 if dev <= -k else (-1 if dev >= k else 0)
        if side == 0:
            continue
        if trend_skip and i >= 8:
            same = np.mean(np.sign(c[: i + 1] - vw[: i + 1]) == np.sign(c[i] - vw[i]))
            if same >= 0.85:
                continue
        entry = D["o"][i + 1] + side * spec.spread / 2
        stop = entry - side * s * a
        target = vw[i] if tgt == "vwap" else entry + 0.5 * (vw[i] - entry)
        if (target - entry) * side <= 0:
            continue
        return _walk(D, side, i + 1, n - 1, stop, target, spec, False) + (D["day"],)
    return None


def sim_fh(D, spec, t, s, exit_min, mode):
    c, o, atr, m = D["c"], D["o"], D["atr"], D["m"]
    n = len(c)
    U = 2 * atr[3]
    r = c[3] - o[0]
    if mode in ("mom", "fade"):
        if U <= 0 or abs(r) < t * U or r == 0:
            return None
        side = int(np.sign(r)) * (1 if mode == "mom" else -1)
    else:  # controles
        side = 1 if mode == "long" else -1
    entry = o[4] + side * spec.spread / 2
    stop = entry - side * s * U
    if exit_min >= CLOSE_MIN:
        i_exit, at_open = n - 1, False
    else:
        i_exit, at_open = int(np.searchsorted(m, exit_min)), True
    out = _walk(D, side, 4, i_exit, stop, None, spec, at_open)
    return None if out is None else out + (D["day"],)


def sim_gao(D, spec, t, s, mode):
    c, o, atr = D["c"], D["o"], D["atr"]
    n = len(c)
    U = 2 * atr[1]
    r = c[1] - o[0]
    if U <= 0 or abs(r) < t * U or r == 0:
        return None
    side = int(np.sign(r)) * (1 if mode == "mom" else -1)
    entry = o[n - 2] + side * spec.spread / 2
    stop = entry - side * s * 2 * atr[n - 3]
    out = _walk(D, side, n - 2, n - 1, stop, None, spec, False)
    return None if out is None else out + (D["day"],)


# ----------------------------------------------------------------------------------------------------------- variantes
def build_variants():
    v = []
    for k, s, tgt, ts in itertools.product([1.0, 1.5, 2.0], [1.0, 1.5], ["vwap", "half"], [False, True]):
        v.append({"family": "VR", "name": f"VR k{k} stop{s}ATR obj {tgt}{' sin tendencia' if ts else ''}",
                  "fn": lambda D, sp, k=k, s=s, tgt=tgt, ts=ts: sim_vr(D, sp, k, s, tgt, ts)})
    for t, s, ex, mode in itertools.product([0.0, 0.5, 1.0], [1.0, 1.5], [12 * 60, 14 * 60, CLOSE_MIN], ["mom", "fade"]):
        label = {12 * 60: "12:00", 14 * 60: "14:00", CLOSE_MIN: "16:00"}[ex]
        v.append({"family": "FH", "name": f"FH {mode} umbral{t}U stop{s}U salida {label}",
                  "fn": lambda D, sp, t=t, s=s, ex=ex, mode=mode: sim_fh(D, sp, t, s, ex, mode)})
    for t, s, mode in itertools.product([0.0, 0.5], [1.0, 1.5], ["mom", "fade"]):
        v.append({"family": "GAO", "name": f"GAO {mode} umbral{t}U stop{s}",
                  "fn": lambda D, sp, t=t, s=s, mode=mode: sim_gao(D, sp, t, s, mode)})
    for mode in ("long", "short"):
        v.append({"family": "CONTROL", "name": f"CONTROL {mode} 10:30-16:00 stop1.5U",
                  "fn": lambda D, sp, mode=mode: sim_fh(D, sp, 0.0, 1.5, CLOSE_MIN, mode)})
    return v


# ------------------------------------------------------------------------------------------------------------ estadistica
def day_boot(df: pd.DataFrame, n_boot: int = 10_000, seed: int = 0):
    """df con columnas day, r. Bootstrap por dias de la media de R por operacion: IC 95 % y p unilateral (H0: media <= 0)."""
    if len(df) < 5:
        return (float("nan"), float("nan")), 1.0
    g = df.groupby("day")["r"].agg(["sum", "count"])
    s, n = g["sum"].to_numpy(), g["count"].to_numpy().astype(float)
    mean = s.sum() / n.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    boots = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    s0 = s - mean * n  # centrado bajo H0
    b0 = s0[idx].sum(axis=1) / n[idx].sum(axis=1)
    p = max(float((b0 >= mean).mean()), 1.0 / n_boot)
    return ci, p


def pf(r):
    w, l = r[r > 0].sum(), -r[r <= 0].sum()
    return round(float(w / l), 2) if l > 0 else None


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0}
    r = df["r"].to_numpy()
    ci, p = day_boot(df)
    yrs = {str(y): {"n": int(len(g)), "mean_r": round(float(g["r"].mean()), 3)} for y, g in df.groupby(pd.to_datetime(df["day"]).dt.year)}
    return {"n": int(len(df)), "mean_r": round(float(r.mean()), 3), "ci95": [round(ci[0], 3), round(ci[1], 3)], "p_value": round(p, 4),
            "win_rate": round(float((r > 0).mean()), 3), "profit_factor": pf(r), "mean_bps": round(float(df["bps"].mean()), 2),
            "by_year": yrs}


def main() -> None:
    data = {s: load_days(s) for s in SYMBOLS}
    variants = build_variants()
    rows = []
    for v in variants:
        for s in SYMBOLS:
            sp = SPECS[s]
            for D in data[s]:
                out = v["fn"](D, sp)
                if out is not None:
                    rows.append((v["name"], v["family"], s, out[3], out[0], out[1], out[2]))
    df = pd.DataFrame(rows, columns=["variant", "family", "symbol", "day", "r", "bps", "why"])
    all_days = sorted({D["day"] for s in SYMBOLS for D in data[s]})
    split = all_days[len(all_days) // 2]
    exp_names = [v["name"] for v in variants if v["family"] != "CONTROL"]
    res = {}
    for v in variants:
        g = df[df.variant == v["name"]]
        res[v["name"]] = {"family": v["family"], **summarize(g)}
        res[v["name"]]["by_symbol"] = {s: summarize(g[g.symbol == s]) for s in SYMBOLS}
        res[v["name"]]["exits"] = {k: int(x) for k, x in g.why.value_counts().items()}
    qs = benjamini_hochberg([res[n]["p_value"] for n in exp_names])
    for n, q in zip(exp_names, qs):
        res[n]["q_value"] = round(q, 3)
    # seleccion en la primera mitad, evaluacion en la segunda
    is_df, oos_df = df[df.day < split], df[df.day >= split]
    ranking = sorted(((n, is_df[is_df.variant == n]["r"].mean()) for n in exp_names if (is_df.variant == n).sum() >= 100), key=lambda x: -x[1])
    sel = []
    for n, m in ranking[:5]:
        o = oos_df[oos_df.variant == n]
        sel.append({"variant": n, "is_mean_r": round(float(m), 3), "oos_n": int(len(o)), "oos_mean_r": round(float(o["r"].mean()), 3) if len(o) else None})
    best = sorted(exp_names, key=lambda n: -res[n].get("mean_r", -9))[:10]
    # subconjunto POST HOC: momentum de la primera hora solo en indices (US500 + NAS100). Se calcula despues de ver que el
    # efecto vive ahi, asi que su correccion por multiplicidad cuenta las 36 variantes FH y es solo una pista, no una prueba.
    fh_names = [n for n in exp_names if res[n]["family"] == "FH" and " mom " in n]
    idx_df = df[df.symbol.isin(["US500", "NAS100"])]
    idx_res = {n: summarize(idx_df[idx_df.variant == n]) for n in fh_names}
    for n, q in zip(fh_names, benjamini_hochberg([idx_res[n]["p_value"] for n in fh_names])):
        idx_res[n]["q_value"] = round(q, 3)
        a, b = idx_df[(idx_df.variant == n) & (idx_df.day < split)], idx_df[(idx_df.variant == n) & (idx_df.day >= split)]
        idx_res[n]["is_mean_r"] = round(float(a["r"].mean()), 3) if len(a) else None
        idx_res[n]["oos_mean_r"] = round(float(b["r"].mean()), 3) if len(b) else None
        idx_res[n]["oos_n"] = int(len(b))
    out = {"posthoc_indices_fh_mom": idx_res, "split_day": str(split), "days": len(all_days), "variants": len(exp_names), "symbols": SYMBOLS,
           "selection_is_oos": sel, "top10_by_mean_r": best, "results": res}
    (ROOT / "docs" / "intraday_vwap_momentum.json").write_text(json.dumps(out, indent=1, default=str))

    # ---- resumen por consola
    print(f"dias {len(all_days)} | variantes {len(exp_names)} | corte IS/OOS {split}")
    for fam in ("VR", "FH", "GAO"):
        names = [n for n in exp_names if res[n]["family"] == fam]
        sig = [n for n in names if res[n].get("q_value", 1) < 0.10]
        pos = [n for n in names if res[n].get("mean_r", 0) > 0]
        print(f"{fam}: {len(names)} variantes, R medio>0 en {len(pos)}, q<0,10 en {len(sig)}")
    print("\nTop 10 por R medio (pooled):")
    for n in best:
        r = res[n]
        yrs = " ".join(f"{y}:{d['mean_r']:+.2f}" for y, d in r["by_year"].items())
        print(f"  {n:55s} n={r['n']:5d} R={r['mean_r']:+.3f} IC={r['ci95']} p={r['p_value']} q={r['q_value']} PF={r['profit_factor']} | {yrs}")
    print("\nPeores 5:")
    for n in sorted(exp_names, key=lambda n: res[n].get("mean_r", 9))[:5]:
        r = res[n]
        print(f"  {n:55s} n={r['n']:5d} R={r['mean_r']:+.3f} IC={r['ci95']} p={r['p_value']}")
    print("\nControles:")
    for v in variants:
        if v["family"] == "CONTROL":
            r = res[v["name"]]
            print(f"  {v['name']:45s} n={r['n']} R={r['mean_r']:+.3f} IC={r['ci95']} bps={r['mean_bps']}")
    print("\nPOST HOC momentum primera hora solo indices (US500+NAS100), q sobre 18 variantes mom:")
    for n in sorted(fh_names, key=lambda n: -idx_res[n].get("mean_r", -9))[:6]:
        r = idx_res[n]
        yrs = " ".join(f"{y}:{d['mean_r']:+.2f}" for y, d in r["by_year"].items())
        print(f"  {n:50s} n={r['n']} R={r['mean_r']:+.3f} IC={r['ci95']} p={r['p_value']} q={r['q_value']} PF={r['profit_factor']} IS={r['is_mean_r']} OOS={r['oos_mean_r']} | {yrs}")
    print("\nSeleccion IS -> OOS (5 mejores de la primera mitad):")
    for x in sel:
        print(" ", x)


if __name__ == "__main__":
    main()
