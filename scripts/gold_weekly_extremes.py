"""Verificacion del hilo "el lunes y el viernes hacen el maximo o minimo semanal del oro" y estrategias derivadas.

Parte A (descriptiva): replica las cuentas del hilo con barras M15 de XAUUSD (septiembre 2023 a septiembre 2026; el hilo usa horarias
desde septiembre 2022). Dia = 18:00 a 17:00 hora de Nueva York (el lunes empieza en la apertura del domingo). Cuenta, por semana
completa, en que dia cae el maximo y el minimo, cuantas semanas hacen "ambos" los dias martes a jueves, y las ventanas "lunes hasta
medianoche NY" y "viernes 08:00-17:00 NY".

Parte B (nulo): que saldria en un paseo aleatorio? Para cualquier camino con incrementos intercambiables el momento del maximo
sigue (casi) la ley del arcoseno: se concentra al principio y al final del periodo (Sparre Andersen). El nulo permuta, dentro de cada
semana real, los incrementos de cierre a cierre de las barras (conserva volatilidad total, colas y deriva, destruye el orden) y recalcula
los mismos estadisticos 2.000 veces. Variante sin deriva (incrementos centrados) para separar el efecto "mercado alcista".
Limite del nulo: al permutar se pierde la estacionalidad horaria de la volatilidad (la Asia es mas tranquila), asi que es una referencia, no un oraculo.

Parte C (reglas operables, una operacion por semana como maximo, sin mirar al futuro, con spread y comision de `autotrader.intraday`):
- F-fade: viernes 08:00-14:00 NY, si el cierre esta a menos de k ATR diarios del maximo (minimo) de la semana, se vende (compra) contra el
  extremo; stop a 0,5 ATR, salida a las 17:00 o a 1R.
- F-break: viernes 08:00-14:00 NY, si el cierre supera el maximo (minimo) de la semana hasta ese momento, se entra a favor; stop a `s` ATR, salida 17:00.
- M-dip: lunes 08:00 NY, si el precio esta a menos de k ATR del minimo del lunes hasta ese momento, largo; stop 0,75 ATR; salida martes 17:00 o viernes 17:00.
- W-fade: miercoles 08:00 a jueves 14:00 NY, ruptura del rango lunes-martes por 0,1 ATR, entrada en contra; stop 0,5 ATR mas alla; salida jueves 17:00 o a la mitad del rango.
- Control (fuera de la correccion): largo fijo de domingo 18:00 a viernes 17:00 (la deriva del oro).
ATR diario = media de los rangos de los 10 dias anteriores a la semana. R = beneficio neto / distancia al stop. p-valor e IC por bootstrap de operaciones
y q de Benjamini-Hochberg sobre las 12 variantes.
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
from autotrader.stats import benjamini_hochberg, summarize_variant  # noqa: E402

COMMISSION = 0.000025
SPREAD = SPECS["XAUUSD"].spread
N_PERM = 2000
DAYS = ["Lun", "Mar", "Mie", "Jue", "Vie"]


def load_weeks():
    d = load_bars(str(ROOT / "data" / "intraday" / "XAUUSD_M15.csv"))
    d.index = d.index.tz_convert("America/New_York")
    shifted = d.index + pd.Timedelta(hours=6)  # 18:00 NY pasa a 00:00 del dia de negociacion
    d["wd"] = shifted.weekday
    d["tday"] = shifted.normalize()
    d["week"] = (shifted - pd.to_timedelta(shifted.weekday, unit="D")).normalize()
    d = d[d.wd < 5]
    # rango diario para el ATR
    daily = d.groupby("tday").agg(h=("high", "max"), l=("low", "min"))
    daily["rng"] = daily.h - daily.l
    weeks = []
    for wk, g in d.groupby("week"):
        if g.wd.nunique() < 5 or g.groupby("wd").size().min() < 40:
            continue
        prev = daily[daily.index < wk].tail(10)
        if len(prev) < 8:
            continue
        weeks.append({"week": wk, "bars": g, "atr": float(prev.rng.mean())})
    return weeks


def extremes(close_or_hl, wd, mins):
    """(dia del maximo, dia del minimo, minuto-NY del maximo, minuto-NY del minimo, idx maximo, idx minimo)."""
    hi, lo = close_or_hl
    ih, il = int(np.argmax(hi)), int(np.argmin(lo))
    return wd[ih], wd[il], mins[ih], mins[il], ih, il


def week_stats(hi, lo, wd, mins):
    dh, dl, mh, ml, _ih, _il = extremes((hi, lo), wd, mins)
    s = {"high_day": np.eye(5)[dh], "low_day": np.eye(5)[dl]}
    s["any_day"] = np.clip(s["high_day"] + s["low_day"], 0, 1)
    s["mid_both"] = float(1 <= dh <= 3 and 1 <= dl <= 3)
    mon_win = lambda d, m: d == 0  # el lunes de negociacion ya acaba a las 17:00; "hasta medianoche NY" = antes de 24:00 (min >= 18*60 o < 0)
    # ventana lunes: desde la apertura del domingo hasta la medianoche NY (minutos >= 18:00 del dia anterior) ; viernes 08:00-17:00 NY
    in_mon = lambda d, m: d == 0 and m >= 18 * 60
    in_fri = lambda d, m: d == 4 and 8 * 60 <= m < 17 * 60
    s["mon_win_ext"] = float(in_mon(dh, mh)) + float(in_mon(dl, ml))
    s["fri_win_ext"] = float(in_fri(dh, mh)) + float(in_fri(dl, ml))
    s["mon_or_fri_win"] = float(in_mon(dh, mh) or in_mon(dl, ml) or in_fri(dh, mh) or in_fri(dl, ml))
    s["mon_or_fri_day"] = float(dh in (0, 4) or dl in (0, 4))
    return s


def aggregate(stats_list):
    out = {k: np.sum([s[k] for s in stats_list], axis=0) for k in stats_list[0]}
    return {k: (v.tolist() if hasattr(v, "tolist") else float(v)) for k, v in out.items()}


def part_ab(weeks):
    obs_close, obs_hl, perms, perms0 = [], [], [], []
    rng = np.random.default_rng(7)
    for w in weeks:
        g = w["bars"]
        wd = g.wd.to_numpy()
        mins = (g.index.hour * 60 + g.index.minute).to_numpy()
        obs_hl.append(week_stats(g.high.to_numpy(), g.low.to_numpy(), wd, mins))
        c = g.close.to_numpy()
        obs_close.append(week_stats(c, c, wd, mins))
        inc = np.diff(np.log(c))
        w["inc"], w["wd_i"], w["mins_i"] = inc, wd[1:], mins[1:]
    keys = list(obs_close[0].keys())
    sims, sims0 = [], []
    for _ in range(N_PERM):
        a, b = [], []
        for w in weeks:
            inc, wd, mins = w["inc"], w["wd_i"], w["mins_i"]
            p = rng.permutation(inc)
            path = np.concatenate([[0.0], np.cumsum(p)])
            a.append(week_stats(path, path, np.concatenate([[w["bars"].wd.iloc[0]], wd]), np.concatenate([[mins[0]], mins])))
            p0 = rng.permutation(inc - inc.mean())
            path0 = np.concatenate([[0.0], np.cumsum(p0)])
            b.append(week_stats(path0, path0, np.concatenate([[w["bars"].wd.iloc[0]], wd]), np.concatenate([[mins[0]], mins])))
        sims.append(aggregate(a)); sims0.append(aggregate(b))
    obs = aggregate(obs_close)
    obs_h = aggregate(obs_hl)
    res = {"weeks": len(weeks), "observed_high_low": obs_h, "observed_close": obs, "null_mean": {}, "null_mean_no_drift": {}, "p_two_sided": {}, "p_two_sided_no_drift": {}}
    for k in keys:
        arr = np.array([s[k] for s in sims]); arr0 = np.array([s[k] for s in sims0]); o = np.array(obs[k])
        res["null_mean"][k] = np.round(arr.mean(axis=0), 1).tolist() if arr.ndim > 1 else round(float(arr.mean()), 1)
        res["null_mean_no_drift"][k] = np.round(arr0.mean(axis=0), 1).tolist() if arr0.ndim > 1 else round(float(arr0.mean()), 1)
        def p2(a):
            m = a.mean(axis=0)
            return (np.minimum((a >= o).mean(axis=0), (a <= o).mean(axis=0)) * 2).clip(0, 1).round(3).tolist() if a.ndim > 1 else round(float(min((a >= o).mean(), (a <= o).mean()) * 2), 3)
        res["p_two_sided"][k] = p2(arr); res["p_two_sided_no_drift"][k] = p2(arr0)
    # por ano: semanas con lunes o viernes en maximo/minimo
    by_year = {}
    for w, s in zip(weeks, obs_hl):
        y = str(w["week"].year); by_year.setdefault(y, [0, 0]); by_year[y][0] += int(s["mon_or_fri_day"]); by_year[y][1] += 1
    res["mon_or_fri_by_year"] = by_year
    return res


# ---------------------------------------------------------------------------------------------------------------- reglas
def trade(side, entry_px, stop_dist, bars, i_entry, i_exit, target_dist=None):
    """Entra en la apertura de la barra i_entry; devuelve (R neto, motivo). Un stop que se salta por hueco se ejecuta a la apertura."""
    o, h, l, c = bars.open.to_numpy(), bars.high.to_numpy(), bars.low.to_numpy(), bars.close.to_numpy()
    entry = o[i_entry] + side * SPREAD / 2
    stop = entry - side * stop_dist
    target = entry + side * target_dist if target_dist else None
    exit_px, why = None, "horario"
    for j in range(i_entry, i_exit + 1):
        if (side == 1 and l[j] <= stop) or (side == -1 and h[j] >= stop):
            exit_px, why = (min(stop, o[j]) if side == 1 else max(stop, o[j])), "stop"; break
        if target is not None and ((side == 1 and h[j] >= target) or (side == -1 and l[j] <= target)):
            exit_px, why = target, "objetivo"; break
    if exit_px is None:
        exit_px = c[i_exit]
    net = (exit_px - side * SPREAD / 2 - entry) * side - COMMISSION * (entry + exit_px)
    return net / stop_dist, why


def pos(bars, wd, hh, mm=0):
    """Indice de la primera barra del dia `wd` con minuto-NY >= hh:mm (considerando que el dia 0 empieza a las 18:00 del domingo)."""
    m = bars.index.hour * 60 + bars.index.minute
    idx = np.where((bars.wd.to_numpy() == wd) & (m >= hh * 60 + mm) & (m < 18 * 60))[0]
    return int(idx[0]) if len(idx) else None


def last_before_close(bars, wd):
    m = bars.index.hour * 60 + bars.index.minute
    idx = np.where((bars.wd.to_numpy() == wd) & (m < 17 * 60) | ((bars.wd.to_numpy() == wd) & (m >= 18 * 60) & False))[0]
    idx = np.where((bars.wd.to_numpy() == wd) & (m < 17 * 60))[0]
    return int(idx[-1]) if len(idx) else None


def rules(weeks):
    out = {}
    def add(name, fam, r, week):
        out.setdefault(name, {"family": fam, "rows": []})["rows"].append((week, r))
    for w in weeks:
        b, atr = w["bars"], w["atr"]
        wd = b.wd.to_numpy(); c = b.close.to_numpy(); h = b.high.to_numpy(); l = b.low.to_numpy()
        m = (b.index.hour * 60 + b.index.minute).to_numpy()
        # control: largo semanal
        i_fri_end = last_before_close(b, 4)
        r, _ = trade(1, 0, 2.0 * atr, b, 0, i_fri_end)
        add("CONTROL largo domingo-viernes (stop 2 ATR)", "CONTROL", r, w["week"])
        # F-fade / F-break
        fri_start, fri_cut = pos(b, 4, 8), pos(b, 4, 14)
        if fri_start is not None and fri_cut is not None and i_fri_end is not None:
            for k, ex in itertools.product([0.25, 0.5], ["cierre", "1R"]):
                fired = False
                for i in range(fri_start, fri_cut):
                    wk_hi, wk_lo = h[: i + 1].max(), l[: i + 1].min()
                    side = -1 if (wk_hi - c[i]) <= k * atr and (c[i] - wk_lo) > k * atr else (1 if (c[i] - wk_lo) <= k * atr and (wk_hi - c[i]) > k * atr else 0)
                    if side and i + 1 <= i_fri_end:
                        r, _ = trade(side, 0, 0.5 * atr, b, i + 1, i_fri_end, 0.5 * atr if ex == "1R" else None)
                        add(f"F-fade k{k} salida {ex}", "F-fade", r, w["week"]); fired = True; break
            for s_ in [0.5, 1.0]:
                for i in range(fri_start, fri_cut):
                    prev_hi, prev_lo = h[:i].max(), l[:i].min()
                    side = 1 if c[i] > prev_hi else (-1 if c[i] < prev_lo else 0)
                    if side and i + 1 <= i_fri_end:
                        r, _ = trade(side, 0, s_ * atr, b, i + 1, i_fri_end)
                        add(f"F-break stop{s_}ATR salida 17:00", "F-break", r, w["week"]); break
        # M-dip
        mon8 = pos(b, 0, 8)
        if mon8 is not None:
            for k, hold in itertools.product([0.25, 0.5], ["martes", "viernes"]):
                i_exit = last_before_close(b, 1 if hold == "martes" else 4)
                mon_lo = l[: mon8 + 1].min()
                if (c[mon8] - mon_lo) <= k * atr and i_exit is not None:
                    r, _ = trade(1, 0, 0.75 * atr, b, mon8 + 1, i_exit)
                    add(f"M-dip k{k} salida {hold}", "M-dip", r, w["week"])
        # W-fade
        wed8, thu_cut, thu_end = pos(b, 2, 8), pos(b, 3, 14), last_before_close(b, 3)
        tue_end = last_before_close(b, 1)
        if None not in (wed8, thu_cut, thu_end, tue_end):
            rng_hi, rng_lo = h[: tue_end + 1].max(), l[: tue_end + 1].min()
            mid = (rng_hi + rng_lo) / 2
            for ex in ["cierre", "mitad"]:
                for i in range(wed8, thu_cut):
                    side = -1 if c[i] > rng_hi + 0.1 * atr else (1 if c[i] < rng_lo - 0.1 * atr else 0)
                    if side and i + 1 <= thu_end:
                        sd = 0.5 * atr + 0.1 * atr
                        tgt = abs(mid - b.open.iloc[i + 1]) if ex == "mitad" else None
                        if tgt is not None and tgt <= 0:
                            break
                        r, _ = trade(side, 0, sd, b, i + 1, thu_end, tgt)
                        add(f"W-fade ruptura Lun-Mar salida {ex}", "W-fade", r, w["week"]); break
    return out


def main():
    weeks = load_weeks()
    print(f"semanas completas: {len(weeks)} ({weeks[0]['week'].date()} a {weeks[-1]['week'].date()})")
    ab = part_ab(weeks)
    n = ab["weeks"]
    print("\n== A/B. Maximo y minimo semanal (por cierres de M15; entre parentesis, esperado en paseo aleatorio con y sin deriva) ==")
    for key, label in [("high_day", "Maximo"), ("low_day", "Minimo"), ("any_day", "Maximo o minimo")]:
        print(label)
        for i, dname in enumerate(DAYS):
            print(f"  {dname}: observado {ab['observed_close'][key][i]:.0f} (hilo-hl {ab['observed_high_low'][key][i]:.0f}) | nulo {ab['null_mean'][key][i]} / sin deriva {ab['null_mean_no_drift'][key][i]} | p {ab['p_two_sided'][key][i]} / {ab['p_two_sided_no_drift'][key][i]}")
    for key, label in [("mid_both", "Mar-Jue hacen maximo Y minimo"), ("mon_or_fri_day", "Lunes o viernes hacen maximo o minimo"), ("mon_win_ext", "Extremos en lunes hasta medianoche"),
                       ("fri_win_ext", "Extremos en viernes 08-17"), ("mon_or_fri_win", "Ventanas lunes/viernes hacen algun extremo")]:
        print(f"{label}: observado {ab['observed_close'][key]:.0f} de {n} | nulo {ab['null_mean'][key]} / sin deriva {ab['null_mean_no_drift'][key]} | p {ab['p_two_sided'][key]} / {ab['p_two_sided_no_drift'][key]}")
    print("Lunes o viernes por ano:", ab["mon_or_fri_by_year"])

    rr = rules(weeks)
    res = {}
    for name, v in rr.items():
        rs = [r for _, r in v["rows"]]
        res[name] = {"family": v["family"], **summarize_variant(rs)}
        yrs = {}
        for wk, r in v["rows"]:
            yrs.setdefault(str(wk.year), []).append(r)
        res[name]["by_year"] = {y: {"n": len(x), "mean_r": round(float(np.mean(x)), 3)} for y, x in yrs.items()}
    exp = [k for k, v in res.items() if v["family"] != "CONTROL"]
    for k, q in zip(exp, benjamini_hochberg([res[k]["p_value"] for k in exp])):
        res[k]["q_value"] = round(q, 3)
    print("\n== C. Reglas (R neto por operacion; q sobre las %d variantes) ==" % len(exp))
    for k in sorted(res, key=lambda x: -res[x].get("mean_r", -9)):
        r = res[k]
        yrs = " ".join(f"{y}:{d['mean_r']:+.2f}" for y, d in r["by_year"].items())
        print(f"  {k:48s} n={r['n']:3d} R={r['mean_r']:+.3f} IC={r['ci95']} p={r['p_value']} q={r.get('q_value','-')} acierto={r['win_rate']} PF={r['profit_factor']} | {yrs}")
    out = {"weeks": n, "first_week": str(weeks[0]["week"].date()), "last_week": str(weeks[-1]["week"].date()), "descriptive": ab, "rules": res}
    (ROOT / "docs" / "gold_weekly_extremes.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
