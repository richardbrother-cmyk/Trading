"""Corroboracion del hilo sobre el rango de Londres en el oro y prueba de la operativa que propone.

Sesiones en hora de Nueva York (con horario de verano): Asia 19:00-03:00, Londres 03:00-08:00, manana de NY 08:00-12:00,
dia completo 18:00 (vispera) a 17:00. Fuente: barras M15 (bid) de XAUUSD en data/intraday.

Uso: python scripts/london_range_gold.py [--days 250] [--risk 100] [--out docs/london_range_gold.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autotrader.intraday import load_bars  # noqa: E402

NY = ZoneInfo("America/New_York")
SPREAD = 0.20


def sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por dia de negociacion (18:00 NY de la vispera a 17:00 NY) con los rangos por sesion."""
    ny = df.tz_convert(NY)
    # dia de negociacion: las barras desde las 18:00 pertenecen al dia siguiente
    tday = (ny.index + pd.Timedelta(hours=6)).date
    ny = ny.assign(tday=tday, hm=ny.index.hour * 60 + ny.index.minute)
    rows = []
    for day, g in ny.groupby("tday"):
        if pd.Timestamp(day).weekday() >= 5 or len(g) < 40:
            continue
        asia = g[(g.hm >= 19 * 60) | (g.hm < 3 * 60)]
        lon = g[(g.hm >= 3 * 60) & (g.hm < 8 * 60)]
        nym = g[(g.hm >= 8 * 60) & (g.hm < 12 * 60)]
        full = g[(g.hm >= 18 * 60) | (g.hm < 17 * 60)]
        if len(lon) < 15 or len(nym) < 12 or len(asia) < 15:
            continue
        rows.append({"day": str(day), "asia_hi": asia.high.max(), "asia_lo": asia.low.min(), "lon_hi": lon.high.max(), "lon_lo": lon.low.min(),
                     "ny_hi": nym.high.max(), "ny_lo": nym.low.min(), "day_hi": full.high.max(), "day_lo": full.low.min(),
                     "ny_open": nym.open.iloc[0], "ny_close": nym.close.iloc[-1], "day_close": full.close.iloc[-1],
                     "nym": nym[["open", "high", "low", "close"]], "after": g[g.hm >= 12 * 60][["open", "high", "low", "close"]]})
    return pd.DataFrame(rows)


def counts(S: pd.DataFrame) -> dict:
    day_rng = S.day_hi - S.day_lo
    broke_hi = S.ny_hi > S.lon_hi
    broke_lo = S.ny_lo < S.lon_lo
    out = {"days": len(S),
           "london_broken_in_ny_morning": round(float((broke_hi | broke_lo).mean()), 3),
           "london_high_broken": round(float(broke_hi.mean()), 3), "london_low_broken": round(float(broke_lo.mean()), 3),
           "both_sides_broken": round(float((broke_hi & broke_lo).mean()), 3),
           "london_range_pct_of_day": round(float(((S.lon_hi - S.lon_lo) / day_rng).mean()), 3),
           "ny_morning_range_pct_of_day": round(float(((S.ny_hi - S.ny_lo) / day_rng).mean()), 3),
           "asia_range_pct_of_day": round(float(((S.asia_hi - S.asia_lo) / day_rng).mean()), 3),
           "london_sets_day_high_or_low": round(float(((S.lon_hi >= S.day_hi) | (S.lon_lo <= S.day_lo)).mean()), 3),
           "ny_morning_sets_day_high_or_low": round(float(((S.ny_hi >= S.day_hi) | (S.ny_lo <= S.day_lo)).mean()), 3),
           "asia_sets_day_high_or_low": round(float(((S.asia_hi >= S.day_hi) | (S.asia_lo <= S.day_lo)).mean()), 3),
           "asia_range_broken_in_ny_morning": round(float(((S.ny_hi > S.asia_hi) | (S.ny_lo < S.asia_lo)).mean()), 3)}
    # tras la primera ruptura: se queda el precio al otro lado?
    stay, ext_r, first_side = [], [], []
    for _, r in S.iterrows():
        side, t_break = 0, None
        for t, b in r.nym.iterrows():
            if b.high > r.lon_hi and b.low < r.lon_lo:
                side = 1 if b.close >= (r.lon_hi + r.lon_lo) / 2 else -1; t_break = t; break
            if b.high > r.lon_hi:
                side, t_break = 1, t; break
            if b.low < r.lon_lo:
                side, t_break = -1, t; break
        if side == 0:
            continue
        first_side.append(side)
        rng = r.lon_hi - r.lon_lo
        if side == 1:
            stay.append(r.ny_close > r.lon_hi); ext_r.append((r.ny_hi - r.lon_hi) / rng)
        else:
            stay.append(r.ny_close < r.lon_lo); ext_r.append((r.lon_lo - r.ny_lo) / rng)
    out.update({"first_break_days": len(stay), "first_break_up_share": round(float(np.mean([s == 1 for s in first_side])), 3),
                "stays_beyond_at_noon": round(float(np.mean(stay)), 3),
                "extension_beyond_level_in_london_ranges_p50": round(float(np.median(ext_r)), 2),
                "extension_p25_p75": [round(float(np.percentile(ext_r, 25)), 2), round(float(np.percentile(ext_r, 75)), 2)]})
    return out


def trade_first_break(S: pd.DataFrame, risk: float, rr: float, stop_mode: str = "mid", fade: bool = False, exit_hm: int = 12 * 60,
                      min_range_pct: float = 0.0, max_range_pct: float = 9.0) -> list[dict]:
    """Entrada al cierre de la primera vela de 15 min de la manana de NY que cierra fuera del rango de Londres.
    stop_mode: 'mid' (mitad del rango) u 'opposite' (lado contrario). fade=True opera en contra (vuelta al rango).
    Salida por stop, objetivo rr x riesgo, o al cierre de la ventana (12:00 NY por defecto; 17:00 con exit_hm=17*60)."""
    trades = []
    for _, r in S.iterrows():
        rng = r.lon_hi - r.lon_lo
        rng_pct = rng / r.ny_open
        if rng <= 0 or rng_pct < min_range_pct or rng_pct > max_range_pct:
            continue
        bars = pd.concat([r.nym, r.after]) if exit_hm > 12 * 60 else r.nym
        hm = bars.index.hour * 60 + bars.index.minute
        bars = bars[hm < exit_hm]
        side, i0 = 0, None
        for i, (t, b) in enumerate(bars.iterrows()):
            if t.hour * 60 + t.minute >= 12 * 60:
                break  # la entrada solo en la manana de NY
            if b.close > r.lon_hi:
                side, i0 = 1, i; break
            if b.close < r.lon_lo:
                side, i0 = -1, i; break
        if side == 0 or i0 is None or i0 + 1 >= len(bars):
            continue
        d = -side if fade else side
        entry = bars.open.iloc[i0 + 1] + d * SPREAD / 2
        if fade:
            stop = entry - d * rng * (0.5 if stop_mode == "mid" else 1.0)
        else:
            stop = (r.lon_hi + r.lon_lo) / 2 if stop_mode == "mid" else (r.lon_lo if side == 1 else r.lon_hi)
        dist = abs(entry - stop)
        if dist <= 0:
            continue
        tp = entry + d * rr * dist
        units = risk / dist
        exit_px, reason = None, ""
        for j in range(i0 + 1, len(bars)):
            b = bars.iloc[j]
            if d == 1 and b.low <= stop or d == -1 and b.high >= stop:
                exit_px, reason = stop, "stop"; break
            if d == 1 and b.high >= tp or d == -1 and b.low <= tp:
                exit_px, reason = tp, "objetivo"; break
        if exit_px is None:
            exit_px, reason = bars.close.iloc[-1], "cierre"
        pnl = (exit_px - entry) * d * units - SPREAD / 2 * units
        trades.append({"day": r.day, "side": d, "entry": round(entry, 2), "stop": round(stop, 2), "tp": round(tp, 2), "exit": round(exit_px, 2),
                       "reason": reason, "pnl": round(pnl, 2), "r": round(pnl / risk, 3), "london_range_pct": round(rng_pct * 100, 3)})
    return trades


def summarize(tr: list[dict]) -> dict:
    if not tr:
        return {"trades": 0}
    R = np.array([t["r"] for t in tr]); pnl = np.array([t["pnl"] for t in tr])
    cum = np.concatenate([[0.0], np.cumsum(pnl)]); dd = float((cum - np.maximum.accumulate(cum)).min())
    streak = mx = 0
    for r in R:
        streak = streak + 1 if r <= 0 else 0; mx = max(mx, streak)
    years = {}
    for t in tr:
        years.setdefault(t["day"][:4], []).append(t["pnl"])
    return {"trades": len(tr), "win_rate": round(float((R > 0).mean()), 3), "target_rate": round(float(np.mean([t["reason"] == "objetivo" for t in tr])), 3),
            "avg_r": round(float(R.mean()), 3), "total_usd": round(float(pnl.sum()), 2),
            "profit_factor": round(float(pnl[pnl > 0].sum() / -pnl[pnl <= 0].sum()), 2) if (pnl <= 0).any() else None,
            "max_dd_usd": round(dd, 2), "max_losing_streak": mx, "by_year": {y: round(float(sum(v)), 2) for y, v in years.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/intraday/XAUUSD_M15.csv")
    ap.add_argument("--days", type=int, default=250)
    ap.add_argument("--risk", type=float, default=100.0)
    ap.add_argument("--out", default="docs/london_range_gold.json")
    args = ap.parse_args()
    S = sessions(load_bars(args.data))
    last = S.tail(args.days)
    res = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "period": [S.day.iloc[0], S.day.iloc[-1]],
           "counts_last": {"days_requested": args.days, **counts(last)}, "counts_all": counts(S), "risk_usd": args.risk, "strategies": {}}
    print(f"periodo {res['period']} · {len(S)} dias")
    for k, v in (("ultimos %d dias" % args.days, res["counts_last"]), ("3 anos", res["counts_all"])):
        print(f"[{k}] " + ", ".join(f"{a}={b}" for a, b in v.items()))
    variants = [
        ("ruptura_mid_rr2", "Primera ruptura, stop en mitad del rango, 2R, salida 12:00 NY", dict(rr=2.0, stop_mode="mid")),
        ("ruptura_mid_rr3", "Primera ruptura, stop en mitad del rango, 3R, salida 12:00 NY", dict(rr=3.0, stop_mode="mid")),
        ("ruptura_opp_rr2", "Primera ruptura, stop en el lado contrario, 2R, salida 12:00 NY", dict(rr=2.0, stop_mode="opposite")),
        ("ruptura_mid_rr2_17h", "Primera ruptura, stop en mitad, 2R, salida 17:00 NY", dict(rr=2.0, stop_mode="mid", exit_hm=17 * 60)),
        ("ruptura_mid_rr3_17h", "Primera ruptura, stop en mitad, 3R, salida 17:00 NY", dict(rr=3.0, stop_mode="mid", exit_hm=17 * 60)),
        ("ruptura_mid_rr2_rango_estrecho", "Primera ruptura, 2R, solo si el rango de Londres < 0,6 % del precio", dict(rr=2.0, stop_mode="mid", max_range_pct=0.006)),
        ("fade_mid_rr2", "Lo contrario: vender la ruptura (vuelta al rango), stop medio rango, 2R", dict(rr=2.0, stop_mode="mid", fade=True)),
        ("fade_mid_rr1", "Lo contrario con objetivo 1R", dict(rr=1.0, stop_mode="mid", fade=True)),
    ]
    for key, desc, kw in variants:
        tr = trade_first_break(S, args.risk, **kw)
        m = summarize(tr)
        res["strategies"][key] = {"description": desc, **m}
        print(f"{key:32s} ops {m['trades']:3d} acierto {m.get('win_rate',0):.0%} objetivo {m.get('target_rate',0):.0%} R medio {m.get('avg_r',0):+.2f} "
              f"total {m.get('total_usd',0):7.0f} PF {m.get('profit_factor')} DD {m.get('max_dd_usd',0):6.0f} racha {m.get('max_losing_streak',0)} años {m.get('by_year')}")
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print("->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
