"""Investigacion "un tiro por semana": una sola operacion intradia por semana, riesgo fijo en USD, objetivo N veces el riesgo.

Regla simulada: cada semana se toma la PRIMERA senal de ruptura del rango de apertura (ORB, motor de autotrader.intraday) que
aparezca en cualquier simbolo del universo; se arriesga una cantidad fija (100 USD), objetivo `rr` x riesgo, y se cierra al
final de la sesion si no toca stop ni objetivo. Despues no se opera mas esa semana, gane o pierda.

Uso: python scripts/one_shot_week.py [--risk 100] [--out docs/one_shot_week.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autotrader.intraday import SPECS, IntradayParams, _session_slices, _simulate_day, load_bars  # noqa: E402

UNIVERSES = {
    "indices_oro_petroleo_fx": ["US500", "NAS100", "XAUUSD", "XTIUSD", "EURUSD", "GBPUSD"],
    "solo_indices": ["US500", "NAS100"],
    "indices_y_oro": ["US500", "NAS100", "XAUUSD"],
    "solo_fx": ["EURUSD", "GBPUSD"],
}


def weekly_trades(data: dict[str, pd.DataFrame], p: IntradayParams, risk_usd: float, weekdays: set[int] | None = None,
                  pick: str = "first") -> list[dict]:
    """Una operacion por semana ISO: la primera senal (o la del simbolo con mayor rango de apertura relativo) del universo."""
    equity = risk_usd / p.risk_pct  # tamano fijo: risk_pct * equity = risk_usd
    days: dict = {}
    for sym, df in data.items():
        for day, g in _session_slices(df, SPECS[sym]):
            days.setdefault(day, []).append((sym, g))
    out, done_weeks = [], set()
    for day in sorted(days):
        wk = pd.Timestamp(day).isocalendar()[:2]
        if wk in done_weeks or (weekdays is not None and pd.Timestamp(day).weekday() not in weekdays):
            continue
        cands = []
        for sym, g in days[day]:
            t, _sk = _simulate_day(sym, SPECS[sym], g, p, equity)
            if t is not None:
                or_range = (g["high"].iloc[: p.or_bars].max() - g["low"].iloc[: p.or_bars].min()) / g["close"].iloc[0]
                cands.append((t.entry_time, -or_range, sym, t))
        if not cands:
            continue
        cands.sort(key=lambda x: (x[0] if pick == "first" else x[1]))
        t = cands[0][3]
        risk = abs(t.entry - t.stop) * t.units
        out.append({"week": f"{wk[0]}-W{wk[1]:02d}", "day": str(day), "symbol": t.symbol, "side": t.side, "entry": t.entry, "stop": t.stop,
                    "exit": t.exit, "reason": t.reason, "pnl": round(t.pnl, 2), "r": round(t.pnl / risk, 3) if risk else 0.0,
                    "risk_usd": round(risk, 2)})
        done_weeks.add(wk)
    return out


FOMC_DAYS = ["2023-09-20", "2023-11-01", "2023-12-13", "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18",
             "2024-11-07", "2024-12-18", "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29",
             "2025-12-10", "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16"]


def nfp_days(start: pd.Timestamp, end: pd.Timestamp) -> set:
    """Primer viernes de cada mes (dato de empleo de EE. UU., 12:30 UTC)."""
    out = set()
    for m in pd.date_range(start.normalize(), end.normalize(), freq="MS"):
        d = m + pd.Timedelta(days=(4 - m.weekday()) % 7)
        out.add(d.date())
    return out


def event_trades(data: dict[str, pd.DataFrame], p: IntradayParams, risk_usd: float, days_set: set, start_utc: dict[str, str] | None = None,
                 hours: float = 3.0) -> list[dict]:
    """Una operacion por dia de evento: ORB anclado a la hora del evento (o a la apertura de sesion) con la primera senal del universo."""
    from autotrader.intraday import SymbolSpec
    equity = risk_usd / p.risk_pct
    out = []
    for day in sorted(days_set):
        cands = []
        for sym, df in data.items():
            spec = SPECS[sym]
            if start_utc:
                st = start_utc[str(day)] if str(day) in start_utc else None
                if st is None:
                    continue
                sh, sm = int(st[:2]), int(st[3:])
                eh = min(23, sh + int(hours)); em = sm
                spec = SymbolSpec(sym, st, f"{eh:02d}:{em:02d}", spec.spread, spec.step, spec.min_units, spec.digits)
            g = df[(df.index.date == day)]
            t_ = g.index; mins = t_.hour * 60 + t_.minute
            s0 = int(spec.session_start[:2]) * 60 + int(spec.session_start[3:]); e0 = int(spec.session_end[:2]) * 60 + int(spec.session_end[3:])
            g = g[(mins >= s0) & (mins < e0)]
            if len(g) < 6:
                continue
            t, _sk = _simulate_day(sym, spec, g, p, equity)
            if t is not None:
                cands.append((t.entry_time, sym, t))
        if not cands:
            continue
        cands.sort(key=lambda x: x[0])
        t = cands[0][2]
        risk = abs(t.entry - t.stop) * t.units
        out.append({"week": pd.Timestamp(day).strftime("%G-W%V"), "day": str(day), "symbol": t.symbol, "side": t.side, "entry": t.entry, "stop": t.stop,
                    "exit": t.exit, "reason": t.reason, "pnl": round(t.pnl, 2), "r": round(t.pnl / risk, 3) if risk else 0.0, "risk_usd": round(risk, 2)})
    return out


def fomc_start_times() -> dict[str, str]:
    """Hora UTC de las 14:00 de Nueva York en cada dia de FOMC (cambia con el horario de verano)."""
    from zoneinfo import ZoneInfo
    out = {}
    for d in FOMC_DAYS:
        ny = pd.Timestamp(f"{d} 14:00", tz=ZoneInfo("America/New_York")).tz_convert("UTC")
        out[d] = ny.strftime("%H:%M")
    return out


def summarize(trades: list[dict], risk_usd: float, weeks_total: int) -> dict:
    if not trades:
        return {"trades": 0}
    R = np.array([t["r"] for t in trades]); pnl = np.array([t["pnl"] for t in trades])
    cum = np.cumsum(pnl); peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))
    dd = float((np.concatenate([[0.0], cum]) - peak).min())
    streak = mx = 0
    for r in R:
        streak = streak + 1 if r <= 0 else 0; mx = max(mx, streak)
    years = {}
    for t in trades:
        years.setdefault(t["day"][:4], []).append(t["pnl"])
    reasons = pd.Series([t["reason"] for t in trades]).value_counts().to_dict()
    return {"trades": len(trades), "weeks_total": weeks_total, "weeks_traded_pct": round(len(trades) / weeks_total, 3),
            "win_rate": round(float((R > 0).mean()), 3), "target_rate": round(reasons.get("objetivo", 0) / len(trades), 3),
            "avg_r": round(float(R.mean()), 3), "sum_r": round(float(R.sum()), 1), "total_usd": round(float(pnl.sum()), 2),
            "profit_factor": round(float(pnl[pnl > 0].sum() / -pnl[pnl <= 0].sum()), 2) if (pnl <= 0).any() else None,
            "max_dd_usd": round(dd, 2), "max_losing_streak": mx, "reasons": reasons,
            "by_year": {y: round(float(sum(v)), 2) for y, v in years.items()},
            "by_symbol": pd.Series([t["symbol"] for t in trades]).value_counts().to_dict(),
            "risk_usd": risk_usd}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/intraday")
    ap.add_argument("--risk", type=float, default=100.0)
    ap.add_argument("--out", default="docs/one_shot_week.json")
    args = ap.parse_args()
    data = {s: load_bars(os.path.join(args.data, f"{s}_M15.csv")) for s in UNIVERSES["indices_oro_petroleo_fx"]}
    start = min(df.index[0] for df in data.values()); end = max(df.index[-1] for df in data.values())
    weeks_total = int(np.ceil((end - start).days / 7))
    base = dict(strategy="orb", or_bars=2, entry_window_bars=12, max_stop_pct=0.004, rr=3.0, breakeven_at_r=99.0, allow_short=True,
                risk_pct=0.01, max_risk_pct=0.05, commission_side=0.000025, max_notional_mult=1000.0)
    variants = [
        ("base_rr3", "Primera señal ORB de la semana, objetivo 3R, sin break even, largos y cortos", dict(), "indices_oro_petroleo_fx", None, "first"),
        ("rr2", "Igual con objetivo 2R", dict(rr=2.0), "indices_oro_petroleo_fx", None, "first"),
        ("rr4", "Igual con objetivo 4R", dict(rr=4.0), "indices_oro_petroleo_fx", None, "first"),
        ("rr3_be1", "Objetivo 3R con stop a break even tras 1R", dict(breakeven_at_r=1.0), "indices_oro_petroleo_fx", None, "first"),
        ("rr3_solo_largos", "Objetivo 3R, solo largos", dict(allow_short=False), "indices_oro_petroleo_fx", None, "first"),
        ("rr3_or4", "Objetivo 3R con rango de apertura de 1 h (4 barras)", dict(or_bars=4), "indices_oro_petroleo_fx", None, "first"),
        ("rr3_indices", "Objetivo 3R, solo US500 y NAS100", dict(), "solo_indices", None, "first"),
        ("rr3_indices_oro", "Objetivo 3R, índices y oro", dict(), "indices_y_oro", None, "first"),
        ("rr3_fx", "Objetivo 3R, solo EURUSD y GBPUSD", dict(), "solo_fx", None, "first"),
        ("rr3_mar_jue", "Objetivo 3R, solo martes a jueves", dict(), "indices_oro_petroleo_fx", {1, 2, 3}, "first"),
        ("rr3_rango_mayor", "Objetivo 3R, elige el símbolo con mayor rango de apertura del día", dict(), "indices_oro_petroleo_fx", None, "range"),
    ]
    rows = []
    for key, desc, over, uni, wd, pick in variants:
        p = IntradayParams(**{**base, **over})
        sub = {s: data[s] for s in UNIVERSES[uni]}
        tr = weekly_trades(sub, p, args.risk, wd, pick)
        m = summarize(tr, args.risk, weeks_total)
        rows.append({"key": key, "description": desc, "universe": UNIVERSES[uni], "params": {**base, **over}, **m,
                     "trades_list": tr if key == "base_rr3" else None})
        print(f"{key:18s} ops {m.get('trades',0):3d}/{weeks_total} sem  acierto {m.get('win_rate',0):.0%}  objetivo {m.get('target_rate',0):.0%}  "
              f"R medio {m.get('avg_r',0):+.2f}  total {m.get('total_usd',0):8.0f} USD  PF {m.get('profit_factor')}  DD {m.get('max_dd_usd',0):7.0f}  "
              f"racha {m.get('max_losing_streak',0)}  años {m.get('by_year')}")
    # Dias de evento: ORB en la apertura de contado en viernes de empleo, y ORB anclado a las 14:00 de Nueva York en dias de FOMC
    idx_gold = {s: data[s] for s in UNIVERSES["indices_y_oro"]}
    for key, desc, tr in [
        ("nfp_orb", "Solo viernes de empleo (primer viernes de mes): ORB en la apertura, índices y oro, objetivo 3R",
         event_trades(idx_gold, IntradayParams(**base), args.risk, nfp_days(start, end))),
        ("fomc_orb", "Solo días de FOMC: ruptura del rango de la primera media hora tras el comunicado (14:00 NY), índices y oro, objetivo 3R",
         event_trades(idx_gold, IntradayParams(**base), args.risk, {pd.Timestamp(d).date() for d in FOMC_DAYS if start <= pd.Timestamp(d, tz="UTC") <= end},
                      fomc_start_times(), hours=2.5)),
        ("fomc_orb_rr2", "Días de FOMC, objetivo 2R", event_trades(idx_gold, IntradayParams(**{**base, "rr": 2.0}), args.risk,
                                                                     {pd.Timestamp(d).date() for d in FOMC_DAYS if start <= pd.Timestamp(d, tz="UTC") <= end}, fomc_start_times(), hours=2.5)),
    ]:
        m = summarize(tr, args.risk, weeks_total)
        rows.append({"key": key, "description": desc, "universe": UNIVERSES["indices_y_oro"], "params": base, **m, "trades_list": tr})
        print(f"{key:18s} ops {m.get('trades',0):3d}  acierto {m.get('win_rate',0):.0%}  objetivo {m.get('target_rate',0):.0%}  R medio {m.get('avg_r',0):+.2f}  "
              f"total {m.get('total_usd',0):8.0f} USD  PF {m.get('profit_factor')}  DD {m.get('max_dd_usd',0):7.0f}  racha {m.get('max_losing_streak',0)}  años {m.get('by_year')}")
    out = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "period": [str(start.date()), str(end.date())],
           "weeks": weeks_total, "risk_usd": args.risk, "variants": rows}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, default=str)
    print("->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
