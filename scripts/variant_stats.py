"""Correccion estadistica de los tres estudios intradia (un tiro por semana, rango de Londres, retest de Asia):
recalcula cada variante para obtener su lista de resultados en R por operacion, y para cada una calcula media, intervalo
de confianza al 95 % por bootstrap, p-valor unilateral (H0: media <= 0) y q-valor de Benjamini-Hochberg dentro de su
estudio y en el conjunto de las tres familias (que es el numero real de "intentos" hechos).

Uso: python scripts/variant_stats.py [--out docs/variant_stats.json] [--q 0.10]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from autotrader.intraday import SPECS, IntradayParams, SymbolSpec, load_bars  # noqa: E402
from autotrader.stats import benjamini_hochberg, summarize_variant  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    saved = sys.argv; sys.argv = ["x"]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = saved
    return mod


def one_shot_variants(risk: float) -> list[tuple[str, str, list[float]]]:
    m = _load("one_shot_week")
    data = {s: load_bars(os.path.join(ROOT, "data/intraday", f"{s}_M15.csv")) for s in m.UNIVERSES["indices_oro_petroleo_fx"]}
    start = min(df.index[0] for df in data.values()); end = max(df.index[-1] for df in data.values())
    base = dict(strategy="orb", or_bars=2, entry_window_bars=12, max_stop_pct=0.004, rr=3.0, breakeven_at_r=99.0, allow_short=True,
                risk_pct=0.01, max_risk_pct=0.05, commission_side=0.000025, max_notional_mult=1000.0)
    out = []
    weekly = [("base_rr3", "Semanal: primera ruptura ORB, 3R", {}, "indices_oro_petroleo_fx", None, "first"),
              ("rr2", "Semanal, 2R", dict(rr=2.0), "indices_oro_petroleo_fx", None, "first"),
              ("rr4", "Semanal, 4R", dict(rr=4.0), "indices_oro_petroleo_fx", None, "first"),
              ("rr3_be1", "Semanal, 3R con break even", dict(breakeven_at_r=1.0), "indices_oro_petroleo_fx", None, "first"),
              ("rr3_solo_largos", "Semanal, solo largos", dict(allow_short=False), "indices_oro_petroleo_fx", None, "first"),
              ("rr3_or4", "Semanal, rango de 1 h", dict(or_bars=4), "indices_oro_petroleo_fx", None, "first"),
              ("rr3_indices", "Semanal, solo índices", {}, "solo_indices", None, "first"),
              ("rr3_indices_oro", "Semanal, índices y oro", {}, "indices_y_oro", None, "first"),
              ("rr3_fx", "Semanal, solo divisas", {}, "solo_fx", None, "first"),
              ("rr3_mar_jue", "Semanal, martes a jueves", {}, "indices_oro_petroleo_fx", {1, 2, 3}, "first"),
              ("rr3_rango_mayor", "Semanal, símbolo con mayor rango", {}, "indices_oro_petroleo_fx", None, "range")]
    for key, desc, over, uni, wd, pick in weekly:
        tr = m.weekly_trades({s: data[s] for s in m.UNIVERSES[uni]}, IntradayParams(**{**base, **over}), risk, wd, pick)
        out.append((key, desc, [t["r"] for t in tr]))
    idx_gold = {s: data[s] for s in m.UNIVERSES["indices_y_oro"]}
    fomc = {pd.Timestamp(d).date() for d in m.FOMC_DAYS if start <= pd.Timestamp(d, tz="UTC") <= end}
    out.append(("nfp_orb", "Viernes de empleo, índices y oro, 3R", [t["r"] for t in m.event_trades(idx_gold, IntradayParams(**base), risk, m.nfp_days(start, end))]))
    out.append(("fomc_orb", "FOMC, índices y oro, 3R", [t["r"] for t in m.event_trades(idx_gold, IntradayParams(**base), risk, fomc, m.fomc_start_times(), hours=2.5)]))
    out.append(("fomc_orb_rr2", "FOMC, 2R", [t["r"] for t in m.event_trades(idx_gold, IntradayParams(**{**base, "rr": 2.0}), risk, fomc, m.fomc_start_times(), hours=2.5)]))
    gold = {"XAUUSD": data["XAUUSD"]}; orig = SPECS["XAUUSD"]
    for key, desc, sess, over, wd in [("oro_ny_rr3", "Solo oro, ORB Nueva York, 3R", ("13:30", "20:00"), {}, None),
                                      ("oro_ny_rr2", "Solo oro, ORB Nueva York, 2R", ("13:30", "20:00"), {"rr": 2.0}, None),
                                      ("oro_ny_be1", "Solo oro, ORB Nueva York, 3R con break even", ("13:30", "20:00"), {"breakeven_at_r": 1.0}, None),
                                      ("oro_londres_rr3", "Solo oro, ORB Londres, 3R", ("07:00", "16:00"), {}, None),
                                      ("oro_londres_or4", "Solo oro, ORB Londres rango 1 h, 3R", ("07:00", "16:00"), {"or_bars": 4}, None),
                                      ("oro_ny_mar_jue", "Solo oro, Nueva York, martes a jueves", ("13:30", "20:00"), {}, {1, 2, 3})]:
        SPECS["XAUUSD"] = SymbolSpec("XAUUSD", sess[0], sess[1], orig.spread, orig.step, orig.min_units, orig.digits)
        tr = m.weekly_trades(gold, IntradayParams(**{**base, **over}), risk, wd)
        out.append((key, desc, [t["r"] for t in tr]))
    SPECS["XAUUSD"] = orig
    out.append(("oro_nfp", "Solo oro, viernes de empleo, 3R", [t["r"] for t in m.event_trades(gold, IntradayParams(**base), risk, m.nfp_days(start, end))]))
    out.append(("oro_fomc", "Solo oro, FOMC, 3R", [t["r"] for t in m.event_trades(gold, IntradayParams(**base), risk, fomc, m.fomc_start_times(), hours=2.5)]))
    return out


def london_variants(risk: float) -> list[tuple[str, str, list[float]]]:
    m = _load("london_range_gold")
    S = m.sessions(load_bars(os.path.join(ROOT, "data/intraday/XAUUSD_M15.csv")))
    out = []
    for key, desc, kw in [("ruptura_mid_rr2", "Primera ruptura de Londres, stop mitad, 2R", dict(rr=2.0, stop_mode="mid")),
                          ("ruptura_mid_rr3", "Primera ruptura, stop mitad, 3R", dict(rr=3.0, stop_mode="mid")),
                          ("ruptura_opp_rr2", "Primera ruptura, stop lado contrario, 2R", dict(rr=2.0, stop_mode="opposite")),
                          ("ruptura_mid_rr2_17h", "Primera ruptura, 2R, salida 17:00", dict(rr=2.0, stop_mode="mid", exit_hm=17 * 60)),
                          ("ruptura_mid_rr3_17h", "Primera ruptura, 3R, salida 17:00", dict(rr=3.0, stop_mode="mid", exit_hm=17 * 60)),
                          ("ruptura_rango_estrecho", "Primera ruptura, 2R, rango de Londres estrecho", dict(rr=2.0, stop_mode="mid", max_range_pct=0.006)),
                          ("fade_mid_rr2", "Vender la ruptura, 2R", dict(rr=2.0, stop_mode="mid", fade=True)),
                          ("fade_mid_rr1", "Vender la ruptura, 1R", dict(rr=1.0, stop_mode="mid", fade=True))]:
        out.append((key, desc, [t["r"] for t in m.trade_first_break(S, risk, **kw)]))
    for key, desc, fn in [("retest_mid_rr2", "Retest de Londres, stop mitad, 2R", lambda: m.trade_retest(S, risk, 2.0)),
                          ("retest_mid_rr3", "Retest de Londres, 3R", lambda: m.trade_retest(S, risk, 3.0)),
                          ("retest_opp_rr2", "Retest de Londres, stop lado contrario, 2R", lambda: m.trade_retest(S, risk, 2.0, stop_mode="opposite")),
                          ("retest_mid_rr2_17h", "Retest de Londres, 2R, salida 17:00", lambda: m.trade_retest(S, risk, 2.0, exit_hm=17 * 60)),
                          ("retest_asia_mid_rr2", "RETEST DE ASIA, stop mitad, 2R (bot en demo)", lambda: m.trade_retest(S, risk, 2.0, level="asia")),
                          ("fallida_rr2", "Ruptura fallida, 2R", lambda: m.trade_failed_break(S, risk, 2.0)),
                          ("fallida_rr1", "Ruptura fallida, 1R", lambda: m.trade_failed_break(S, risk, 1.0)),
                          ("fallida_lado_contrario", "Ruptura fallida, objetivo lado contrario", lambda: m.trade_failed_break(S, risk, None, target="opposite")),
                          ("fallida_rr2_17h", "Ruptura fallida, 2R, salida 17:00", lambda: m.trade_failed_break(S, risk, 2.0, exit_hm=17 * 60)),
                          ("fallida_asia_rr2", "Ruptura fallida de Asia, 2R", lambda: m.trade_failed_break(S, risk, 2.0, level="asia"))]:
        out.append((key, desc, [t["r"] for t in fn()]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/variant_stats.json")
    ap.add_argument("--q", type=float, default=0.10, help="nivel de la tasa de falsos descubrimientos")
    ap.add_argument("--risk", type=float, default=100.0)
    args = ap.parse_args()
    families = {"un_tiro_por_semana": one_shot_variants(args.risk), "rango_londres_y_asia": london_variants(args.risk)}
    rows = []
    for fam, variants in families.items():
        for key, desc, R in variants:
            s = summarize_variant(R)
            rows.append({"family": fam, "key": key, "description": desc, **s})
    # q-valores dentro de cada familia y en el conjunto
    for fam in families:
        idx = [i for i, r in enumerate(rows) if r["family"] == fam and r.get("n", 0) > 1]
        q = benjamini_hochberg([rows[i]["p_value"] for i in idx])
        for i, qv in zip(idx, q):
            rows[i]["q_family"] = round(qv, 4)
    idx = [i for i, r in enumerate(rows) if r.get("n", 0) > 1]
    q = benjamini_hochberg([rows[i]["p_value"] for i in idx])
    for i, qv in zip(idx, q):
        rows[i]["q_all"] = round(qv, 4)
        rows[i]["significant_all"] = bool(qv <= args.q)
        rows[i]["significant_family"] = bool(rows[i]["q_family"] <= args.q)
    rows.sort(key=lambda r: r.get("p_value", 1.0))
    out = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "q_level": args.q, "tests": len(idx),
           "method": "media de R por operacion; IC 95 % bootstrap percentil (10.000); p unilateral bootstrap centrado (20.000); q Benjamini-Hochberg",
           "rows": rows}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"{'variante':48s} {'n':>4s} {'R medio':>8s} {'IC 95 %':>16s} {'p':>7s} {'q fam':>7s} {'q todo':>7s}")
    for r in rows:
        if r.get("n", 0) < 2:
            continue
        print(f"{r['description'][:48]:48s} {r['n']:4d} {r['mean_r']:+8.3f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] {r['p_value']:7.4f} {r['q_family']:7.3f} {r['q_all']:7.3f}"
              + ("  *" if r["significant_all"] else ("  (fam)" if r["significant_family"] else "")))
    print("->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
