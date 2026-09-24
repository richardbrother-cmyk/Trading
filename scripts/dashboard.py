"""Genera reports/dashboard.html: backtests comparados, estado de la cuenta paper y órdenes en cola.

Uso: python scripts/dashboard.py [--out reports/dashboard.html] [--no-live]
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autotrader.backtest import run_backtest  # noqa: E402
from autotrader.attribution import bot_metrics, classify, cutoff_for, fifo_trades, load_exclusions
from autotrader.config import Settings  # noqa: E402
from autotrader.data import DataProvider  # noqa: E402
from autotrader.events import active_events, load_events, upcoming_events  # noqa: E402
from autotrader.risk import RiskParams  # noqa: E402
from autotrader.strategy import StrategyParams  # noqa: E402

UNIVERSES = [
    ("Acciones e índices", ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"], 5, 0.25),
    ("Metales físicos", ["GLD", "SLV", "PPLT", "PALL"], 5, 0.25),
    ("Universo completo (activo)", ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "GLD", "SLV", "PPLT", "PALL", "USO", "WEAT", "CORN", "DBA"], 10, 0.12),
]


ALPACA_CACHE = "docs/alpaca_state.json"
CTRADER_STATE = "docs/ctrader_state.json"
SWING_STATE = "docs/swing_state.json"
AGGR_STATE = "docs/aggr_state.json"
AGGR_BREAKEVEN = "docs/aggr_breakeven.json"
AGGR_SIMULATION = "docs/aggr_simulation.json"
ASIA_STATE = "docs/asia_state.json"
SWING_WALKFORWARD = "docs/swing_walkforward.json"
KRONOS_REPORT = "docs/kronos_report.json"
ELLIOTT_STUDY = "docs/elliott_study.json"
ELLIOTT_STATE = "docs/elliott_state.json"
RESISTANCE_STUDY = "docs/resistance_exit.json"
ELLIOTT_DEPLOYED_KEY = "XAUUSD_H4_zz2_w2_ruptura_t1.618_inicio_l"  # variante del estudio que opera el bot de Elliott
ALPACA_VARIANTS = "docs/alpaca_variants.json"


def collect(no_live: bool) -> dict:
    s = Settings.from_env(".env")
    provider = DataProvider("yahoo")
    all_symbols = sorted({sym for _, syms, _, _ in UNIVERSES for sym in syms})
    data = provider.bars_many(all_symbols)
    strategy = StrategyParams(s.fast_sma, s.slow_sma, s.rsi_period, s.rsi_max_entry)
    backtests = []
    for name, syms, max_pos, cap in UNIVERSES:
        sub = {k: v for k, v in data.items() if k in syms}
        res = run_backtest(sub, strategy, RiskParams(s.risk_per_trade, max_pos, cap, s.max_daily_loss_pct, s.stop_loss_pct), s.initial_cash)
        eq = res.equity
        backtests.append({
            "name": name, "symbols": syms, "max_positions": max_pos, "max_position_pct": cap,
            "metrics": res.metrics(),
            "equity": [[d.strftime("%Y-%m-%d"), round(float(v), 2)] for d, v in eq.items()],
            "trades": [{"symbol": t.symbol, "entry": t.entry_date.strftime("%Y-%m-%d"), "exit": t.exit_date.strftime("%Y-%m-%d"),
                        "entry_price": round(t.entry_price, 2), "exit_price": round(t.exit_price, 2), "qty": t.qty,
                        "pnl": round(t.pnl, 2), "ret": round(t.ret, 4), "reason": t.reason} for t in res.trades],
        })
    live = {"available": False}
    if no_live and os.path.exists(ALPACA_CACHE):
        with open(ALPACA_CACHE, encoding="utf-8") as fh:
            live = json.load(fh)
        live["stale"] = True
    if not no_live and s.alpaca_api_key:
        from autotrader.broker import AlpacaBroker
        b = AlpacaBroker(s.alpaca_api_key, s.alpaca_secret_key)
        acct = b._get("/v2/account")
        clock = b._get("/v2/clock")
        orders = b._get("/v2/orders", status="open", nested="true")
        stops = {}
        for o in orders:
            for leg in o.get("legs") or []:
                if leg.get("type") == "stop" and leg.get("stop_price"):
                    stops[o["symbol"]] = float(leg["stop_price"])
            if o.get("type") == "stop" and o.get("side") == "sell" and o.get("stop_price"):
                stops[o["symbol"]] = float(o["stop_price"])
        orders = [o for o in orders if o.get("type") != "stop"]
        positions = b._get("/v2/positions")
        hist = b._get("/v2/account/portfolio/history", period="3M", timeframe="1D")
        history = [[datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d"), round(float(e), 2)]
                   for t, e in zip(hist.get("timestamp", []), hist.get("equity", [])) if e and float(e) > 0]
        all_orders = b._get("/v2/orders", status="all", limit=500, direction="desc")
        live = {
            "history": history,
            "orders_all": [{"symbol": o["symbol"], "side": o["side"], "qty": int(float(o["qty"])), "type": o["type"],
                            "stop_price": float(o["stop_price"]) if o.get("stop_price") else None,
                            "tif": o["time_in_force"], "status": o["status"],
                            "filled_price": float(o["filled_avg_price"]) if o.get("filled_avg_price") else None,
                            "at": (o.get("filled_at") or o.get("canceled_at") or o["submitted_at"])[:16].replace("T", " ")} for o in all_orders],
            "available": True, "broker": "Alpaca paper", "equity": float(acct["equity"]), "cash": float(acct["cash"]),
            "buying_power": float(acct["buying_power"]), "is_open": clock["is_open"], "next_open": clock["next_open"],
            "orders": [{"symbol": o["symbol"], "side": o["side"], "qty": int(float(o["qty"])), "status": o["status"],
                        "stop": stops.get(o["symbol"]), "submitted_at": o["submitted_at"][:16].replace("T", " ")} for o in orders],
            "stops": stops,
            "positions": [{"symbol": p["symbol"], "qty": int(float(p["qty"])), "avg": float(p["avg_entry_price"]),
                           "price": float(p["current_price"]), "pnl": float(p["unrealized_pl"])} for p in positions],
        }
    if live.get("available"):
        # Operaciones cerradas (FIFO sobre las ejecuciones) y metricas del bot sin las entradas excluidas
        exclusions = load_exclusions()
        cutoff = cutoff_for("alpaca", exclusions)
        fills = [{"symbol": o["symbol"], "side": o["side"], "qty": o["qty"], "price": o["filled_price"], "at": o["at"]}
                 for o in live.get("orders_all", []) if o.get("status") == "filled" and o.get("filled_price")]
        trades, lots = fifo_trades(fills)
        for t in trades:
            t["origin"] = classify(True, t["opened_at"], cutoff)
        price_by = {p["symbol"]: p["price"] for p in live.get("positions", [])}
        open_items = [{"symbol": l["symbol"], "origin": classify(True, l["opened_at"], cutoff),
                       "pnl": round((price_by.get(l["symbol"], l["entry"]) - l["entry"]) * l["qty"], 2)} for l in lots]
        for pos in live.get("positions", []):
            origins = {l["origin"] for l in open_items if l["symbol"] == pos["symbol"]}
            pos["origin"] = "excluded" if origins == {"excluded"} else ("mixed" if len(origins) > 1 else "bot")
        live["trades"] = trades
        live["bot_metrics"] = bot_metrics(trades, open_items, s.initial_cash)
        live["exclusions"] = {"reason": exclusions.get("reason", ""), "entries_before": cutoff.strftime("%Y-%m-%dT%H:%MZ") if cutoff else None}
    if live.get("available") and not live.get("stale"):
        live["at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
        os.makedirs("docs", exist_ok=True)
        with open(ALPACA_CACHE, "w", encoding="utf-8") as fh:
            json.dump(live, fh, ensure_ascii=False)
    ctrader = {"available": False}
    if os.path.exists(CTRADER_STATE):
        with open(CTRADER_STATE, encoding="utf-8") as fh:
            ctrader = json.load(fh)
    swing = {"available": False}
    if os.path.exists(SWING_STATE):
        with open(SWING_STATE, encoding="utf-8") as fh:
            swing = json.load(fh)
    if os.path.exists(SWING_WALKFORWARD):
        with open(SWING_WALKFORWARD, encoding="utf-8") as fh:
            swing["walkforward"] = json.load(fh)
    aggr = {"available": False}
    if os.path.exists(AGGR_STATE):
        with open(AGGR_STATE, encoding="utf-8") as fh:
            aggr = json.load(fh)
    if os.path.exists(AGGR_BREAKEVEN):
        with open(AGGR_BREAKEVEN, encoding="utf-8") as fh:
            aggr["breakeven_grid"] = json.load(fh)
    if os.path.exists(AGGR_SIMULATION):
        with open(AGGR_SIMULATION, encoding="utf-8") as fh:
            aggr["simulation"] = json.load(fh)
    if os.path.exists(ASIA_STATE):
        with open(ASIA_STATE, encoding="utf-8") as fh:
            aggr["asia"] = json.load(fh)
    if os.path.exists(ELLIOTT_STATE):
        with open(ELLIOTT_STATE, encoding="utf-8") as fh:
            ctrader["elliott_bot"] = json.load(fh)
    if os.path.exists(RESISTANCE_STUDY):
        with open(RESISTANCE_STUDY, encoding="utf-8") as fh:
            ctrader["resistance"] = json.load(fh)
    if os.path.exists(KRONOS_REPORT):
        with open(KRONOS_REPORT, encoding="utf-8") as fh:
            swing["kronos"] = json.load(fh)
        alt = KRONOS_REPORT.replace(".json", "_h3.json")
        if os.path.exists(alt):
            with open(alt, encoding="utf-8") as fh:
                h3 = json.load(fh)
            top = (h3.get("variants") or {}).get("kronos_tercil_superior", {})
            swing["kronos"]["horizon3"] = {"ic_spearman": (h3.get("information") or {}).get("ic_spearman"), "top_tercile_pf": top.get("profit_factor"),
                                          "top_tercile_return": top.get("return"), "verdict": h3.get("verdict")}
    if os.path.exists(ELLIOTT_STUDY):
        with open(ELLIOTT_STUDY, encoding="utf-8") as fh:
            es = json.load(fh)
        # al panel solo van las variantes con al menos 10 operaciones, ordenadas por lo que baten al azar
        rows = [r for r in es.get("variants", []) if r.get("stats", {}).get("n", 0) >= 10 and r.get("random")]
        rows.sort(key=lambda r: (r["random"]["share_random_avg_r_at_least_real"], -r["stats"]["mean_r"]))
        es["top"] = [{k: r.get(k) for k in ("key", "symbol", "timeframe", "label", "metrics", "stats", "by_year", "random", "exits", "q_all", "significant_all")} for r in rows[:15]]
        dep = next((r for r in es.get("variants", []) if r.get("key") == ELLIOTT_DEPLOYED_KEY), None)
        if dep:
            es["deployed_variant"] = {k: dep[k] for k in ("key", "timeframe", "label", "metrics", "stats", "by_year", "random")}
        es["counts"] = {"tested": es.get("tests"), "significant": sum(1 for r in es.get("variants", []) if r.get("significant_all")),
                        "beat_random": sum(1 for r in rows if r["random"]["share_random_avg_r_at_least_real"] <= 0.05),
                        "with_10_trades": len(rows),
                        "beat_random_by_tf": {tf: sum(1 for r in rows if r["timeframe"] == tf and r["random"]["share_random_avg_r_at_least_real"] <= 0.05)
                                              for tf in ("H4", "D1")}}
        es["cross_symbol"] = (es.get("cross_symbol") or [])[:10]
        es.pop("variants", None)
        swing["elliott"] = es
        if "elliott_bot" in ctrader and es.get("deployed_variant"):
            ctrader["elliott_bot"]["study"] = es["deployed_variant"]
    last_run = None
    cycles = []
    log = os.path.join(s.state_dir, "run_log.jsonl")
    if os.path.exists(log):
        with open(log, encoding="utf-8") as fh:
            lines = [ln for ln in fh if ln.strip()]
        for ln in lines:
            rec = json.loads(ln)
            if not rec.get("broker", "").startswith("alpaca"):
                continue
            if rec.get("kind") == "preopen":
                if rec.get("skipped"):
                    note = f"Validación previa a la apertura: {rec['skipped']}"
                else:
                    parts = []
                    if rec["cancelled"]:
                        parts.append("canceladas " + ", ".join(rec["cancelled"]))
                    if rec["replaced"]:
                        parts.append("recolocadas " + ", ".join(rec["replaced"]))
                    if rec["kept"]:
                        parts.append("mantenidas " + ", ".join(rec["kept"]))
                    note = "Validación previa a la apertura: " + "; ".join(parts)
                cycles.append({"at": rec["timestamp"][:16].replace("T", " "), "equity": None, "positions": "", "buys": "", "sells": "",
                               "errors": 0, "note": note, "kind": "preopen"})
                continue
            last_run = rec
            cycles.append({"at": rec["timestamp"][:16].replace("T", " "), "equity": rec["equity"], "positions": len(rec["positions"]),
                           "buys": sum(1 for o in rec["orders"] if o["side"] == "buy" and not str(o["status"]).startswith("error")),
                           "sells": sum(1 for o in rec["orders"] if o["side"] == "sell" and not str(o["status"]).startswith("error")),
                           "errors": sum(1 for o in rec["orders"] if str(o["status"]).startswith("error")),
                           "note": (rec["skipped"][0][:60] + (f" (+{len(rec['skipped']) - 1})" if len(rec["skipped"]) > 1 else "")) if rec["skipped"] else ("sin cambios" if not rec["orders"] else "")})
        cycles = cycles[-30:]
        # Curva de capital por ciclo (equity del broker en cada ejecucion del bot) + valor actual.
        # La serie diaria de Alpaca marca cada dia con el capital de apertura, es decir, el cierre del dia
        # anterior, y confundia: se conserva aparte como history_daily.
        curve = [[rec["timestamp"][:16] + "Z", round(float(rec["equity"]), 2)] for rec in map(json.loads, lines)
                 if rec.get("broker", "").startswith("alpaca") and rec.get("kind") != "preopen" and rec.get("equity")]
        if live.get("available") and len(curve) >= 2:
            live["history_daily"] = live.get("history", [])
            if not live.get("stale") and live.get("equity"):
                curve.append([datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), round(float(live["equity"]), 2)])
            live["history"] = curve
    now = datetime.now(timezone.utc)
    evs = load_events(s.events_path or None)
    active_names = {e.name for e in active_events(evs, now, s.event_hours_before, s.event_hours_after)} if s.event_mode != "off" else set()
    events = [{"name": e.name, "at": e.at.strftime("%Y-%m-%dT%H:%MZ"), "tags": list(e.tags), "active": e.name in active_names,
               "mode": e.mode or s.event_mode, "hours_before": e.hours_before if e.hours_before is not None else s.event_hours_before,
               "hours_after": e.hours_after if e.hours_after is not None else s.event_hours_after}
              for e in upcoming_events(evs, now, days=21)]
    return {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "events": events, "event_settings": {
        "mode": s.event_mode, "hours_before": s.event_hours_before, "hours_after": s.event_hours_after,
        "min_gain": s.event_min_gain, "trail_pct": s.event_trail_pct}, "settings": {
        "fast_sma": s.fast_sma, "slow_sma": s.slow_sma, "rsi_max_entry": s.rsi_max_entry, "risk_per_trade": s.risk_per_trade,
        "stop_loss_pct": s.stop_loss_pct, "max_daily_loss_pct": s.max_daily_loss_pct, "initial_cash": s.initial_cash},
        "backtests": backtests, "live": live, "last_run": last_run, "cycles": cycles, "ctrader": ctrader, "swing": swing, "aggr": aggr,
        "alpaca_variants": (json.load(open(ALPACA_VARIANTS, encoding="utf-8")) if os.path.exists(ALPACA_VARIANTS) else None)}


def render(d: dict) -> str:
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_template.html")
    with open(template_path, encoding="utf-8") as fh:
        tpl = fh.read()
    payload = json.dumps(d, ensure_ascii=False).replace("</", "<\\/")
    return tpl.replace("/*__DATA__*/null", payload).replace("__GENERATED_AT__", html.escape(d["generated_at"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/dashboard.html")
    ap.add_argument("--json", default="reports/dashboard.json")
    ap.add_argument("--no-live", action="store_true")
    args = ap.parse_args()
    d = collect(args.no_live)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(d, fh, ensure_ascii=False, indent=1)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(render(d))
    print(f"Panel generado: {args.out} ({len(d['backtests'])} backtests, live={'sí' if d['live']['available'] else 'no'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
