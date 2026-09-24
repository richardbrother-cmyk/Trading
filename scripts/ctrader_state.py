"""Vuelca el estado de la cuenta de cTrader a un JSON para el panel (se ejecuta en GitHub Actions).

Mantiene un historial acumulado (curva de capital y ciclos) en el propio fichero de salida.
Uso: python scripts/ctrader_state.py --out docs/ctrader_state.json
     python scripts/ctrader_state.py --swing --out docs/swing_state.json   (cuenta del bot swing)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autotrader.config import Settings  # noqa: E402
from autotrader.guard import detect_cash_flow, withdrawal_status  # noqa: E402
from autotrader.ctrader import CTraderSession  # noqa: E402
from autotrader.ctrader_auth import load_access_token  # noqa: E402
from autotrader.events import active_events, load_events  # noqa: E402
from autotrader.swingbot import DEFAULT_MAX_SIGNAL_AGE_HOURS, SWING_LABEL, default_max_risk_pct, describe, params_from_env  # noqa: E402
from autotrader.attribution import bot_metrics, classify, cutoff_for, load_exclusions  # noqa: E402
from autotrader.ctrader import BOT_LABEL  # noqa: E402

MAX_HISTORY = 400
MAX_CYCLES = 60
MAX_SLIPPAGE = 200


def merge_history(prev: dict | None, snapshot: dict, cycle: dict | None) -> dict:
    """Combina el estado nuevo con el historial previo (curva de capital y ciclos)."""
    prev = prev or {}
    history = list(prev.get("history", []))
    history.append([snapshot["at"], round(snapshot["equity"], 2)])
    history = history[-MAX_HISTORY:]
    cycles = list(prev.get("cycles", []))
    if cycle:
        cycles.append(cycle)
    cycles = cycles[-MAX_CYCLES:]
    return {**snapshot, "history": history, "cycles": cycles}


def realized_slippage(cycle: dict | None, positions: list[dict]) -> list[dict]:
    """Compara el precio de la senal (cierre de la barra) con el precio real de entrada de la posicion abierta en este ciclo."""
    out = []
    if not cycle:
        return out
    by_symbol = {p["symbol"]: p for p in positions}
    for o in cycle.get("orders", []):
        st = str(o.get("status", ""))
        pos = by_symbol.get(o["symbol"])
        if st.startswith("error") or st == "dry_run" or not pos or not o.get("entry"):
            continue
        units = float(o.get("units") or pos["qty"])
        diff = float(pos["avg"]) - float(o["entry"])
        out.append({"symbol": o["symbol"], "at": cycle["at"], "bar": o.get("bar"), "bar_age_h": o.get("bar_age_h"),
                    "signal": o["entry"], "fill": pos["avg"], "units": units, "slip_price": round(diff, 6),
                    "slip_bps": round(diff / float(o["entry"]) * 1e4, 2), "slip_usd": round(diff * units, 2)})
    return out


def merge_slippage(prev: dict | None, new: list[dict]) -> dict:
    items = list((prev or {}).get("slippage", {}).get("items", [])) + new
    items = items[-MAX_SLIPPAGE:]
    usd = round(sum(i["slip_usd"] for i in items), 2)
    bps = round(sum(i["slip_bps"] for i in items) / len(items), 2) if items else 0.0
    return {"items": items, "total_usd": usd, "avg_bps": bps, "count": len(items)}


def last_cycle(state_dir: str, swing: bool = False, label: str = SWING_LABEL) -> dict | None:
    log = os.path.join(state_dir, "run_log.jsonl")
    if not os.path.exists(log):
        return None
    with open(log, encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip()]
    for ln in reversed(lines):
        rec = json.loads(ln)
        if swing:
            if rec.get("kind") == "swing" and rec.get("label", SWING_LABEL) == label:
                return rec
        elif rec.get("broker", "").startswith("ctrader") and rec.get("kind") in (None, "run"):
            # solo los ciclos del bot tendencial: los de swing, asia o elliott llevan su propio "kind"
            return rec
    return None


def swing_cycle(rec: dict) -> dict:
    """Resumen de un ciclo del bot swing para el panel."""
    ok_orders = [o for o in rec.get("orders", []) if not str(o.get("status", "")).startswith("error") and o.get("status") != "dry_run"]
    errors = sum(1 for o in rec.get("orders", []) if str(o.get("status", "")).startswith("error"))
    stale = [d for d in rec.get("decisions", []) if str(d.get("reason", "")).startswith("senal caducada")]
    guard = rec.get("guard") or {}
    moved = [m for m in rec.get("stops_moved", []) if m.get("status") == "amended"]
    if guard.get("mode") and guard["mode"] != "off":
        note = f"freno {guard['mode']}: {guard.get('reason', '')}"[:80]
    elif rec.get("skipped"):
        note = rec["skipped"][0][:60] + (f" (+{len(rec['skipped']) - 1})" if len(rec["skipped"]) > 1 else "")
    elif ok_orders:
        note = "compra " + ", ".join(o["symbol"] for o in ok_orders)
    elif moved:
        note = "stop a break even: " + ", ".join(m["symbol"] for m in moved)
    elif stale:
        note = "señal caducada en " + ", ".join(d["symbol"] for d in stale)
    else:
        note = "sin cambios"
    return {"at": rec["timestamp"][:16].replace("T", " "), "equity": rec.get("equity"), "positions": len(rec.get("positions", {})),
            "buys": len(ok_orders), "sells": len(rec.get("closed", [])), "errors": errors, "note": note,
            "decisions": [{"symbol": d["symbol"], "action": d["action"], "rsi": d.get("rsi"), "close": d.get("close"), "bb_lo": d.get("bb_lo"),
                           "bb_mid": d.get("bb_mid"), "bar": d.get("bar"), "bar_age_h": d.get("bar_age_h"), "reason": d.get("reason")}
                          for d in rec.get("decisions", [])],
            "orders": [{"symbol": o["symbol"], "units": o.get("units"), "entry": o.get("entry_ref"), "stop": o.get("stop"), "target": o.get("target"),
                        "risk_usd": o.get("risk_usd"), "status": o.get("status")} for o in rec.get("orders", [])],
            "stops_moved": [{"symbol": m["symbol"], "entry": m.get("entry"), "old_stop": m.get("old_stop"), "new_stop": m.get("new_stop"),
                             "gained_r": m.get("gained_r"), "status": m.get("status")} for m in rec.get("stops_moved", [])],
            "closed": rec.get("closed", []), "event_window": rec.get("event_window", []), "guard": rec.get("guard")}


def collect(s: Settings, swing: bool = False, initial: float | None = None, label: str = SWING_LABEL, prev: dict | None = None,
            also_label: str = "") -> tuple[dict, dict | None]:
    exclusions = load_exclusions()
    cutoff = cutoff_for("ctrader", exclusions)
    token = load_access_token(os.path.join(s.state_dir, "ctrader_tokens.json"), s.ctrader_client_id, s.ctrader_client_secret,
                              s.ctrader_access_token, s.ctrader_refresh_token)
    session = CTraderSession(s.ctrader_client_id, s.ctrader_client_secret, token, s.ctrader_account_login or None, demo=s.ctrader_demo)
    try:
        session.load_symbols(s.symbols)
        balance, _digits, leverage = session.trader()
        by_pos = session.position_pnl()
        pnl = sum(by_pos.values())
        positions = []
        prices: dict[str, float] = {}
        labels = {x.strip() for x in str(label).split(",") if x.strip()}
        bot_labels = {BOT_LABEL} | ({x.strip() for x in str(also_label).split(",") if x.strip()} if also_label else set())
        for p in session.positions(only_bot=False):
            if swing and p.label not in labels:
                continue
            if not swing and p.side != "buy":
                continue
            if p.position_id in by_pos and p.units > 0:
                # precio implicito en el resultado neto del servidor: mas fiel que el ultimo cierre diario
                pos_pnl = by_pos[p.position_id]
                px = p.price + pos_pnl / p.units
            else:
                if p.symbol not in prices:
                    try:
                        prices[p.symbol] = float(session.daily_bars(p.symbol, days=10)["close"].iloc[-1])
                    except Exception:  # noqa: BLE001
                        prices[p.symbol] = p.price
                px = prices[p.symbol]
                pos_pnl = (px - p.price) * p.units * (1 if p.side == "buy" else -1)
            is_own = (p.label in labels) if swing else (p.label in bot_labels)
            positions.append({"symbol": p.symbol, "qty": p.units, "side": p.side, "avg": p.price, "price": round(px, 6), "stop": p.stop_loss,
                              "target": p.take_profit or None, "opened_at": p.opened_at.strftime("%Y-%m-%dT%H:%MZ") if p.opened_at else None,
                              "pnl": round(pos_pnl, 2), "position_id": p.position_id, "bot": is_own, "label": p.label,
                              "origin": classify(is_own, p.opened_at, cutoff)})
        trades = []
        try:
            for d in session.deals(days=14):
                if d["closes"]:
                    is_bot = (d.get("label") in labels) if swing else (d.get("label") in bot_labels)
                    trades.append({"symbol": d["symbol"], "at": d["at"].strftime("%Y-%m-%dT%H:%MZ"), "units": d["units"], "entry": d["entry_price"],
                                   "exit": d["price"], "gross": round(d["gross"], 2), "swap": round(d["swap"], 2),
                                   "commission": round(d["close_commission"], 2), "net": d["net"], "balance_after": d["balance_after"],
                                   "label": d.get("label", ""), "opened_at": d["opened_at"].strftime("%Y-%m-%dT%H:%MZ") if d.get("opened_at") else None,
                                   "origin": classify(is_bot, d.get("opened_at"), cutoff)})
        except Exception as exc:  # noqa: BLE001
            trades = [{"error": str(exc)[:120]}]
    finally:
        session.close()
    now = datetime.now(timezone.utc)
    evs = load_events(s.events_path or None)
    active = [e.name for e in active_events(evs, now, s.event_hours_before, s.event_hours_after)] if s.event_mode != "off" else []
    snapshot = {
        "available": True, "broker": "Fusion Markets · cTrader demo", "account": s.ctrader_account_login,
        "at": now.strftime("%Y-%m-%dT%H:%MZ"), "balance": round(balance, 2), "equity": round(balance + pnl, 2),
        "unrealized_pnl": round(pnl, 2), "leverage": leverage, "positions": positions, "symbols": s.symbols, "trades": trades,
        "settings": {"stop_loss_pct": s.stop_loss_pct, "max_positions": s.max_positions, "max_position_pct": s.max_position_pct,
                     "exposure_leverage": s.exposure_leverage, "risk_per_trade": s.risk_per_trade},
        "event_window": active, "initial": initial,
        "bot_metrics": bot_metrics([t for t in trades if "error" not in t], positions, initial or 0.0),
        "exclusions": {"reason": exclusions.get("reason", ""), "entries_before": cutoff.strftime("%Y-%m-%dT%H:%MZ") if cutoff else None},
    }
    if swing:
        snapshot["broker"] = "Fusion Markets · cTrader demo · swing"
        snapshot["initial"] = initial
        prm = params_from_env(s)
        snapshot["settings"] = {"risk_per_trade": s.risk_per_trade,
                                "max_risk_pct": float(os.getenv("SWING_MAX_RISK_PCT", "0")) or default_max_risk_pct(s.risk_per_trade),
                                "max_signal_age_hours": float(os.getenv("SWING_MAX_SIGNAL_AGE_HOURS", str(DEFAULT_MAX_SIGNAL_AGE_HOURS))),
                                "stop_atr": prm.stop_atr, "tp_atr": prm.tp_atr if (prm.pure_rr or prm.strategy != "bands") else None,
                                "strategy": prm.strategy, "description": describe(prm), "max_hold_days": prm.max_hold_days,
                                "max_positions": int(os.getenv("SWING_MAX_POSITIONS", "0")), "label": label,
                                "breakeven_r": prm.breakeven_r, "breakeven_lock_r": prm.breakeven_lock_r}
        trigger = float(os.getenv("WITHDRAW_TRIGGER_PCT", "0") or 0)
        if trigger > 0:
            withdraw_pct = float(os.getenv("WITHDRAW_PCT", "0.30") or 0.30)
            # Retiros y depositos por conciliacion: entre dos lecturas, lo que el saldo cambia y no explican las operaciones
            # cerradas es un movimiento de caja (la consulta directa del historial de caja no responde en este broker).
            # El ultimo retiro conocido y la base se conservan en el estado publicado.
            prev = prev or {}
            prev_wd = prev.get("withdrawal") or {}
            base, last_at = float(initial or 0), ""
            if prev_wd.get("base"):
                base, last_at = float(prev_wd["base"]), str(prev_wd.get("last_at") or "")
            flows = list(prev.get("cash_flows") or [])
            prev_at = str(prev.get("at") or "")
            if prev.get("balance") is not None and prev_at and not any("error" in t for t in trades):
                net_since = sum(float(t["net"]) for t in trades if str(t["at"]) > prev_at)
                flow = detect_cash_flow(float(prev["balance"]), balance, net_since)
                if flow < 0:
                    base, last_at = round(balance, 2), snapshot["at"]
                    flows.append({"at": snapshot["at"], "type": "withdraw", "delta": flow, "balance_after": round(balance, 2)})
                elif flow > 0:
                    base = round(base + flow, 2)  # un deposito sube la base en la misma cantidad
                    flows.append({"at": snapshot["at"], "type": "deposit", "delta": flow, "balance_after": round(balance, 2)})
            snapshot["cash_flows"] = flows[-20:]
            snapshot["withdrawal"] = withdrawal_status(snapshot["equity"], base, trigger, withdraw_pct, last_at)
        rec = last_cycle(s.state_dir, swing=True, label=str(label).split(",")[0].strip())
        cycle = swing_cycle(rec) if rec else None
        if cycle:
            snapshot["last_run"] = cycle
        return snapshot, cycle
    rec = last_cycle(s.state_dir)
    cycle = None
    if rec:
        cycle = {"at": rec["timestamp"][:16].replace("T", " "), "equity": rec.get("equity"), "positions": len(rec.get("positions", {})),
                 "buys": sum(1 for o in rec.get("orders", []) if o["side"] == "buy" and not str(o["status"]).startswith("error")),
                 "sells": sum(1 for o in rec.get("orders", []) if o["side"] == "sell" and not str(o["status"]).startswith("error")),
                 "errors": sum(1 for o in rec.get("orders", []) if str(o["status"]).startswith("error")),
                 "note": (rec["skipped"][0][:60] + (f" (+{len(rec['skipped']) - 1})" if len(rec["skipped"]) > 1 else "")) if rec.get("skipped")
                 else ("sin cambios" if not rec.get("orders") else ""),
                 "decisions": [{"symbol": d["symbol"], "action": d["action"], "rsi": d.get("rsi"), "trend_up": d.get("trend_up"), "close": d.get("close")}
                               for d in rec.get("decisions", [])],
                 "protection": rec.get("protection", [])}
        snapshot["last_run"] = cycle
    return snapshot, cycle


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/ctrader_state.json")
    ap.add_argument("--swing", action="store_true", help="cuenta del bot swing: solo posiciones con su etiqueta y su ultimo ciclo")
    ap.add_argument("--initial", type=float, default=None, help="capital inicial de la cuenta (10000 tendencial, 200 swing)")
    ap.add_argument("--label", default=os.getenv("SWING_LABEL", SWING_LABEL), help="etiqueta de las posiciones del bot swing/agresivo")
    ap.add_argument("--also-label", default="autotrader-elliott", help="cuenta tendencial: otras etiquetas de bots propios (separadas por comas)")
    args = ap.parse_args()
    s = Settings.from_env("/dev/null")
    s.broker = "ctrader"
    s.validate()
    prev = None
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            prev = json.load(fh)
    initial = args.initial if args.initial is not None else (200.0 if args.swing else 10_000.0)
    snapshot, cycle = collect(s, swing=args.swing, initial=initial, label=args.label, prev=prev, also_label=args.also_label)
    if args.swing:
        snapshot["slippage"] = merge_slippage(prev, realized_slippage(cycle, snapshot["positions"]))
        snapshot["guard"] = (cycle or {}).get("guard")
        snapshot["settings"]["max_drawdown_pct"] = s.max_drawdown_pct
        snapshot["settings"]["halt_mode"] = s.halt_mode
    merged = merge_history(prev, snapshot, cycle)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, ensure_ascii=False, indent=1)
    print(f"Estado cTrader guardado en {args.out}: equity {merged['equity']:,.2f}, {len(merged['positions'])} posiciones, {len(merged['history'])} puntos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
