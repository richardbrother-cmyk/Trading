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
from autotrader.ctrader import CTraderSession  # noqa: E402
from autotrader.ctrader_auth import load_access_token  # noqa: E402
from autotrader.events import active_events, load_events  # noqa: E402
from autotrader.swingbot import DEFAULT_MAX_SIGNAL_AGE_HOURS, SWING_LABEL, default_max_risk_pct  # noqa: E402

MAX_HISTORY = 400
MAX_CYCLES = 60


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


def last_cycle(state_dir: str, swing: bool = False) -> dict | None:
    log = os.path.join(state_dir, "run_log.jsonl")
    if not os.path.exists(log):
        return None
    with open(log, encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip()]
    for ln in reversed(lines):
        rec = json.loads(ln)
        if swing:
            if rec.get("kind") == "swing":
                return rec
        elif rec.get("broker", "").startswith("ctrader") and rec.get("kind") not in ("preopen", "swing"):
            return rec
    return None


def swing_cycle(rec: dict) -> dict:
    """Resumen de un ciclo del bot swing para el panel."""
    ok_orders = [o for o in rec.get("orders", []) if not str(o.get("status", "")).startswith("error") and o.get("status") != "dry_run"]
    errors = sum(1 for o in rec.get("orders", []) if str(o.get("status", "")).startswith("error"))
    stale = [d for d in rec.get("decisions", []) if str(d.get("reason", "")).startswith("senal caducada")]
    if rec.get("skipped"):
        note = rec["skipped"][0][:60] + (f" (+{len(rec['skipped']) - 1})" if len(rec["skipped"]) > 1 else "")
    elif ok_orders:
        note = "compra " + ", ".join(o["symbol"] for o in ok_orders)
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
            "closed": rec.get("closed", []), "event_window": rec.get("event_window", [])}


def collect(s: Settings, swing: bool = False, initial: float | None = None) -> tuple[dict, dict | None]:
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
        for p in session.positions(only_bot=False):
            if p.side != "buy" or (swing and p.label != SWING_LABEL):
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
                pos_pnl = (px - p.price) * p.units
            positions.append({"symbol": p.symbol, "qty": p.units, "avg": p.price, "price": round(px, 6), "stop": p.stop_loss,
                              "target": p.take_profit or None, "opened_at": p.opened_at.strftime("%Y-%m-%dT%H:%MZ") if p.opened_at else None,
                              "pnl": round(pos_pnl, 2), "position_id": p.position_id, "bot": p.is_bot})
        trades = []
        try:
            for d in session.deals(days=14):
                if d["closes"]:
                    trades.append({"symbol": d["symbol"], "at": d["at"].strftime("%Y-%m-%dT%H:%MZ"), "units": d["units"], "entry": d["entry_price"],
                                   "exit": d["price"], "gross": round(d["gross"], 2), "swap": round(d["swap"], 2),
                                   "commission": round(d["close_commission"], 2), "net": d["net"], "balance_after": d["balance_after"]})
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
        "event_window": active,
    }
    if swing:
        snapshot["broker"] = "Fusion Markets · cTrader demo · swing"
        snapshot["initial"] = initial
        snapshot["settings"] = {"risk_per_trade": s.risk_per_trade,
                                "max_risk_pct": float(os.getenv("SWING_MAX_RISK_PCT", "0")) or default_max_risk_pct(s.risk_per_trade),
                                "max_signal_age_hours": float(os.getenv("SWING_MAX_SIGNAL_AGE_HOURS", str(DEFAULT_MAX_SIGNAL_AGE_HOURS))),
                                "stop_atr": 2.0, "max_hold_days": 3}
        rec = last_cycle(s.state_dir, swing=True)
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
    ap.add_argument("--initial", type=float, default=200.0, help="capital inicial de la cuenta swing, para el % desde el inicio")
    args = ap.parse_args()
    s = Settings.from_env("/dev/null")
    s.broker = "ctrader"
    s.validate()
    prev = None
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            prev = json.load(fh)
    snapshot, cycle = collect(s, swing=args.swing, initial=args.initial if args.swing else None)
    merged = merge_history(prev, snapshot, cycle)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, ensure_ascii=False, indent=1)
    print(f"Estado cTrader guardado en {args.out}: equity {merged['equity']:,.2f}, {len(merged['positions'])} posiciones, {len(merged['history'])} puntos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
