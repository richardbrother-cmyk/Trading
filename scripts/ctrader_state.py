"""Vuelca el estado de la cuenta de cTrader a un JSON para el panel (se ejecuta en GitHub Actions).

Mantiene un historial acumulado (curva de capital y ciclos) en el propio fichero de salida.
Uso: python scripts/ctrader_state.py --out docs/ctrader_state.json
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


def last_cycle(state_dir: str) -> dict | None:
    log = os.path.join(state_dir, "run_log.jsonl")
    if not os.path.exists(log):
        return None
    with open(log, encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip()]
    for ln in reversed(lines):
        rec = json.loads(ln)
        if rec.get("broker", "").startswith("ctrader") and rec.get("kind") != "preopen":
            return rec
    return None


def collect(s: Settings) -> tuple[dict, dict | None]:
    token = load_access_token(os.path.join(s.state_dir, "ctrader_tokens.json"), s.ctrader_client_id, s.ctrader_client_secret,
                              s.ctrader_access_token, s.ctrader_refresh_token)
    session = CTraderSession(s.ctrader_client_id, s.ctrader_client_secret, token, s.ctrader_account_login or None, demo=s.ctrader_demo)
    try:
        session.load_symbols(s.symbols)
        balance, _digits, leverage = session.trader()
        pnl = session.unrealized_pnl()
        positions = []
        prices: dict[str, float] = {}
        for p in session.positions(only_bot=False):
            if p.side != "buy":
                continue
            if p.symbol not in prices:
                try:
                    prices[p.symbol] = float(session.daily_bars(p.symbol, days=10)["close"].iloc[-1])
                except Exception:  # noqa: BLE001
                    prices[p.symbol] = p.price
            px = prices[p.symbol]
            positions.append({"symbol": p.symbol, "qty": p.units, "avg": p.price, "price": px, "stop": p.stop_loss,
                              "pnl": round((px - p.price) * p.units, 2), "position_id": p.position_id, "bot": p.is_bot})
    finally:
        session.close()
    now = datetime.now(timezone.utc)
    evs = load_events(s.events_path or None)
    active = [e.name for e in active_events(evs, now, s.event_hours_before, s.event_hours_after)] if s.event_mode != "off" else []
    snapshot = {
        "available": True, "broker": "Fusion Markets · cTrader demo", "account": s.ctrader_account_login,
        "at": now.strftime("%Y-%m-%dT%H:%MZ"), "balance": round(balance, 2), "equity": round(balance + pnl, 2),
        "unrealized_pnl": round(pnl, 2), "leverage": leverage, "positions": positions, "symbols": s.symbols,
        "settings": {"stop_loss_pct": s.stop_loss_pct, "max_positions": s.max_positions, "max_position_pct": s.max_position_pct,
                     "exposure_leverage": s.exposure_leverage, "risk_per_trade": s.risk_per_trade},
        "event_window": active,
    }
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
    args = ap.parse_args()
    s = Settings.from_env("/dev/null")
    s.broker = "ctrader"
    s.validate()
    prev = None
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            prev = json.load(fh)
    snapshot, cycle = collect(s)
    merged = merge_history(prev, snapshot, cycle)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, ensure_ascii=False, indent=1)
    print(f"Estado cTrader guardado en {args.out}: equity {merged['equity']:,.2f}, {len(merged['positions'])} posiciones, {len(merged['history'])} puntos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
