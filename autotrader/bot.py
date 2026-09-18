"""Ciclo de operacion en vivo (paper): datos -> decision -> orden -> registro."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pandas as pd

from .broker import Broker, SimulatedBroker
from .config import Settings
from .data import DataProvider
from .events import active_events, effective_mode, load_events, trailed_stop
from .guard import evaluate as evaluate_guard
from .preopen import gap_verdict
from .risk import RiskParams, daily_loss_breached, position_size, stop_hit
from .strategy import StrategyParams, latest_decision


def _log_path(state_dir: str) -> str:
    os.makedirs(state_dir, exist_ok=True)
    return os.path.join(state_dir, "run_log.jsonl")


def _append(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def _day_start_equity(state_dir: str, equity_now: float) -> float:
    """Guarda el equity del primer ciclo del dia (UTC) para el limite de perdida diaria."""
    path = os.path.join(state_dir, "day_start.json")
    today = datetime.now(timezone.utc).date().isoformat()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        if raw.get("date") == today:
            return float(raw["equity"])
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"date": today, "equity": equity_now}, fh)
    return equity_now


def run_cycle(settings: Settings, broker: Broker, provider: DataProvider, dry_run: bool = False, force: bool = False) -> dict:
    """Ejecuta un ciclo completo y devuelve un resumen serializable."""
    strategy = StrategyParams(settings.fast_sma, settings.slow_sma, settings.rsi_period, settings.rsi_max_entry)
    risk = RiskParams(settings.risk_per_trade, settings.max_positions, settings.max_position_pct,
                      settings.max_daily_loss_pct, settings.stop_loss_pct, settings.exposure_leverage)
    log_path = _log_path(settings.state_dir)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    data = provider.bars_many(settings.symbols)
    prices = {s: float(df["close"].iloc[-1]) for s, df in data.items()}
    if isinstance(broker, SimulatedBroker):
        broker.mark(prices)
    account = broker.account(prices)
    day_start = _day_start_equity(settings.state_dir, account.equity)
    summary = {"timestamp": ts, "broker": broker.name, "dry_run": dry_run, "equity": account.equity,
               "cash": account.cash, "positions": {s: p.qty for s, p in account.positions.items()},
               "decisions": [], "orders": [], "skipped": []}

    market_open = force or broker.is_market_open()
    if not market_open:
        summary["skipped"].append("mercado cerrado")

    # Ventana de proteccion alrededor de eventos de alto impacto (FOMC, CPI, NFP...)
    events_now = []
    if settings.event_mode != "off":
        events_now = active_events(load_events(settings.events_path or None), datetime.now(timezone.utc),
                                   settings.event_hours_before, settings.event_hours_after)
    ev_mode = effective_mode(events_now, settings.event_mode) if events_now else settings.event_mode
    if events_now:
        summary["event_window"] = [f"{e.name} ({e.at.strftime('%Y-%m-%d %H:%M')} UTC)" for e in events_now]
        summary["event_mode"] = ev_mode
        summary["protection"] = []
    current_stops = {}
    if events_now and hasattr(broker, "current_stops"):
        try:
            current_stops = broker.current_stops()
        except Exception as exc:  # noqa: BLE001
            summary["skipped"].append(f"current_stops: {exc}")
    halted = daily_loss_breached(account.equity, day_start, risk)
    if halted:
        summary["skipped"].append(f"limite de perdida diaria alcanzado ({account.equity:.2f} vs {day_start:.2f})")
    # Freno global: interruptor manual (BOT_HALT) o drawdown acumulado desde el maximo
    guard = evaluate_guard(account.equity, settings.halt_mode, settings.max_drawdown_pct, settings.history_path(), settings.state_dir)
    summary["guard"] = guard.as_dict()
    if guard.blocks_entries:
        summary["skipped"].append(f"freno activo: {guard.reason}")

    open_slots = risk.max_positions - len(account.positions) - len(account.pending_buys)
    for symbol, df in data.items():
        pos = account.positions.get(symbol)
        if pos is None and symbol in account.pending_buys:
            summary["skipped"].append(f"{symbol}: orden de compra pendiente de ejecucion")
            continue
        decision = latest_decision(df, strategy, in_position=pos is not None)
        price = prices[symbol]
        if pos is not None and stop_hit(pos.avg_price, price, risk):
            decision = {**decision, "action": "SELL", "reason": f"stop loss ({price:.2f} <= {pos.avg_price * (1 - risk.stop_loss_pct):.2f})"}
        if pos is not None and guard.closes_positions:
            decision = {**decision, "action": "SELL", "reason": f"cierre por freno: {guard.reason}"}
        if events_now and pos is not None and decision["action"] != "SELL" and market_open:
            gain = price / pos.avg_price - 1
            if gain >= settings.event_min_gain:
                if ev_mode == "close":
                    decision = {**decision, "action": "SELL", "reason": f"cierre preventivo antes de evento (+{gain * 100:.2f} %)"}
                elif ev_mode == "trail":
                    cur = current_stops.get(symbol, pos.avg_price * (1 - risk.stop_loss_pct))
                    new = trailed_stop(cur, price, settings.event_trail_pct)
                    if new > cur + 1e-9 and not dry_run:
                        try:
                            broker.update_stop(symbol, new)
                            summary["protection"].append(f"{symbol}: stop {cur:.2f} -> {new:.2f} (+{gain * 100:.2f} %)")
                        except Exception as exc:  # noqa: BLE001
                            summary["protection"].append(f"{symbol}: error al subir stop: {exc}")
                    elif new > cur + 1e-9:
                        summary["protection"].append(f"{symbol}: subiria stop {cur:.2f} -> {new:.2f} (dry run)")
        if events_now and decision["action"] == "BUY":
            summary["skipped"].append(f"{symbol}: sin entradas nuevas durante ventana de evento")
            decision = {**decision, "action": "HOLD", "reason": "ventana de evento"}
        decision["symbol"] = symbol
        summary["decisions"].append(decision)

        if not market_open or decision["action"] == "HOLD":
            continue
        if decision["action"] == "BUY" and (halted or guard.blocks_entries):
            continue
        if decision["action"] == "SELL" and halted and not guard.closes_positions:
            continue
        if decision["action"] == "BUY":
            quote = provider.quote(symbol)
            if quote:
                # Referencia: ultimo cierre COMPLETO (la barra de hoy es parcial durante la sesion)
                today = pd.Timestamp(datetime.now(timezone.utc).date())
                completed = df[df.index < today]["close"]
                ref_close = float(completed.iloc[-1]) if len(completed) else price
                gap = quote["last"] / ref_close - 1
                verdict = gap_verdict(gap, settings.max_gap_down, settings.max_gap_up)
                if verdict:
                    summary["skipped"].append(f"{symbol}: {verdict.replace('antes de la apertura', 'hoy')}")
                    continue
                price = quote["last"]
            if open_slots <= 0:
                summary["skipped"].append(f"{symbol}: sin huecos libres (max {risk.max_positions})")
                continue
            step = broker.qty_step(symbol) if hasattr(broker, "qty_step") else 1.0
            qty = position_size(account.equity, account.cash, price, risk, step=step)
            if qty <= 0:
                summary["skipped"].append(f"{symbol}: tamano de posicion 0")
                continue
            side = "buy"
        else:
            qty = pos.qty if pos else 0
            if qty <= 0:
                continue
            side = "sell"
        if dry_run:
            summary["orders"].append({"symbol": symbol, "side": side, "qty": qty, "price": price, "status": "dry_run"})
            continue
        try:
            order = broker.submit_market_order(symbol, qty, side, price_hint=price)
            summary["orders"].append({"symbol": symbol, "side": side, "qty": qty, "price": price,
                                      "status": order.get("status"), "id": order.get("id")})
            if side == "buy":
                open_slots -= 1
                # Con apalancamiento de exposicion (CFDs) el efectivo "consumido" es el nominal / apalancamiento
                account.cash -= qty * price / risk.exposure_leverage
            else:
                account.cash += qty * price / risk.exposure_leverage
        except Exception as exc:  # noqa: BLE001
            summary["orders"].append({"symbol": symbol, "side": side, "qty": qty, "price": price, "status": f"error: {exc}"})

    # Red de seguridad: toda posicion abierta debe tener su stop vivo en el broker
    if hasattr(broker, "ensure_stops") and not dry_run:
        try:
            placed = broker.ensure_stops()
            if placed:
                summary["stops_placed"] = placed
        except Exception as exc:  # noqa: BLE001
            summary["skipped"].append(f"ensure_stops: {exc}")
    _append(log_path, summary)
    return summary
