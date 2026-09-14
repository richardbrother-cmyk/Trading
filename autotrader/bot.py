"""Ciclo de operacion en vivo (paper): datos -> decision -> orden -> registro."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from .broker import Broker, SimulatedBroker
from .config import Settings
from .data import DataProvider
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
    halted = daily_loss_breached(account.equity, day_start, risk)
    if halted:
        summary["skipped"].append(f"limite de perdida diaria alcanzado ({account.equity:.2f} vs {day_start:.2f})")

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
        decision["symbol"] = symbol
        summary["decisions"].append(decision)

        if not market_open or halted or decision["action"] == "HOLD":
            continue
        if decision["action"] == "BUY":
            quote = provider.quote(symbol)
            if quote:
                gap = quote["last"] / price - 1
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
                account.cash -= qty * price
            else:
                account.cash += qty * price
        except Exception as exc:  # noqa: BLE001
            summary["orders"].append({"symbol": symbol, "side": side, "qty": qty, "price": price, "status": f"error: {exc}"})

    _append(log_path, summary)
    return summary
