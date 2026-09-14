"""Validacion previa a la apertura: revisa las ordenes en cola con el precio en vivo.

Las ordenes se decidieron con el ultimo cierre diario. Antes de que se ejecuten en la apertura,
se comprueba con el precio de premercado (o el ultimo disponible) si la hipotesis sigue en pie:

- se cancela la compra si el precio en vivo cae mas de `max_gap_down` respecto al ultimo cierre
  (el mercado va en contra antes de entrar), si sube mas de `max_gap_up` (se perseguiria un hueco),
  o si la senal recalculada con el precio proyectado deja de ser BUY;
- si se mantiene y el precio se ha movido mas de un 1 %, se recoloca con el stop y el tamano
  recalculados sobre el precio proyectado.

La misma regla de hueco se aplica a las compras nuevas durante la sesion (ver bot.run_cycle).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pandas as pd

from .config import Settings
from .data import DataProvider
from .risk import RiskParams, position_size
from .strategy import StrategyParams, latest_decision


def gap_verdict(gap: float, max_gap_down: float, max_gap_up: float) -> str | None:
    """Motivo de cancelacion por hueco, o None si el hueco es aceptable."""
    if gap <= -max_gap_down:
        return f"cae {gap * 100:.2f} % antes de la apertura (limite -{max_gap_down * 100:.1f} %)"
    if gap >= max_gap_up:
        return f"sube {gap * 100:+.2f} % antes de la apertura (limite +{max_gap_up * 100:.1f} %)"
    return None


def projected_decision(df: pd.DataFrame, live_price: float, params: StrategyParams) -> dict:
    """Recalcula la senal como si la barra de hoy cerrase al precio en vivo."""
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    hist = df[df.index < today]
    row = pd.DataFrame({"open": [live_price], "high": [live_price], "low": [live_price], "close": [live_price], "volume": [0.0]}, index=[today])
    return latest_decision(pd.concat([hist, row]), params, in_position=False)


def preopen_check(settings: Settings, broker, provider: DataProvider, dry_run: bool = False) -> dict:
    strategy = StrategyParams(settings.fast_sma, settings.slow_sma, settings.rsi_period, settings.rsi_max_entry)
    risk = RiskParams(settings.risk_per_trade, settings.max_positions, settings.max_position_pct,
                      settings.max_daily_loss_pct, settings.stop_loss_pct, settings.exposure_leverage)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = {"timestamp": ts, "kind": "preopen", "broker": broker.name, "dry_run": dry_run, "checks": [], "cancelled": [], "replaced": [], "kept": []}

    if broker.is_market_open():
        summary["skipped"] = "mercado abierto: las ordenes ya se han ejecutado o se ejecutaran de inmediato"
        _log(settings.state_dir, summary)
        return summary
    pending = [o for o in broker.open_buy_orders()] if hasattr(broker, "open_buy_orders") else []
    if not pending:
        summary["skipped"] = "sin ordenes de compra en cola"
        _log(settings.state_dir, summary)
        return summary

    account = broker.account()
    for order in pending:
        symbol = order["symbol"]
        df = provider.bars(symbol)
        today = pd.Timestamp(datetime.now(timezone.utc).date())
        last_close = float(df[df.index < today]["close"].iloc[-1])
        quote = provider.quote(symbol)
        if not quote:
            summary["checks"].append({"symbol": symbol, "result": "sin cotizacion en vivo; se mantiene"})
            summary["kept"].append(symbol)
            continue
        live = quote["last"]
        gap = live / last_close - 1
        reason = gap_verdict(gap, settings.max_gap_down, settings.max_gap_up)
        proj = projected_decision(df, live, strategy)
        if reason is None and proj["action"] != "BUY":
            reason = f"senal recalculada con {live:.2f}: {proj['action']} ({proj['reason']})"
        check = {"symbol": symbol, "qty": order["qty"], "last_close": round(last_close, 2), "live": round(live, 2),
                 "live_at": quote["at"].strftime("%H:%M UTC"), "gap": round(gap, 4), "projected_rsi": round(proj.get("rsi", float("nan")), 1)}
        if reason:
            check["result"] = f"CANCELAR: {reason}"
            if not dry_run:
                broker.cancel_symbol_orders(symbol)
            summary["cancelled"].append(symbol)
        elif abs(gap) > 0.01:
            qty = position_size(account.equity, account.cash, live, risk)
            check["result"] = f"RECOLOCAR con precio {live:.2f}: {qty} uds, stop {live * (1 - settings.stop_loss_pct):.2f}"
            if not dry_run and qty > 0:
                broker.cancel_symbol_orders(symbol)
                broker.submit_market_order(symbol, qty, "buy", price_hint=live)
            summary["replaced"].append(symbol)
        else:
            check["result"] = "MANTENER"
            summary["kept"].append(symbol)
        summary["checks"].append(check)
    _log(settings.state_dir, summary)
    return summary


def _log(state_dir: str, record: dict) -> None:
    os.makedirs(state_dir, exist_ok=True)
    with open(os.path.join(state_dir, "run_log.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
