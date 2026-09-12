"""Reglas de gestion de riesgo: tamano de posicion y limites globales."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RiskParams:
    risk_per_trade: float = 0.02
    max_positions: int = 5
    max_position_pct: float = 0.25
    max_daily_loss_pct: float = 0.03
    stop_loss_pct: float = 0.05


def position_size(equity: float, cash: float, price: float, params: RiskParams) -> int:
    """Numero entero de acciones a comprar.

    Riesgo por operacion = equity * risk_per_trade, con stop a stop_loss_pct del precio,
    acotado por max_position_pct del equity y por el efectivo disponible.
    """
    if price <= 0 or equity <= 0 or cash <= 0:
        return 0
    risk_dollars = equity * params.risk_per_trade
    per_share_risk = price * params.stop_loss_pct
    qty_by_risk = risk_dollars / per_share_risk
    qty_by_cap = (equity * params.max_position_pct) / price
    qty_by_cash = cash / price
    qty = math.floor(min(qty_by_risk, qty_by_cap, qty_by_cash))
    return max(qty, 0)


def daily_loss_breached(equity_now: float, equity_day_start: float, params: RiskParams) -> bool:
    if equity_day_start <= 0:
        return False
    return (equity_now - equity_day_start) / equity_day_start <= -params.max_daily_loss_pct


def stop_hit(entry_price: float, current_price: float, params: RiskParams) -> bool:
    return current_price <= entry_price * (1.0 - params.stop_loss_pct)
