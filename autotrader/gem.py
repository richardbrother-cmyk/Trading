"""Bot "Dual Momentum" (GEM, Gary Antonacci) sobre ETFs en la cuenta paper de Alpaca.

Reglas, evaluadas con el cierre del ultimo mes completo:
  1. Momentum absoluto: retorno total a 12 meses de la renta variable USA frente a las letras del Tesoro.
     Si USA no supera a las letras -> bonos agregados (AGG).
  2. Momentum relativo: si lo supera, el mejor a 12 meses entre USA (IVV) e internacional ex-USA (VEU).
Una sola posicion, sin stop loss (la salida es la propia senal mensual), rebalanceo como mucho una vez al mes.

Convive con el bot SMA en la misma cuenta: usa simbolos que no estan en el universo del bot SMA (IVV en vez de SPY) y
el bot SMA solo pone stops a su propio universo. Toda posicion en IVV/VEU/AGG se considera de este bot, asi que esos
tres simbolos no deben operarse a mano en la cuenta paper.

El capital asignado (GEM_NOTIONAL, por defecto 20.000 USD) es una porcion fija de la cuenta paper; en cada cambio se
reinvierte el producto estimado de la venta. El estado (capital, ultimo rebalanceo, historial de senales) se guarda en
GEM_STATE (por defecto docs/gem_state.json, que el workflow publica en el repositorio porque state/ no sobrevive entre
ejecuciones de GitHub Actions).

Resultados de referencia (scripts/gem_backtest.py, Yahoo, 1997-2026, fondos indexados): CAGR 10,8 % con caida maxima del
20 % frente a 9,6 % y 51 % del S&P 500. Desde 2010 rinde menos que el S&P (9-10 % frente a 14-15 %) con la mitad de caida.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from .broker import Broker
from .config import Settings
from .data import yahoo_monthly_adjclose
from .guard import evaluate as evaluate_guard

DEFAULT_SYMBOLS = {"us": "IVV", "intl": "VEU", "bond": "AGG"}
TBILL_YIELD = "^IRX"  # letra a 13 semanas, rendimiento anualizado en %


@dataclass
class GemParams:
    lookback: int = 12  # meses del momentum
    notional: float = 20_000.0  # capital asignado al bot (USD)
    symbols: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SYMBOLS))
    state_path: str = "docs/gem_state.json"

    @classmethod
    def from_env(cls) -> "GemParams":
        syms = dict(DEFAULT_SYMBOLS)
        for key in syms:
            v = os.environ.get(f"GEM_{key.upper()}", "").strip().upper()
            if v:
                syms[key] = v
        return cls(
            lookback=int(os.environ.get("GEM_LOOKBACK", "12")),
            notional=float(os.environ.get("GEM_NOTIONAL", "20000")),
            symbols=syms,
            state_path=os.environ.get("GEM_STATE", "docs/gem_state.json"),
        )


def tbill_index(irx_monthly: pd.Series) -> pd.Series:
    """Indice de letras del Tesoro: capitaliza el rendimiento anualizado (%) mes a mes."""
    m = irx_monthly.clip(lower=0.0) / 100.0
    return (1 + m / 12.0).cumprod()


def gem_signal(us: pd.Series, intl: pd.Series, tbill: pd.Series, lookback: int = 12) -> dict:
    """Decision GEM con los cierres mensuales (ajustados) de los tres activos. Devuelve la eleccion y los momentos."""
    df = pd.concat({"us": us, "intl": intl, "tbill": tbill}, axis=1, sort=True).dropna()
    if len(df) <= lookback:
        raise ValueError(f"Hacen falta mas de {lookback} meses completos; hay {len(df)}")
    last = df.iloc[-1]
    base = df.iloc[-1 - lookback]
    mom = {k: float(last[k] / base[k] - 1) for k in ("us", "intl", "tbill")}
    if mom["us"] <= mom["tbill"]:
        choice, reason = "bond", f"momentum absoluto negativo: USA {mom['us'] * 100:.1f} % <= letras {mom['tbill'] * 100:.1f} %"
    elif mom["us"] >= mom["intl"]:
        choice, reason = "us", f"USA {mom['us'] * 100:.1f} % >= internacional {mom['intl'] * 100:.1f} %"
    else:
        choice, reason = "intl", f"internacional {mom['intl'] * 100:.1f} % > USA {mom['us'] * 100:.1f} %"
    return {"as_of": df.index[-1].date().isoformat(), "choice": choice, "reason": reason,
            "momentum": {k: round(v, 4) for k, v in mom.items()}}


def fetch_signal(params: GemParams, session=None) -> dict:
    us = yahoo_monthly_adjclose(params.symbols["us"], session=session)
    intl = yahoo_monthly_adjclose(params.symbols["intl"], session=session)
    tb = tbill_index(yahoo_monthly_adjclose(TBILL_YIELD, session=session))
    return gem_signal(us, intl, tb, params.lookback)


def load_state(path: str) -> dict:
    if path and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            pass
    return {"capital": None, "holding": None, "last_rebalance": None, "history": []}


def save_state(path: str, state: dict) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=1)
        fh.write("\n")


def run_gem(settings: Settings, broker: Broker, params: GemParams, prices: dict[str, float], signal: dict,
            dry_run: bool = False, force: bool = False) -> dict:
    """Aplica la senal GEM: vende lo que no toca, compra el activo elegido. `prices` = ultimo precio por simbolo."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    state = load_state(params.state_path)
    if state.get("capital") is None:
        state["capital"] = float(params.notional)
    account = broker.account(prices)
    inv = {v: k for k, v in params.symbols.items()}
    held = {inv[s]: p for s, p in account.positions.items() if s in inv and p.qty > 0}
    target = signal["choice"]
    target_symbol = params.symbols[target]
    summary = {"timestamp": ts, "broker": broker.name, "dry_run": dry_run, "signal": signal, "target": target_symbol,
               "holding": {params.symbols[k]: {"qty": p.qty, "avg_price": p.avg_price, "value": round(p.qty * prices.get(params.symbols[k], p.avg_price), 2)}
                           for k, p in held.items()},
               "capital": round(float(state["capital"]), 2), "orders": [], "skipped": []}
    gem_value = sum(p.qty * prices.get(params.symbols[k], p.avg_price) for k, p in held.items())
    summary["gem_equity"] = round(gem_value if held else float(state["capital"]), 2)

    guard = evaluate_guard(account.equity, settings.halt_mode, settings.max_drawdown_pct, settings.history_path(), settings.state_dir,
                           reset_at=settings.peak_reset_at)
    summary["guard"] = guard.as_dict()
    market_open = force or broker.is_market_open()
    if not market_open:
        summary["skipped"].append("mercado cerrado")

    to_sell = [k for k in held if k != target or guard.closes_positions]
    need_buy = target not in held and not guard.closes_positions
    if not to_sell and not need_buy:
        summary["skipped"].append(f"ya en {target_symbol}; sin cambios")
    pending = [s for s in account.pending_buys if s in inv]
    if pending:
        summary["skipped"].append(f"ordenes de compra pendientes: {','.join(pending)}")
        need_buy = False
    if need_buy and guard.blocks_entries:
        summary["skipped"].append(f"freno activo: {guard.reason}")
        need_buy = False

    if market_open and not pending:
        for k in to_sell:
            sym, pos = params.symbols[k], held[k]
            price = prices.get(sym, pos.avg_price)
            if dry_run:
                summary["orders"].append({"symbol": sym, "side": "sell", "qty": pos.qty, "price": price, "status": "dry_run"})
            else:
                try:
                    order = broker.submit_market_order(sym, pos.qty, "sell", price_hint=price)
                    summary["orders"].append({"symbol": sym, "side": "sell", "qty": pos.qty, "price": price, "status": order.get("status")})
                except Exception as exc:  # noqa: BLE001
                    summary["orders"].append({"symbol": sym, "side": "sell", "qty": pos.qty, "status": f"error: {exc}"})
                    need_buy = False
                    continue
            state["capital"] = round(pos.qty * price, 2)
        if need_buy:
            price = prices.get(target_symbol)
            if not price:
                summary["skipped"].append(f"{target_symbol}: sin precio")
            else:
                # Sin margen: nunca se compra por encima del efectivo disponible de la cuenta (el bot SMA usa el resto).
                budget = float(state["capital"])
                cash_after_sales = float(account.cash) + sum(held[k].qty * prices.get(params.symbols[k], held[k].avg_price) for k in to_sell)
                if cash_after_sales < budget:
                    summary["skipped"].append(f"efectivo limitado: {cash_after_sales:.2f} < capital {budget:.2f}")
                    budget = max(cash_after_sales, 0.0)
                qty = int(math.floor(budget / price))
                if qty <= 0:
                    summary["skipped"].append(f"{target_symbol}: capital insuficiente ({budget:.2f} < {price:.2f})")
                elif dry_run:
                    summary["orders"].append({"symbol": target_symbol, "side": "buy", "qty": qty, "price": price, "status": "dry_run"})
                else:
                    try:
                        order = broker.submit_market_order(target_symbol, qty, "buy", price_hint=price)
                        summary["orders"].append({"symbol": target_symbol, "side": "buy", "qty": qty, "price": price, "status": order.get("status")})
                        state["holding"] = target_symbol
                        state["last_rebalance"] = ts[:10]
                        state["capital"] = round(qty * price, 2)  # capital realmente desplegado
                    except Exception as exc:  # noqa: BLE001
                        summary["orders"].append({"symbol": target_symbol, "side": "buy", "qty": qty, "status": f"error: {exc}"})
    if not dry_run:
        if held and not to_sell:
            state["holding"] = target_symbol
        state["last_signal"] = signal
        state["last_run"] = {k: summary[k] for k in ("timestamp", "target", "orders", "skipped", "gem_equity")}
        hist = [h for h in state.get("history", []) if h.get("as_of") != signal["as_of"]]
        hist.append({"as_of": signal["as_of"], "choice": signal["choice"], "symbol": target_symbol,
                     "momentum": signal["momentum"], "gem_equity": summary["gem_equity"]})
        state["history"] = hist[-240:]
        save_state(params.state_path, state)
    summary["capital"] = round(float(state["capital"]), 2)
    return summary
