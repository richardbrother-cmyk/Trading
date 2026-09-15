"""Brokers: simulador local (paper local) y Alpaca (solo cuenta paper)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Protocol

import requests

from .config import PAPER_URL


@dataclass
class Position:
    symbol: str
    qty: int
    avg_price: float


@dataclass
class Account:
    cash: float
    equity: float
    positions: dict[str, Position] = field(default_factory=dict)
    pending_buys: set[str] = field(default_factory=set)  # simbolos con ordenes de compra abiertas


class Broker(Protocol):
    name: str

    def account(self, prices: dict[str, float] | None = None) -> Account: ...
    def submit_market_order(self, symbol: str, qty: int, side: str, price_hint: float | None = None) -> dict: ...
    def is_market_open(self) -> bool: ...


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SimulatedBroker:
    """Cuenta simulada persistida en disco. No requiere claves ni red."""

    name = "sim"

    def __init__(self, initial_cash: float = 100_000.0, state_dir: str = "state"):
        self.state_dir = state_dir
        self.path = os.path.join(state_dir, "sim_account.json")
        self.orders_path = os.path.join(state_dir, "sim_orders.jsonl")
        os.makedirs(state_dir, exist_ok=True)
        self.cash = initial_cash
        self.positions: dict[str, Position] = {}
        self.last_prices: dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, encoding="utf-8") as fh:
            raw = json.load(fh)
        self.cash = float(raw["cash"])
        self.positions = {s: Position(**p) for s, p in raw.get("positions", {}).items()}
        self.last_prices = {k: float(v) for k, v in raw.get("last_prices", {}).items()}

    def _save(self) -> None:
        raw = {
            "cash": self.cash,
            "positions": {s: asdict(p) for s, p in self.positions.items()},
            "last_prices": self.last_prices,
            "updated_at": _now(),
        }
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(raw, fh, indent=2)

    def mark(self, prices: dict[str, float]) -> None:
        self.last_prices.update(prices)
        self._save()

    def account(self, prices: dict[str, float] | None = None) -> Account:
        if prices:
            self.last_prices.update(prices)
        equity = self.cash + sum(p.qty * self.last_prices.get(s, p.avg_price) for s, p in self.positions.items())
        return Account(cash=self.cash, equity=equity, positions=dict(self.positions))

    def is_market_open(self) -> bool:
        return True

    def submit_market_order(self, symbol: str, qty: int, side: str, price_hint: float | None = None) -> dict:
        if qty <= 0:
            raise ValueError("qty debe ser > 0")
        price = price_hint if price_hint is not None else self.last_prices.get(symbol)
        if price is None:
            raise ValueError(f"Sin precio para {symbol}")
        side = side.lower()
        if side == "buy":
            cost = qty * price
            if cost > self.cash + 1e-9:
                raise ValueError(f"Efectivo insuficiente para comprar {qty} {symbol} ({cost:.2f} > {self.cash:.2f})")
            pos = self.positions.get(symbol)
            if pos:
                total = pos.qty + qty
                pos.avg_price = (pos.avg_price * pos.qty + price * qty) / total
                pos.qty = total
            else:
                self.positions[symbol] = Position(symbol, qty, price)
            self.cash -= cost
        elif side == "sell":
            pos = self.positions.get(symbol)
            if not pos or pos.qty < qty:
                raise ValueError(f"No hay suficiente posicion en {symbol} para vender {qty}")
            pos.qty -= qty
            self.cash += qty * price
            if pos.qty == 0:
                self.positions.pop(symbol)
        else:
            raise ValueError("side debe ser buy o sell")
        self.last_prices[symbol] = price
        order = {"id": f"sim-{int(datetime.now(timezone.utc).timestamp() * 1000)}", "symbol": symbol, "qty": qty,
                 "side": side, "filled_avg_price": price, "status": "filled", "submitted_at": _now(), "broker": self.name}
        with open(self.orders_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(order) + "\n")
        self._save()
        return order


class AlpacaBroker:
    """Cliente minimo de la API de trading de Alpaca. Rechaza cualquier URL que no sea paper."""

    name = "alpaca-paper"

    def __init__(self, api_key: str, secret_key: str, base_url: str = PAPER_URL, session: requests.Session | None = None,
                 stop_loss_pct: float | None = 0.05):
        if base_url.rstrip("/") != PAPER_URL:
            raise ValueError(f"AlpacaBroker solo acepta {PAPER_URL} (paper trading)")
        self.base_url = PAPER_URL
        self.stop_loss_pct = stop_loss_pct  # None desactiva el stop en el broker
        self.session = session or requests.Session()
        self.session.headers.update({"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key})

    def _get(self, path: str, **params):
        resp = self.session.get(f"{self.base_url}{path}", params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, payload: dict):
        resp = self.session.post(f"{self.base_url}{path}", json=payload, timeout=20)
        if resp.status_code >= 400:
            raise RuntimeError(f"Alpaca {resp.status_code}: {resp.text}")
        return resp.json()

    def _delete(self, path: str):
        resp = self.session.delete(f"{self.base_url}{path}", timeout=20)
        if resp.status_code >= 400 and resp.status_code != 404:
            raise RuntimeError(f"Alpaca {resp.status_code}: {resp.text}")
        return resp

    def open_buy_orders(self) -> list[dict]:
        """Ordenes de compra pendientes (sin los tramos de stop)."""
        return [{"id": o["id"], "symbol": o["symbol"], "qty": int(float(o["qty"])), "status": o["status"]}
                for o in self._get("/v2/orders", status="open", nested="false")
                if o.get("side") == "buy" and o.get("type") == "market"]

    def current_stops(self) -> dict[str, float]:
        """Precio del stop de venta abierto por simbolo."""
        return {o["symbol"]: float(o["stop_price"]) for o in self._get("/v2/orders", status="open", nested="false")
                if o.get("side") == "sell" and o.get("type") == "stop" and o.get("stop_price")}

    def update_stop(self, symbol: str, stop_price: float) -> dict:
        """Sube el stop de una posicion (modifica la orden abierta o crea una GTC si no existe)."""
        stop_price = round(stop_price, 2)
        for o in self._get("/v2/orders", status="open", nested="false"):
            if o["symbol"] == symbol and o.get("side") == "sell" and o.get("type") == "stop":
                resp = self.session.patch(f"{self.base_url}/v2/orders/{o['id']}", json={"stop_price": f"{stop_price:.2f}"}, timeout=20)
                if resp.status_code >= 400:
                    raise RuntimeError(f"Alpaca {resp.status_code}: {resp.text}")
                return {"symbol": symbol, "stop_price": stop_price, "status": "replaced"}
        for p in self._get("/v2/positions"):
            if p["symbol"] == symbol:
                order = self._post("/v2/orders", {"symbol": symbol, "qty": p["qty"], "side": "sell", "type": "stop",
                                                  "stop_price": f"{stop_price:.2f}", "time_in_force": "gtc"})
                return {"symbol": symbol, "stop_price": stop_price, "status": order.get("status")}
        raise ValueError(f"Sin posicion en {symbol}")

    def ensure_stops(self) -> list[dict]:
        """Coloca un stop GTC para toda posicion larga que no tenga ya una orden de venta stop abierta."""
        if not self.stop_loss_pct:
            return []
        open_stops = {o["symbol"] for o in self._get("/v2/orders", status="open", nested="false")
                      if o.get("side") == "sell" and o.get("type") == "stop"}
        placed = []
        for p in self._get("/v2/positions"):
            qty = int(float(p["qty"]))
            if p["symbol"] in open_stops or qty <= 0:
                continue
            stop_price = round(float(p["avg_entry_price"]) * (1 - self.stop_loss_pct), 2)
            order = self._post("/v2/orders", {"symbol": p["symbol"], "qty": str(qty), "side": "sell", "type": "stop",
                                              "stop_price": f"{stop_price:.2f}", "time_in_force": "gtc"})
            placed.append({"symbol": p["symbol"], "qty": qty, "stop_price": stop_price, "status": order.get("status")})
        return placed

    def cancel_symbol_orders(self, symbol: str) -> int:
        """Cancela las ordenes abiertas de un simbolo (p.ej. el stop vinculado antes de vender por senal)."""
        n = 0
        for o in self._get("/v2/orders", status="open", symbols=symbol, nested="false"):
            if o["symbol"] == symbol:
                self._delete(f"/v2/orders/{o['id']}")
                n += 1
        return n

    def account(self, prices: dict[str, float] | None = None) -> Account:
        acct = self._get("/v2/account")
        positions = {
            p["symbol"]: Position(p["symbol"], int(float(p["qty"])), float(p["avg_entry_price"]))
            for p in self._get("/v2/positions")
        }
        pending = {o["symbol"] for o in self._get("/v2/orders", status="open", nested="false") if o.get("side") == "buy"}
        return Account(cash=float(acct["cash"]), equity=float(acct["equity"]), positions=positions, pending_buys=pending)

    def is_market_open(self) -> bool:
        return bool(self._get("/v2/clock")["is_open"])

    def submit_market_order(self, symbol: str, qty: int, side: str, price_hint: float | None = None) -> dict:
        """Orden de mercado. Las compras llevan un stop loss vinculado (OTO) que el broker activa al ejecutarse.

        Antes de vender se cancelan las ordenes abiertas del simbolo, porque el stop vinculado
        retiene las acciones y la venta seria rechazada por cantidad insuficiente.
        """
        if qty <= 0:
            raise ValueError("qty debe ser > 0")
        side = side.lower()
        payload = {"symbol": symbol, "qty": str(qty), "side": side, "type": "market", "time_in_force": "day"}
        if side == "buy" and self.stop_loss_pct and price_hint:
            # El tramo de stop hereda la vigencia del padre: con "day" caducaria al cierre y la posicion
            # quedaria sin proteccion al dia siguiente. Con "gtc" el stop sobrevive hasta ejecutarse o cancelarse.
            stop_price = round(price_hint * (1 - self.stop_loss_pct), 2)
            payload.update({"order_class": "oto", "time_in_force": "gtc", "stop_loss": {"stop_price": f"{stop_price:.2f}"}})
        elif side == "sell":
            self.cancel_symbol_orders(symbol)
        order = self._post("/v2/orders", payload)
        order["broker"] = self.name
        return order


def build_broker(settings) -> Broker:
    if settings.broker == "alpaca":
        return AlpacaBroker(settings.alpaca_api_key, settings.alpaca_secret_key, settings.alpaca_base_url,
                            stop_loss_pct=settings.stop_loss_pct)
    if settings.broker == "ctrader":
        from .ctrader import CTraderBroker, CTraderSession
        from .ctrader_auth import load_access_token

        token = load_access_token(os.path.join(settings.state_dir, "ctrader_tokens.json"), settings.ctrader_client_id,
                                  settings.ctrader_client_secret, settings.ctrader_access_token, settings.ctrader_refresh_token)
        session = CTraderSession(settings.ctrader_client_id, settings.ctrader_client_secret, token,
                                 settings.ctrader_account_login or None, demo=settings.ctrader_demo)
        return CTraderBroker(session, settings.symbols, stop_loss_pct=settings.stop_loss_pct)
    return SimulatedBroker(initial_cash=settings.initial_cash, state_dir=settings.state_dir)
