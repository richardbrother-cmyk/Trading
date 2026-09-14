"""Broker cTrader (Open API de Spotware) para cuentas demo/reales de brokers como Fusion Markets.

Usa el SDK oficial `ctrader-open-api` (protobuf sobre TCP/TLS, puerto 5035) con el reactor de
Twisted corriendo en un hilo, de modo que el resto del bot lo usa de forma sincrona.

Convenciones del protocolo (documentacion de cTrader):
- Volumenes en centesimas de unidad: 1000 => 10.00 unidades. lotSize/minVolume/stepVolume igual.
- Precios de barras y spots en 1/100000: 405012345 => 4050.12345.
- Dinero con `moneyDigits` decimales (normalmente 2): 1000000 => 10000.00.
- Stop loss de ordenes de mercado solo como `relativeStopLoss`, en 1/100000 de precio.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd

from .broker import Account, Position

PRICE_SCALE = 100_000
VOLUME_SCALE = 100
DEMO_HOST = "demo.ctraderapi.com"
LIVE_HOST = "live.ctraderapi.com"
PORT = 5035


def _lazy_sdk():
    from ctrader_open_api import Client, Protobuf, TcpProtocol  # importacion tardia: requiere twisted
    from ctrader_open_api.messages import OpenApiMessages_pb2 as msgs
    from ctrader_open_api.messages import OpenApiModelMessages_pb2 as model
    from twisted.internet import reactor, threads

    return Client, Protobuf, TcpProtocol, msgs, model, reactor, threads


# ----------------------------------------------------------------------------------------------
# Decodificadores puros (testeables sin red)
# ----------------------------------------------------------------------------------------------

def decode_trendbars(bars: list, period_minutes: int = 1440) -> pd.DataFrame:
    """Convierte ProtoOATrendbar (low + deltas) en un DataFrame OHLCV indexado por fecha."""
    rows = []
    for b in bars:
        low = b.low / PRICE_SCALE
        rows.append({
            "date": pd.Timestamp(b.utcTimestampInMinutes * 60, unit="s"),
            "open": low + b.deltaOpen / PRICE_SCALE,
            "high": low + b.deltaHigh / PRICE_SCALE,
            "low": low,
            "close": low + b.deltaClose / PRICE_SCALE,
            "volume": float(b.volume),
        })
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows).set_index("date").sort_index()
    df.index = df.index.normalize() if period_minutes >= 1440 else df.index
    return df[~df.index.duplicated(keep="last")]


def money(value: int, digits: int) -> float:
    return value / (10 ** digits)


def relative_stop(price: float, stop_loss_pct: float, digits: int) -> int:
    """Stop relativo en 1/100000 de precio, redondeado a la precision del simbolo (cTrader lo exige)."""
    tick = 10 ** max(5 - digits, 0)
    raw = price * stop_loss_pct * PRICE_SCALE
    return max(int(round(raw / tick)) * tick, tick)


def round_volume(units: float, min_volume: int, step_volume: int, max_volume: int | None = None) -> int:
    """Convierte unidades a volumen del protocolo (centesimas) respetando minimo y paso."""
    raw = int(units * VOLUME_SCALE)
    if raw < min_volume:
        return 0
    step = max(step_volume, 1)
    vol = (raw // step) * step
    if max_volume:
        vol = min(vol, max_volume)
    return vol if vol >= min_volume else 0


def cfd_market_open(now: datetime | None = None) -> bool:
    """Horario aproximado de forex/CFDs: cierra viernes 21:00 UTC y abre domingo 22:00 UTC."""
    now = now or datetime.now(timezone.utc)
    wd, hm = now.weekday(), now.hour + now.minute / 60
    if wd == 5:
        return False
    if wd == 4 and hm >= 21:
        return False
    if wd == 6 and hm < 22:
        return False
    return True


@dataclass
class SymbolInfo:
    symbol_id: int
    name: str
    digits: int = 5
    lot_size: int = 100 * VOLUME_SCALE
    min_volume: int = VOLUME_SCALE
    step_volume: int = VOLUME_SCALE
    max_volume: int = 0


@dataclass
class OpenPosition:
    position_id: int
    symbol: str
    units: float
    side: str
    price: float


# ----------------------------------------------------------------------------------------------
# Sesion sincrona sobre el SDK
# ----------------------------------------------------------------------------------------------

class CTraderSession:
    _reactor_thread: threading.Thread | None = None
    _lock = threading.Lock()

    def __init__(self, client_id: str, client_secret: str, access_token: str, account_login: int | None = None,
                 demo: bool = True, timeout: float = 15.0):
        self.client_id, self.client_secret, self.access_token = client_id, client_secret, access_token
        self.account_login = account_login
        self.demo, self.timeout = demo, timeout
        self.account_id: int | None = None
        self.symbols: dict[str, SymbolInfo] = {}
        (self._Client, self._Protobuf, self._TcpProtocol, self.msgs, self.model,
         self._reactor, self._threads) = _lazy_sdk()
        self._client = None
        self._connect()

    # -- infraestructura --------------------------------------------------------------------
    @classmethod
    def _ensure_reactor(cls, reactor):
        with cls._lock:
            if cls._reactor_thread is None or not cls._reactor_thread.is_alive():
                t = threading.Thread(target=reactor.run, kwargs={"installSignalHandlers": False}, daemon=True)
                t.start()
                cls._reactor_thread = t
                time.sleep(0.2)

    def _in_reactor(self, fn, *args, **kwargs):
        return self._threads.blockingCallFromThread(self._reactor, fn, *args, **kwargs)

    def _connect(self) -> None:
        self._ensure_reactor(self._reactor)
        host = DEMO_HOST if self.demo else LIVE_HOST
        connected = threading.Event()

        def build():
            client = self._Client(host, PORT, self._TcpProtocol)
            client.setConnectedCallback(lambda _c: connected.set())
            client.startService()
            return client

        self._client = self._in_reactor(build)
        if not connected.wait(self.timeout):
            raise TimeoutError(f"Sin conexion con {host}:{PORT} (¿puerto bloqueado por la red?)")
        self.call("ProtoOAApplicationAuthReq", clientId=self.client_id, clientSecret=self.client_secret)
        accounts = self.call("ProtoOAGetAccountListByAccessTokenReq", accessToken=self.access_token)
        chosen = None
        for acc in accounts.ctidTraderAccount:
            if self.account_login is None or int(acc.traderLogin) == int(self.account_login):
                if bool(acc.isLive) != self.demo:
                    chosen = acc
                    break
        if chosen is None:
            found = [(a.traderLogin, "live" if a.isLive else "demo") for a in accounts.ctidTraderAccount]
            raise RuntimeError(f"Cuenta {self.account_login} ({'demo' if self.demo else 'live'}) no autorizada. Disponibles: {found}")
        self.account_id = int(chosen.ctidTraderAccountId)
        self.call("ProtoOAAccountAuthReq", ctidTraderAccountId=self.account_id, accessToken=self.access_token)

    def call(self, name: str, timeout: float | None = None, **params):
        """Envia un mensaje y devuelve el payload de respuesta; lanza excepcion ante ProtoOAErrorRes."""
        msg = self._Protobuf.get(name, **params)
        response = self._in_reactor(self._client.send, msg, responseTimeoutInSeconds=timeout or self.timeout)
        payload = self._Protobuf.extract(response)
        ptype = self.model.ProtoOAPayloadType
        if response.payloadType in (ptype.PROTO_OA_ERROR_RES, ptype.PROTO_OA_ORDER_ERROR_EVENT):
            raise RuntimeError(f"cTrader {payload.errorCode}: {payload.description}")
        return payload

    def close(self) -> None:
        if self._client is not None:
            try:
                self._in_reactor(self._client.stopService)
            except Exception:  # noqa: BLE001
                pass

    # -- datos de cuenta y mercado -------------------------------------------------------------
    def load_symbols(self, names: list[str]) -> dict[str, SymbolInfo]:
        wanted = {n.upper() for n in names}
        res = self.call("ProtoOASymbolsListReq", ctidTraderAccountId=self.account_id, includeArchivedSymbols=False)
        light = {s.symbolName.upper(): s for s in res.symbol if s.symbolName.upper() in wanted}
        missing = wanted - set(light)
        if missing:
            raise ValueError(f"Simbolos no disponibles en este broker: {sorted(missing)}")
        ids = [int(s.symbolId) for s in light.values()]
        details = self.call("ProtoOASymbolByIdReq", ctidTraderAccountId=self.account_id, symbolId=ids)
        by_id = {int(s.symbolId): s for s in details.symbol}
        for name, s in light.items():
            d = by_id[int(s.symbolId)]
            self.symbols[name] = SymbolInfo(int(s.symbolId), name, int(d.digits), int(d.lotSize),
                                            int(d.minVolume), int(d.stepVolume), int(d.maxVolume))
        return self.symbols

    def all_symbol_names(self) -> list[str]:
        res = self.call("ProtoOASymbolsListReq", ctidTraderAccountId=self.account_id, includeArchivedSymbols=False)
        return sorted(s.symbolName for s in res.symbol)

    def trader(self) -> tuple[float, int, float]:
        """(balance, moneyDigits, apalancamiento)."""
        res = self.call("ProtoOATraderReq", ctidTraderAccountId=self.account_id)
        t = res.trader
        return money(int(t.balance), int(t.moneyDigits)), int(t.moneyDigits), int(t.leverageInCents) / 100

    def unrealized_pnl(self) -> float:
        try:
            res = self.call("ProtoOAGetPositionUnrealizedPnLReq", ctidTraderAccountId=self.account_id)
        except Exception:  # noqa: BLE001 - no todos los brokers lo soportan
            return 0.0
        return sum(money(int(p.netUnrealizedPnL), int(res.moneyDigits)) for p in res.positionUnrealizedPnL)

    def positions(self) -> list[OpenPosition]:
        res = self.call("ProtoOAReconcileReq", ctidTraderAccountId=self.account_id)
        id_to_name = {info.symbol_id: name for name, info in self.symbols.items()}
        out = []
        for p in res.position:
            if p.positionStatus != self.model.ProtoOAPositionStatus.POSITION_STATUS_OPEN:
                continue
            td = p.tradeData
            name = id_to_name.get(int(td.symbolId), str(td.symbolId))
            side = "buy" if td.tradeSide == self.model.ProtoOATradeSide.BUY else "sell"
            out.append(OpenPosition(int(p.positionId), name, int(td.volume) / VOLUME_SCALE, side, float(p.price)))
        return out

    def daily_bars(self, symbol: str, days: int = 400) -> pd.DataFrame:
        info = self.symbols[symbol.upper()]
        end = datetime.now(timezone.utc)
        frames = []
        chunk = 300
        for start_days in range(days, 0, -chunk):
            frm = end - timedelta(days=start_days)
            to = end - timedelta(days=max(start_days - chunk, 0))
            res = self.call("ProtoOAGetTrendbarsReq", timeout=30, ctidTraderAccountId=self.account_id,
                            fromTimestamp=int(frm.timestamp() * 1000), toTimestamp=int(to.timestamp() * 1000),
                            period=self.model.ProtoOATrendbarPeriod.D1, symbolId=info.symbol_id)
            frames.append(decode_trendbars(list(res.trendbar)))
        df = pd.concat(frames)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df.index.name = "date"
        return df

    # -- ordenes ---------------------------------------------------------------------------------
    def market_buy(self, symbol: str, units: float, stop_loss_pct: float, price_hint: float, label: str = "autotrader") -> dict:
        info = self.symbols[symbol.upper()]
        volume = round_volume(units, info.min_volume, info.step_volume, info.max_volume)
        if volume <= 0:
            raise ValueError(f"{symbol}: {units} unidades por debajo del minimo {info.min_volume / VOLUME_SCALE}")
        rel_sl = relative_stop(price_hint, stop_loss_pct, info.digits)
        res = self.call("ProtoOANewOrderReq", timeout=30, ctidTraderAccountId=self.account_id, symbolId=info.symbol_id,
                        orderType=self.model.ProtoOAOrderType.MARKET, tradeSide=self.model.ProtoOATradeSide.BUY,
                        volume=volume, relativeStopLoss=rel_sl, label=label)
        exec_type = self.model.ProtoOAExecutionType.Name(res.executionType) if hasattr(res, "executionType") else "?"
        return {"id": str(getattr(res.order, "orderId", "")), "status": exec_type.lower(), "volume": volume / VOLUME_SCALE,
                "position_id": int(getattr(res.position, "positionId", 0))}

    def close_symbol(self, symbol: str) -> list[dict]:
        out = []
        for pos in self.positions():
            if pos.symbol != symbol.upper() or pos.side != "buy":
                continue
            res = self.call("ProtoOAClosePositionReq", timeout=30, ctidTraderAccountId=self.account_id,
                            positionId=pos.position_id, volume=int(pos.units * VOLUME_SCALE))
            exec_type = self.model.ProtoOAExecutionType.Name(res.executionType) if hasattr(res, "executionType") else "?"
            out.append({"id": str(pos.position_id), "status": exec_type.lower(), "volume": pos.units})
        return out


# ----------------------------------------------------------------------------------------------
# Adaptador a la interfaz Broker del bot
# ----------------------------------------------------------------------------------------------

class CTraderBroker:
    name = "ctrader-demo"

    def __init__(self, session: CTraderSession, symbols: list[str], stop_loss_pct: float = 0.05):
        self.session = session
        self.stop_loss_pct = stop_loss_pct
        self.session.load_symbols(symbols)
        self.name = "ctrader-demo" if session.demo else "ctrader-live"

    def account(self, prices: dict[str, float] | None = None) -> Account:
        balance, _digits, _lev = self.session.trader()
        equity = balance + self.session.unrealized_pnl()
        positions: dict[str, Position] = {}
        for p in self.session.positions():
            if p.side != "buy":
                continue
            cur = positions.get(p.symbol)
            if cur:
                total = cur.qty + p.units
                cur.avg_price = (cur.avg_price * cur.qty + p.price * p.units) / total
                cur.qty = total
            else:
                positions[p.symbol] = Position(p.symbol, p.units, p.price)
        # En CFDs el "efectivo" disponible para nuevas posiciones es el equity: el margen lo limita el broker
        return Account(cash=equity, equity=equity, positions=positions)

    def is_market_open(self) -> bool:
        return cfd_market_open()

    def qty_step(self, symbol: str) -> float:
        info = self.session.symbols[symbol.upper()]
        return info.step_volume / VOLUME_SCALE

    def bars(self, symbol: str) -> pd.DataFrame:
        return self.session.daily_bars(symbol)

    def submit_market_order(self, symbol: str, qty: float, side: str, price_hint: float | None = None) -> dict:
        if side.lower() == "buy":
            if price_hint is None:
                raise ValueError("Se necesita price_hint para calcular el stop relativo")
            order = self.session.market_buy(symbol, qty, self.stop_loss_pct, price_hint)
        else:
            closed = self.session.close_symbol(symbol)
            order = {"id": ",".join(c["id"] for c in closed), "status": closed[0]["status"] if closed else "nothing_to_close"}
        order["broker"] = self.name
        return order
