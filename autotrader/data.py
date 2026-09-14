"""Proveedores de datos de mercado (barras diarias OHLCV)."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

COLUMNS = ["open", "high", "low", "close", "volume"]


def _validate(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"Sin datos para {symbol}")
    df = df[COLUMNS].astype(float).dropna()
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df.sort_index()


def synthetic_bars(symbol: str, days: int = 500, seed: int | None = None, start_price: float = 100.0) -> pd.DataFrame:
    """Serie sintetica (GBM con cambios de regimen) para pruebas sin red."""
    rng = np.random.default_rng(seed if seed is not None else abs(hash(symbol)) % (2**32))
    regimes = rng.choice([0.0008, -0.0004, 0.0002], size=days, p=[0.45, 0.25, 0.30])
    vol = rng.uniform(0.008, 0.02, size=days)
    returns = rng.normal(regimes, vol)
    close = start_price * np.exp(np.cumsum(returns))
    open_ = np.concatenate([[start_price], close[:-1]]) * (1 + rng.normal(0, 0.002, size=days))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, size=days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, size=days)))
    volume = rng.integers(1_000_000, 5_000_000, size=days).astype(float)
    end = datetime.now(timezone.utc).date()
    index = pd.bdate_range(end=end, periods=days)
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index)
    return _validate(df, symbol)


def yahoo_bars(symbol: str, range_: str = "2y", interval: str = "1d", session: requests.Session | None = None) -> pd.DataFrame:
    """Barras historicas desde el endpoint publico de Yahoo Finance (sin claves)."""
    sess = session or requests.Session()
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"range": range_, "interval": interval, "events": "div,splits"}
    headers = {"User-Agent": "Mozilla/5.0 (autotrader)"}
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = sess.get(url, params=params, headers=headers, timeout=20)
            resp.raise_for_status()
            payload = resp.json()
            break
        except (requests.RequestException, ValueError) as exc:
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    else:
        raise RuntimeError(f"Yahoo no respondio para {symbol}: {last_err}")
    result = payload.get("chart", {}).get("result")
    if not result:
        raise ValueError(f"Yahoo sin resultados para {symbol}: {payload.get('chart', {}).get('error')}")
    node = result[0]
    quote = node["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "volume": quote["volume"],
        },
        index=pd.to_datetime(node["timestamp"], unit="s", utc=True).tz_convert(None).normalize(),
    )
    return _validate(df, symbol)


def yahoo_quote(symbol: str, session: requests.Session | None = None) -> dict:
    """Ultimo precio disponible de Yahoo incluyendo pre y post mercado (velas de 1 minuto de hoy).

    Devuelve {"last": precio, "at": datetime UTC, "prev_close": ultimo cierre regular segun Yahoo}.
    """
    sess = session or requests.Session()
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"range": "1d", "interval": "1m", "includePrePost": "true"}
    resp = sess.get(url, params=params, headers={"User-Agent": "Mozilla/5.0 (autotrader)"}, timeout=20)
    resp.raise_for_status()
    result = resp.json().get("chart", {}).get("result")
    if not result:
        raise ValueError(f"Yahoo sin cotizacion para {symbol}")
    node = result[0]
    closes = node["indicators"]["quote"][0].get("close") or []
    stamps = node.get("timestamp") or []
    pairs = [(t, c) for t, c in zip(stamps, closes) if c is not None]
    if not pairs:
        raise ValueError(f"Yahoo sin velas de hoy para {symbol}")
    t, c = pairs[-1]
    return {"last": float(c), "at": datetime.fromtimestamp(t, tz=timezone.utc), "prev_close": node["meta"].get("regularMarketPrice")}


def alpaca_bars(symbol: str, api_key: str, secret_key: str, days: int = 500, session: requests.Session | None = None) -> pd.DataFrame:
    """Barras diarias desde la API de datos de Alpaca (feed IEX, gratuito)."""
    sess = session or requests.Session()
    start = (datetime.now(timezone.utc) - timedelta(days=int(days * 1.6))).date().isoformat()
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}
    rows: list[dict] = []
    page_token = None
    while True:
        params = {"timeframe": "1Day", "start": start, "limit": 1000, "adjustment": "all", "feed": "iex"}
        if page_token:
            params["page_token"] = page_token
        resp = sess.get(url, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        rows.extend(payload.get("bars") or [])
        page_token = payload.get("next_page_token")
        if not page_token:
            break
    if not rows:
        raise ValueError(f"Alpaca sin barras para {symbol}")
    df = pd.DataFrame(rows).rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None).dt.normalize()
    return _validate(df.set_index("date"), symbol)


class DataProvider:
    """Fachada que elige el proveedor segun la configuracion."""

    def __init__(self, provider: str, api_key: str = "", secret_key: str = "", days: int = 500, bars_fn=None):
        self.provider = provider
        self.api_key = api_key
        self.secret_key = secret_key
        self.days = days
        self.bars_fn = bars_fn  # p.ej. CTraderBroker.bars
        self.session = requests.Session()

    def bars(self, symbol: str) -> pd.DataFrame:
        if self.provider == "ctrader":
            if self.bars_fn is None:
                raise ValueError("DATA_PROVIDER=ctrader requiere un broker cTrader")
            return _validate(self.bars_fn(symbol), symbol)
        if self.provider == "synthetic":
            return synthetic_bars(symbol, days=self.days)
        if self.provider == "yahoo":
            return yahoo_bars(symbol, session=self.session)
        if self.provider == "alpaca":
            return alpaca_bars(symbol, self.api_key, self.secret_key, days=self.days, session=self.session)
        raise ValueError(f"Proveedor desconocido: {self.provider}")

    def quote(self, symbol: str) -> dict | None:
        """Precio en vivo si el proveedor lo ofrece; None en caso contrario."""
        if self.provider == "yahoo":
            try:
                return yahoo_quote(symbol, session=self.session)
            except Exception as exc:  # noqa: BLE001
                print(f"[quote] {symbol}: {exc}")
        return None

    def bars_many(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                out[symbol] = self.bars(symbol)
            except Exception as exc:  # noqa: BLE001 - un simbolo caido no debe tumbar el bot
                print(f"[data] {symbol}: {exc}")
        return out
