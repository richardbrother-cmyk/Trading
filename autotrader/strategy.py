"""Estrategia: cruce de medias moviles con filtro RSI y stop loss.

Regla de entrada (largo): SMA rapida > SMA lenta, el cruce ocurrio en la ultima barra
o la posicion no existe y la tendencia sigue vigente, y RSI < rsi_max_entry.
Regla de salida: SMA rapida < SMA lenta, o el precio cae por debajo del stop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyParams:
    fast_sma: int = 20
    slow_sma: int = 50
    rsi_period: int = 14
    rsi_max_entry: float = 70.0

    def min_bars(self) -> int:
        return max(self.slow_sma, self.rsi_period) + 2


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.fillna(100.0).where(avg_loss.notna(), np.nan)


def compute_indicators(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    out = df.copy()
    out["sma_fast"] = out["close"].rolling(params.fast_sma).mean()
    out["sma_slow"] = out["close"].rolling(params.slow_sma).mean()
    out["rsi"] = rsi(out["close"], params.rsi_period)
    out["trend_up"] = out["sma_fast"] > out["sma_slow"]
    return out


def generate_signals(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    """Devuelve el DataFrame con columnas `signal` (1 largo, 0 fuera) y `event`.

    `signal` es la posicion deseada al cierre de cada barra (se ejecuta en la apertura
    siguiente en el backtest). `event` es "BUY", "SELL" o "".
    """
    ind = compute_indicators(df, params)
    trend = ind["trend_up"].fillna(False).to_numpy()
    rsi_ok = (ind["rsi"] < params.rsi_max_entry).fillna(False).to_numpy()
    n = len(ind)
    signal = np.zeros(n, dtype=int)
    event = np.array([""] * n, dtype=object)
    for i in range(1, n):
        prev = signal[i - 1]
        if prev == 0 and trend[i] and rsi_ok[i]:
            signal[i] = 1
            event[i] = "BUY"
        elif prev == 1 and not trend[i]:
            signal[i] = 0
            event[i] = "SELL"
        else:
            signal[i] = prev
    ind["signal"] = signal
    ind["event"] = event
    return ind


def latest_decision(df: pd.DataFrame, params: StrategyParams, in_position: bool) -> dict:
    """Decision para operar en vivo a partir de la ultima barra cerrada."""
    if len(df) < params.min_bars():
        return {"action": "HOLD", "reason": f"insuficientes barras ({len(df)} < {params.min_bars()})"}
    ind = compute_indicators(df, params)
    last = ind.iloc[-1]
    trend_up = bool(last["trend_up"])
    rsi_val = float(last["rsi"]) if pd.notna(last["rsi"]) else float("nan")
    info = {
        "close": float(last["close"]),
        "sma_fast": float(last["sma_fast"]),
        "sma_slow": float(last["sma_slow"]),
        "rsi": rsi_val,
        "trend_up": trend_up,
    }
    if not in_position and trend_up and rsi_val < params.rsi_max_entry:
        return {"action": "BUY", "reason": "tendencia alcista y RSI no sobrecomprado", **info}
    if in_position and not trend_up:
        return {"action": "SELL", "reason": "SMA rapida por debajo de SMA lenta", **info}
    return {"action": "HOLD", "reason": "sin cambio de regimen", **info}
