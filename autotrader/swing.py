"""Backtest swing sobre barras de 1 h y 4 h (posiciones de 1 a 3 dias) para CFDs.

Estrategias (largos y cortos):
- "pullback": tendencia por EMA50/EMA200; entrada a favor de la tendencia cuando el RSI14 vuelve
  desde sobreventa (largo) o sobrecompra (corto). Stop 1,5 ATR, objetivo 2 ATR.
- "breakout": cierre por encima del maximo de N barras (o debajo del minimo) a favor de la EMA200.
  Stop 2 ATR, salida por cierre bajo la EMA20 (largo) o sobre ella (corto).
- "bands": reversion: cierre fuera de las bandas de Bollinger (20, 2) con RSI extremo; objetivo la
  media; stop 2 ATR.

Siempre: ejecucion en la apertura de la barra siguiente, spread pagado a medias en cada lado,
comision proporcional, swap diario por nominal mantenido de un dia a otro, y cierre forzoso al
agotar `max_hold_bars`. Tamano: riesgo fijo por operacion, lote minimo del simbolo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .intraday import SPECS, ITrade, SymbolSpec

TF_RULE = {"H1": "1h", "H4": "4h"}


@dataclass(frozen=True)
class SwingParams:
    strategy: str = "pullback"
    timeframe: str = "H4"
    max_hold_days: float = 3.0
    atr_period: int = 14
    stop_atr: float = 1.5
    tp_atr: float = 2.0
    rsi_period: int = 14
    rsi_entry: float = 35.0  # pullback: RSI vuelve por encima de este nivel (largo) / 100-nivel (corto)
    breakout_bars: int = 20
    bb_period: int = 20
    bb_std: float = 2.0
    allow_short: bool = True
    risk_pct: float = 0.01
    max_risk_pct: float = 0.03
    commission_side: float = 0.000025
    swap_daily: float = 0.0001  # 0,01 % del nominal por dia mantenido
    max_notional_mult: float = 20.0

    def max_hold_bars(self) -> int:
        per_day = 24 if self.timeframe == "H1" else 6
        return int(self.max_hold_days * per_day)


def resample(df15: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    rule = TF_RULE[timeframe]
    out = df15.resample(rule, label="left", closed="left").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return out.dropna(subset=["open", "close"])


def indicators(df: pd.DataFrame, p: SwingParams) -> pd.DataFrame:
    d = df.copy()
    c = d["close"]
    d["ema20"] = c.ewm(span=20, adjust=False).mean()
    d["ema50"] = c.ewm(span=50, adjust=False).mean()
    d["ema200"] = c.ewm(span=200, adjust=False).mean()
    tr = pd.concat([d["high"] - d["low"], (d["high"] - c.shift()).abs(), (d["low"] - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / p.atr_period, adjust=False).mean()
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / p.rsi_period, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / p.rsi_period, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    d["hh"] = d["high"].rolling(p.breakout_bars).max().shift()
    d["ll"] = d["low"].rolling(p.breakout_bars).min().shift()
    m = c.rolling(p.bb_period).mean()
    s = c.rolling(p.bb_period).std()
    d["bb_mid"], d["bb_up"], d["bb_lo"] = m, m + p.bb_std * s, m - p.bb_std * s
    return d


def signal(d: pd.DataFrame, i: int, p: SwingParams) -> int:
    """+1 largo, -1 corto, 0 nada, evaluado al cierre de la barra i."""
    r = d.iloc[i]
    if np.isnan(r["ema200"]) or np.isnan(r["atr"]) or np.isnan(r["rsi"]):
        return 0
    if p.strategy == "pullback":
        prev = d.iloc[i - 1]
        if r["ema50"] > r["ema200"] and prev["rsi"] < p.rsi_entry <= r["rsi"]:
            return 1
        if p.allow_short and r["ema50"] < r["ema200"] and prev["rsi"] > 100 - p.rsi_entry >= r["rsi"]:
            return -1
        return 0
    if p.strategy == "breakout":
        if r["close"] > r["hh"] and r["close"] > r["ema200"]:
            return 1
        if p.allow_short and r["close"] < r["ll"] and r["close"] < r["ema200"]:
            return -1
        return 0
    if p.strategy == "bands":
        if r["close"] < r["bb_lo"] and r["rsi"] < 30:
            return 1
        if p.allow_short and r["close"] > r["bb_up"] and r["rsi"] > 70:
            return -1
        return 0
    raise ValueError(p.strategy)


def _size(equity: float, entry: float, stop: float, spec: SymbolSpec, p: SwingParams) -> float:
    dist = abs(entry - stop)
    if dist <= 0:
        return 0.0
    units = np.floor(equity * p.risk_pct / dist / spec.step) * spec.step
    units = min(units, np.floor(equity * p.max_notional_mult / entry / spec.step) * spec.step)
    if units < spec.min_units:
        units = spec.min_units if spec.min_units * dist <= equity * p.max_risk_pct else 0.0
    return float(units)


def backtest_symbol(df15: pd.DataFrame, sym: str, p: SwingParams, initial: float = 10_000.0) -> list[ITrade]:
    spec = SPECS[sym]
    d = indicators(resample(df15, p.timeframe), p)
    o, h, l, c = d["open"].to_numpy(), d["high"].to_numpy(), d["low"].to_numpy(), d["close"].to_numpy()
    idx = d.index
    n = len(d)
    trades: list[ITrade] = []
    equity = initial
    i = max(200, p.bb_period, p.breakout_bars) + 1
    while i < n - 1:
        side = signal(d, i, p)
        if side == 0:
            i += 1
            continue
        atr = float(d["atr"].iloc[i])
        entry = o[i + 1] + side * spec.spread / 2
        stop = entry - side * p.stop_atr * atr
        tp = entry + side * p.tp_atr * atr if p.strategy != "bands" else float(d["bb_mid"].iloc[i])
        units = _size(equity, entry, stop, spec, p)
        if units <= 0:
            i += 1
            continue
        t = ITrade(sym, side, idx[i + 1], entry, stop, units)
        t.costs = entry * units * p.commission_side
        j_end = min(n - 1, i + 1 + p.max_hold_bars())
        exit_price, reason, j_exit = None, "", None
        for j in range(i + 1, j_end + 1):
            if side == 1 and l[j] <= stop:
                exit_price, reason, j_exit = stop, "stop", j; break
            if side == -1 and h[j] >= stop:
                exit_price, reason, j_exit = stop, "stop", j; break
            if p.strategy != "bands" or True:
                if side == 1 and h[j] >= tp:
                    exit_price, reason, j_exit = tp, "objetivo", j; break
                if side == -1 and l[j] <= tp:
                    exit_price, reason, j_exit = tp, "objetivo", j; break
            if p.strategy == "breakout" and j > i + 1:
                if (side == 1 and c[j] < d["ema20"].iloc[j]) or (side == -1 and c[j] > d["ema20"].iloc[j]):
                    exit_price, reason, j_exit = c[j], "salida ema20", j; break
        if exit_price is None:
            exit_price, reason, j_exit = c[j_end], "tiempo maximo", j_end
        t.exit_time, t.reason = idx[j_exit], reason
        t.exit = exit_price - side * spec.spread / 2
        days_held = max((idx[j_exit] - idx[i + 1]).total_seconds() / 86400, 0)
        t.costs += t.exit * units * p.commission_side + entry * units * p.swap_daily * days_held
        trades.append(t)
        equity += t.pnl
        i = j_exit + 1
    return trades


def metrics(trades: list[ITrade], initial: float, days: float) -> dict:
    if not trades:
        return {"trades": 0}
    pnl = np.array([t.pnl for t in trades])
    eq = initial + np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[initial], eq]))
    dd = float(((np.concatenate([[initial], eq]) / peak) - 1).min())
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gw, gl = wins.sum(), -losses.sum()
    hold = np.mean([(t.exit_time - t.entry_time).total_seconds() / 86400 for t in trades])
    return {"trades": len(trades), "win_rate": round(len(wins) / len(trades), 3),
            "profit_factor": round(float(gw / gl), 2) if gl > 0 else float("inf"),
            "avg_r": round(float(np.mean([t.r for t in trades])), 3), "return": round(float(eq[-1] / initial - 1), 4),
            "max_drawdown": round(dd, 4), "avg_hold_days": round(float(hold), 2), "trades_per_month": round(len(trades) / days * 30, 1),
            "costs": round(float(sum(t.costs for t in trades)), 2)}
