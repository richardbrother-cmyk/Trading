"""Backtest intradia sobre barras de 15 minutos (CFDs de cTrader).

Estrategias:
- "orb": ruptura del rango de apertura. Rango = primeras `or_bars` barras de la sesion. Entrada al
  cierre de la primera barra que rompe el rango (a favor de la EMA20), ejecutada en la apertura de
  la siguiente. Stop en el extremo opuesto del rango, acotado a `max_stop_pct`. Objetivo a `rr`
  veces el riesgo; el stop pasa a punto de equilibrio al alcanzar 1R. Cierre forzoso al final de
  la sesion. Una operacion por simbolo y dia.
- "ema": cruce EMA9/EMA21 dentro de la sesion; stop `max_stop_pct`; salida en cruce contrario o al
  cierre de la sesion.

Costes: spread (se paga la mitad en cada lado) y comision proporcional al nominal por lado.
Tamano: riesgo fijo por operacion sobre el equity, redondeado al paso minimo del simbolo; si el
minimo obliga a arriesgar mas de `max_risk_pct`, la operacion se descarta (importante con 200 USD).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SymbolSpec:
    name: str
    session_start: str  # "HH:MM" UTC
    session_end: str
    spread: float  # en unidades de precio
    step: float  # paso minimo de volumen en unidades
    min_units: float
    digits: int = 2


# Horario de verano (UTC). Indices y materias primas: sesion de contado de EE. UU. Forex: Londres.
SPECS: dict[str, SymbolSpec] = {
    "US500": SymbolSpec("US500", "13:30", "20:00", 0.4, 0.01, 0.01, 2),
    "NAS100": SymbolSpec("NAS100", "13:30", "20:00", 1.0, 0.01, 0.01, 2),
    "GER40": SymbolSpec("GER40", "07:00", "15:30", 1.0, 0.01, 0.01, 2),
    "XAUUSD": SymbolSpec("XAUUSD", "13:30", "20:00", 0.20, 1, 1, 2),
    "XTIUSD": SymbolSpec("XTIUSD", "13:30", "20:00", 0.03, 10, 10, 3),
    "EURUSD": SymbolSpec("EURUSD", "07:00", "16:00", 0.00006, 1000, 1000, 5),
    "GBPUSD": SymbolSpec("GBPUSD", "07:00", "16:00", 0.00009, 1000, 1000, 5),
}


@dataclass(frozen=True)
class IntradayParams:
    strategy: str = "orb"
    or_bars: int = 2
    entry_window_bars: int = 12  # hasta 3 h tras la apertura
    max_stop_pct: float = 0.004
    rr: float = 2.0
    breakeven_at_r: float = 1.0
    allow_short: bool = True
    ema_fast: int = 9
    ema_slow: int = 21
    risk_pct: float = 0.01
    max_risk_pct: float = 0.03  # si el lote minimo arriesga mas, no se opera
    commission_side: float = 0.000025  # 0,0025 % del nominal por lado
    max_notional_mult: float = 20.0  # exposicion maxima como multiplo del equity


@dataclass
class ITrade:
    symbol: str
    side: int  # +1 largo, -1 corto
    entry_time: pd.Timestamp
    entry: float
    stop: float
    units: float
    exit_time: pd.Timestamp | None = None
    exit: float | None = None
    reason: str = ""
    costs: float = 0.0

    @property
    def pnl(self) -> float:
        if self.exit is None:
            return 0.0
        return (self.exit - self.entry) * self.units * self.side - self.costs

    @property
    def r(self) -> float:
        risk = abs(self.entry - self.stop) * self.units
        return self.pnl / risk if risk > 0 else 0.0


@dataclass
class IResult:
    equity: pd.Series
    trades: list[ITrade] = field(default_factory=list)
    initial: float = 0.0
    skipped_unsizeable: int = 0

    def metrics(self) -> dict:
        closed = [t for t in self.trades if t.exit is not None]
        eq = self.equity
        wins = [t for t in closed if t.pnl > 0]
        losses = [t for t in closed if t.pnl <= 0]
        gw = sum(t.pnl for t in wins)
        gl = -sum(t.pnl for t in losses)
        dd = (eq / eq.cummax() - 1.0).min() if len(eq) else 0.0
        days = max((eq.index[-1] - eq.index[0]).days, 1) if len(eq) > 1 else 1
        return {
            "initial": self.initial, "final": round(float(eq.iloc[-1]), 2) if len(eq) else self.initial,
            "return": round(float(eq.iloc[-1] / self.initial - 1), 4) if len(eq) else 0.0,
            "max_drawdown": round(float(dd), 4), "trades": len(closed), "trades_per_week": round(len(closed) / days * 7, 2),
            "win_rate": round(len(wins) / len(closed), 3) if closed else 0.0,
            "profit_factor": round(gw / gl, 2) if gl > 0 else (float("inf") if gw > 0 else 0.0),
            "avg_r": round(float(np.mean([t.r for t in closed])), 3) if closed else 0.0,
            "expectancy_usd": round(float(np.mean([t.pnl for t in closed])), 2) if closed else 0.0,
            "total_costs": round(sum(t.costs for t in closed), 2), "skipped_unsizeable": self.skipped_unsizeable,
        }


def load_bars(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"]).set_index("time").sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def _session_slices(df: pd.DataFrame, spec: SymbolSpec):
    t = df.index
    mins = t.hour * 60 + t.minute
    s = int(spec.session_start[:2]) * 60 + int(spec.session_start[3:])
    e = int(spec.session_end[:2]) * 60 + int(spec.session_end[3:])
    inside = df[(mins >= s) & (mins < e)]
    for day, g in inside.groupby(inside.index.date):
        if g.index[0].weekday() >= 5 or len(g) < 6:
            continue
        yield day, g


def size_units(equity: float, entry: float, stop: float, spec: SymbolSpec, p: IntradayParams) -> float:
    dist = abs(entry - stop)
    if dist <= 0:
        return 0.0
    units = equity * p.risk_pct / dist
    units = np.floor(units / spec.step) * spec.step
    units = min(units, np.floor(equity * p.max_notional_mult / entry / spec.step) * spec.step)
    if units < spec.min_units:
        # el minimo arriesga mas de lo permitido?
        if spec.min_units * dist <= equity * p.max_risk_pct:
            units = spec.min_units
        else:
            return 0.0
    return float(round(units, 6))


def backtest_intraday(data: dict[str, pd.DataFrame], p: IntradayParams, initial: float = 10_000.0) -> IResult:
    equity = initial
    curve: list[tuple[pd.Timestamp, float]] = []
    trades: list[ITrade] = []
    skipped = 0
    # eventos ordenados por dia entre simbolos: para simplificar, simulamos simbolo a simbolo por dia
    days: dict = {}
    for sym, df in data.items():
        spec = SPECS[sym]
        for day, g in _session_slices(df, spec):
            days.setdefault(day, []).append((sym, spec, g))
    for day in sorted(days):
        for sym, spec, g in days[day]:
            t, sk = _simulate_day(sym, spec, g, p, equity)
            skipped += sk
            if t is not None:
                trades.append(t)
                equity += t.pnl
        curve.append((pd.Timestamp(day), equity))
    eq = pd.Series(dict(curve)).sort_index() if curve else pd.Series([initial], index=[pd.Timestamp("1970-01-01")])
    return IResult(eq, trades, initial, skipped)


def _simulate_day(sym: str, spec: SymbolSpec, g: pd.DataFrame, p: IntradayParams, equity: float):
    o, h, l, c = g["open"].to_numpy(), g["high"].to_numpy(), g["low"].to_numpy(), g["close"].to_numpy()
    idx = g.index
    n = len(g)
    if p.strategy == "orb":
        ema = g["close"].ewm(span=20, adjust=False).mean().to_numpy()
        or_hi, or_lo = h[: p.or_bars].max(), l[: p.or_bars].min()
        for i in range(p.or_bars, min(n - 1, p.or_bars + p.entry_window_bars)):
            side = 0
            if c[i] > or_hi and c[i] > ema[i]:
                side = 1
            elif p.allow_short and c[i] < or_lo and c[i] < ema[i]:
                side = -1
            if side == 0:
                continue
            entry = o[i + 1] + side * spec.spread / 2
            raw_stop = or_lo if side == 1 else or_hi
            cap_stop = entry * (1 - side * p.max_stop_pct)
            stop = max(raw_stop, cap_stop) if side == 1 else min(raw_stop, cap_stop)
            dist = abs(entry - stop)
            if dist <= 0:
                return None, 0
            units = size_units(equity, entry, stop, spec, p)
            if units <= 0:
                return None, 1
            tp = entry + side * p.rr * dist
            trade = ITrade(sym, side, idx[i + 1], entry, stop, units)
            trade.costs = entry * units * p.commission_side
            be_done = False
            for j in range(i + 1, n):
                if side == 1:
                    if l[j] <= stop:
                        return _close(trade, idx[j], stop, "stop", spec, p), 0
                    if h[j] >= tp:
                        return _close(trade, idx[j], tp, "objetivo", spec, p), 0
                    if not be_done and h[j] >= entry + p.breakeven_at_r * dist:
                        stop, be_done = entry, True
                else:
                    if h[j] >= stop:
                        return _close(trade, idx[j], stop, "stop", spec, p), 0
                    if l[j] <= tp:
                        return _close(trade, idx[j], tp, "objetivo", spec, p), 0
                    if not be_done and l[j] <= entry - p.breakeven_at_r * dist:
                        stop, be_done = entry, True
            return _close(trade, idx[n - 1], c[n - 1], "cierre de sesion", spec, p), 0
        return None, 0
    if p.strategy == "ema":
        ef = g["close"].ewm(span=p.ema_fast, adjust=False).mean().to_numpy()
        es = g["close"].ewm(span=p.ema_slow, adjust=False).mean().to_numpy()
        trade = None
        stop = 0.0
        for i in range(p.ema_slow, n - 1):
            cross_up = ef[i] > es[i] and ef[i - 1] <= es[i - 1]
            cross_dn = ef[i] < es[i] and ef[i - 1] >= es[i - 1]
            if trade is None:
                side = 1 if cross_up else (-1 if (cross_dn and p.allow_short) else 0)
                if side == 0:
                    continue
                entry = o[i + 1] + side * spec.spread / 2
                stop = entry * (1 - side * p.max_stop_pct)
                units = size_units(equity, entry, stop, spec, p)
                if units <= 0:
                    return None, 1
                trade = ITrade(sym, side, idx[i + 1], entry, stop, units)
                trade.costs = entry * units * p.commission_side
                continue
            j = i + 1
            if trade.side == 1 and l[j] <= stop:
                return _close(trade, idx[j], stop, "stop", spec, p), 0
            if trade.side == -1 and h[j] >= stop:
                return _close(trade, idx[j], stop, "stop", spec, p), 0
            if (trade.side == 1 and cross_dn) or (trade.side == -1 and cross_up):
                return _close(trade, idx[j], o[j], "cruce contrario", spec, p), 0
        if trade is not None:
            return _close(trade, idx[n - 1], c[n - 1], "cierre de sesion", spec, p), 0
        return None, 0
    raise ValueError(p.strategy)


def _close(trade: ITrade, when, price: float, reason: str, spec: SymbolSpec, p: IntradayParams) -> ITrade:
    trade.exit_time, trade.reason = when, reason
    trade.exit = price - trade.side * spec.spread / 2
    trade.costs += trade.exit * trade.units * p.commission_side
    return trade
