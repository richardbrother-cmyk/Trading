"""Backtester de cartera (multi-simbolo) con ejecucion en la apertura siguiente."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .risk import RiskParams, position_size, stop_hit
from .strategy import StrategyParams, generate_signals


@dataclass
class Trade:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    qty: int
    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    reason: str = ""

    @property
    def pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) * self.qty

    @property
    def ret(self) -> float:
        if self.exit_price is None:
            return 0.0
        return self.exit_price / self.entry_price - 1.0


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: list[Trade] = field(default_factory=list)
    initial_cash: float = 0.0
    commission_per_share: float = 0.0

    def metrics(self) -> dict:
        eq = self.equity.dropna()
        if eq.empty:
            return {}
        rets = eq.pct_change().dropna()
        total_return = eq.iloc[-1] / self.initial_cash - 1.0
        years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
        cagr = (eq.iloc[-1] / self.initial_cash) ** (1.0 / years) - 1.0 if years > 0.05 else total_return
        sharpe = float(np.sqrt(252) * rets.mean() / rets.std()) if rets.std() > 0 else 0.0
        drawdown = eq / eq.cummax() - 1.0
        closed = [t for t in self.trades if t.exit_price is not None]
        wins = [t for t in closed if t.pnl > 0]
        losses = [t for t in closed if t.pnl <= 0]
        gross_win = sum(t.pnl for t in wins)
        gross_loss = -sum(t.pnl for t in losses)
        return {
            "initial_cash": round(self.initial_cash, 2),
            "final_equity": round(float(eq.iloc[-1]), 2),
            "total_return": round(float(total_return), 4),
            "cagr": round(float(cagr), 4),
            "sharpe": round(sharpe, 2),
            "max_drawdown": round(float(drawdown.min()), 4),
            "trades": len(closed),
            "win_rate": round(len(wins) / len(closed), 3) if closed else 0.0,
            "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0,
            "avg_trade_return": round(float(np.mean([t.ret for t in closed])), 4) if closed else 0.0,
        }


def run_backtest(
    data: dict[str, pd.DataFrame],
    strategy: StrategyParams,
    risk: RiskParams,
    initial_cash: float = 100_000.0,
    commission_per_share: float = 0.0,
    slippage_bps: float = 2.0,
    entry_allowed=None,
    size_fn=None,
    drop_exit_pct: float = 0.0,
    drop_exit_only_in_profit: bool = False,
    regime_exit: bool = False,
    reentry_cooldown_days: int = 0,
) -> BacktestResult:
    """Simula la estrategia sobre varias series alineadas por fecha.

    Las senales se calculan al cierre de la barra t y se ejecutan a la apertura de t+1,
    con deslizamiento (slippage) en puntos basicos. Los stops se evaluan sobre el minimo
    intradia de cada barra.

    Ganchos opcionales de investigacion (por defecto no cambian nada):
    - `entry_allowed(date) -> bool`: filtro de regimen; si es False no se abren posiciones nuevas ese dia.
      Con `regime_exit=True` ademas se cierran todas las posiciones cuando el filtro pasa a False.
    - `size_fn(symbol, date, equity, cash, price) -> qty`: sustituye a `position_size` (p.ej. tamano por volatilidad).
    - `drop_exit_pct`: cierra un largo en la apertura siguiente si el cierre cae mas de ese porcentaje respecto al
      cierre anterior (regla de proteccion de beneficios); con `drop_exit_only_in_profit` solo si la posicion gana.
    - `reentry_cooldown_days`: tras una salida por caida diaria o por stop, no se reentra en ese simbolo hasta pasados
      N dias de calendario salvo que haya un cruce alcista nuevo (evento BUY).
    """
    signals = {s: generate_signals(df, strategy) for s, df in data.items() if len(df) >= strategy.min_bars()}
    if not signals:
        raise ValueError("Ningun simbolo tiene barras suficientes para el backtest")
    calendar = sorted(set().union(*[set(df.index) for df in signals.values()]))
    slip = slippage_bps / 10_000.0

    cash = initial_cash
    positions: dict[str, Trade] = {}
    pending: dict[str, str] = {}  # ordenes a ejecutar en la apertura siguiente
    equity_curve: list[tuple[pd.Timestamp, float]] = []
    trades: list[Trade] = []
    cooldown_until: dict[str, pd.Timestamp] = {}

    for date in calendar:
        # 1) Ejecutar ordenes pendientes a la apertura
        equity_open = cash + sum(
            t.qty * float(signals[s].at[date, "open"]) if date in signals[s].index else t.qty * t.entry_price
            for s, t in positions.items()
        )
        for symbol, action in list(pending.items()):
            df = signals[symbol]
            if date not in df.index:
                continue
            open_px = float(df.at[date, "open"])
            if action.startswith("SELL") and symbol in positions:
                trade = positions.pop(symbol)
                px = open_px * (1 - slip)
                cash += trade.qty * px - commission_per_share * trade.qty
                trade.exit_date, trade.exit_price, trade.reason = date, px, (action.split(":", 1)[1] if ":" in action else "signal")
                trades.append(trade)
                if trade.reason == "drop" and reentry_cooldown_days > 0:
                    cooldown_until[symbol] = date + pd.Timedelta(days=reentry_cooldown_days)
            elif action == "BUY" and symbol not in positions and len(positions) < risk.max_positions:
                px = open_px * (1 + slip)
                qty = size_fn(symbol, date, equity_open, cash, px) if size_fn else position_size(equity_open, cash, px, risk)
                if qty > 0:
                    cash -= qty * px + commission_per_share * qty
                    positions[symbol] = Trade(symbol, date, px, qty)
            pending.pop(symbol, None)

        # 2) Evaluar stops intradia
        for symbol, trade in list(positions.items()):
            df = signals[symbol]
            if date not in df.index:
                continue
            low = float(df.at[date, "low"])
            if stop_hit(trade.entry_price, low, risk):
                px = trade.entry_price * (1 - risk.stop_loss_pct) * (1 - slip)
                cash += trade.qty * px - commission_per_share * trade.qty
                trade.exit_date, trade.exit_price, trade.reason = date, px, "stop"
                trades.append(trade)
                positions.pop(symbol)
                pending.pop(symbol, None)
                if reentry_cooldown_days > 0:
                    cooldown_until[symbol] = date + pd.Timedelta(days=reentry_cooldown_days)

        # 3) Nuevas senales al cierre
        allowed = entry_allowed(date) if entry_allowed is not None else True
        for symbol, df in signals.items():
            if date not in df.index:
                continue
            event = df.at[date, "event"]
            if event == "BUY" and symbol not in positions:
                if allowed:
                    pending[symbol] = "BUY"
            elif event == "SELL" and symbol in positions:
                pending[symbol] = "SELL"
            elif symbol not in positions and int(df.at[date, "signal"]) == 1 and symbol not in pending:
                # La tendencia sigue vigente pero no tenemos posicion (p.ej. por stop): reentrar
                if allowed and date >= cooldown_until.get(symbol, date):
                    pending[symbol] = "BUY"
            if symbol in positions and pending.get(symbol) != "SELL":
                if regime_exit and not allowed:
                    pending[symbol] = "SELL:regime"
                elif drop_exit_pct > 0:
                    i = df.index.get_loc(date)
                    if i > 0:
                        close, prev_close = float(df.at[date, "close"]), float(df["close"].iloc[i - 1])
                        in_profit = close > positions[symbol].entry_price
                        if close < prev_close * (1 - drop_exit_pct) and (in_profit or not drop_exit_only_in_profit):
                            pending[symbol] = "SELL:drop"

        # 4) Valorar cartera al cierre
        equity_close = cash + sum(
            t.qty * float(signals[s].at[date, "close"]) if date in signals[s].index else t.qty * t.entry_price
            for s, t in positions.items()
        )
        equity_curve.append((date, equity_close))

    # Cerrar posiciones abiertas al ultimo cierre para las metricas
    last_date = calendar[-1]
    for symbol, trade in positions.items():
        df = signals[symbol]
        px = float(df["close"].iloc[-1])
        trade.exit_date, trade.exit_price, trade.reason = last_date, px, "open_at_end"
        trades.append(trade)

    equity = pd.Series(dict(equity_curve)).sort_index()
    equity.index.name = "date"
    return BacktestResult(equity=equity, trades=trades, initial_cash=initial_cash, commission_per_share=commission_per_share)
