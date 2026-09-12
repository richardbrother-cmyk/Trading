import numpy as np
import pandas as pd

from autotrader.data import synthetic_bars
from autotrader.strategy import StrategyParams, generate_signals, latest_decision, rsi


def _trending(n=200, up=True):
    close = np.linspace(100, 200, n) if up else np.linspace(200, 100, n)
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1e6}, index=idx)


def test_rsi_bounds():
    df = synthetic_bars("TEST", days=300, seed=1)
    r = rsi(df["close"]).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_signals_alternate_buy_sell():
    df = synthetic_bars("TEST", days=400, seed=7)
    sig = generate_signals(df, StrategyParams(10, 30, 14, 100.0))
    events = [e for e in sig["event"] if e]
    for a, b in zip(events, events[1:]):
        assert a != b, "los eventos deben alternar BUY/SELL"
    assert set(sig["signal"].unique()) <= {0, 1}


def test_latest_decision_downtrend_sells():
    d = latest_decision(_trending(up=False), StrategyParams(10, 30), in_position=True)
    assert d["action"] == "SELL"


def test_latest_decision_needs_history():
    d = latest_decision(_trending(n=20), StrategyParams(10, 30), in_position=False)
    assert d["action"] == "HOLD"
