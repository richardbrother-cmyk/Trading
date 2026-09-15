import numpy as np
import pandas as pd

from autotrader.swing import SwingParams, backtest_symbol, metrics, resample


def _m15(days=120, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-05", periods=days * 96, freq="15min", tz="UTC")
    ret = rng.normal(0.00003, 0.0012, len(idx))
    close = 7000 * np.exp(np.cumsum(ret))
    o = np.concatenate([[close[0]], close[:-1]])
    hi = np.maximum(o, close) * (1 + np.abs(rng.normal(0, 0.0004, len(idx))))
    lo = np.minimum(o, close) * (1 - np.abs(rng.normal(0, 0.0004, len(idx))))
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": close, "volume": 1.0}, index=idx)


def test_resample_and_strategies_hold_at_most_three_days():
    df = _m15()
    h4 = resample(df, "H4")
    assert len(h4) == len(df) // 16
    for strat in ["pullback", "breakout", "bands"]:
        for tf in ["H1", "H4"]:
            p = SwingParams(strategy=strat, timeframe=tf)
            trades = backtest_symbol(df, "US500", p)
            for t in trades:
                assert (t.exit_time - t.entry_time).total_seconds() <= 3 * 86400 + 4 * 3600
                assert t.exit is not None
            m = metrics(trades, 10_000, 120)
            assert "trades" in m
