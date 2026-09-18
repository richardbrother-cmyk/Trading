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
    assert abs(len(h4) - len(df) // 16) <= 1  # el desfase de 1 h puede añadir una barra parcial
    for strat in ["pullback", "breakout", "bands"]:
        for tf in ["H1", "H4"]:
            p = SwingParams(strategy=strat, timeframe=tf)
            trades = backtest_symbol(df, "US500", p)
            for t in trades:
                assert (t.exit_time - t.entry_time).total_seconds() <= 3 * 86400 + 4 * 3600
                assert t.exit is not None
            m = metrics(trades, 10_000, 120)
            assert "trades" in m


def test_backtest_breakeven_exits_at_entry_after_gain():
    # Sube 2 R en una barra y luego cae hasta el stop original: con break even la salida es en la entrada, sin el, en el stop.
    idx = pd.date_range("2026-01-01", periods=1200, freq="15min", tz="UTC")
    close = np.full(len(idx), 100.0)
    rng = np.random.default_rng(3)
    close[:] = 100 + np.cumsum(rng.normal(0, 0.02, len(idx)))
    df = pd.DataFrame({"open": close, "high": close + 0.01, "low": close - 0.01, "close": close, "volume": 1.0}, index=idx)
    base = dict(strategy="breakout", timeframe="H1", stop_atr=1.0, tp_atr=6.0, pure_rr=True, allow_short=False, max_hold_days=3.0,
                risk_pct=0.02, max_risk_pct=0.1, commission_side=0.0, swap_daily=0.0, breakout_bars=5)
    d = resample(df, "H1")
    n = len(d)
    # forzamos una ruptura en la barra n-40, subida de 2 R en la siguiente y caida posterior
    i = n - 40
    df2 = df.copy()
    t0 = d.index[i]
    lvl = float(d["high"].iloc[i-6:i].max()) + 0.5
    sel = df2.index >= t0
    df2.loc[sel, ["open","high","low","close"]] = lvl
    nxt = df2.index >= d.index[i+2]
    df2.loc[nxt, ["open","high","low","close"]] = lvl + 0.5   # la barra siguiente a la entrada sube varios R
    later = df2.index >= d.index[i+3]
    df2.loc[later, ["open","high","low","close"]] = lvl - 1.0   # hasta muy por debajo del stop
    df2["high"] = df2[["open","close","high"]].max(axis=1); df2["low"] = df2[["open","close","low"]].min(axis=1)
    no_be = backtest_symbol(df2, "XTIUSD", SwingParams(**base), 1000.0)
    with_be = backtest_symbol(df2, "XTIUSD", SwingParams(**base, breakeven_r=1.0, breakeven_lock_r=0.0), 1000.0)
    assert not [t for t in no_be if t.reason == "break even"]
    be = [t for t in with_be if t.reason == "break even"]
    assert be, "la subida de varios R seguida de la caida debe cerrar en break even"
    for t in be:
        assert abs(t.exit - t.entry) <= 0.03 + 1e-9  # sale en la entrada (menos el spread del simbolo)
        assert t.pnl > -0.01 * 1000  # la perdida queda limitada al spread, muy por debajo del 2 % arriesgado
    # el mismo tramo sin la regla termina en el stop original o en el objetivo, nunca en la entrada
    assert all(t.reason in {"stop", "objetivo", "tiempo maximo"} for t in no_be)
