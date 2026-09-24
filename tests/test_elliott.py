import numpy as np
import pandas as pd

from autotrader.elliott import ElliottParams, Pivot, backtest_symbol, impulse_setup, resample, zigzag


def _bars_from_path(path, bars_per_leg=4, start="2024-01-01 01:00", freq="4h"):
    """Serie M15 sintetica que recorre en linea recta los puntos de `path` (una pierna por tramo)."""
    closes = []
    for a, b in zip(path[:-1], path[1:]):
        closes += list(np.linspace(a, b, bars_per_leg, endpoint=False))
    closes.append(path[-1])
    n = len(closes)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    c = np.array(closes, dtype=float)
    o = np.concatenate([[c[0]], c[:-1]])
    df = pd.DataFrame({"open": o, "high": np.maximum(o, c) + 0.5, "low": np.minimum(o, c) - 0.5, "close": c, "volume": 1.0}, index=idx)
    # a M15 para que `resample` la vuelva a agregar
    m15 = df.resample("15min").ffill()
    m15["volume"] = 1.0
    return m15


def test_zigzag_is_causal_and_alternates():
    high = np.array([10, 12, 15, 13, 11, 14, 18, 16, 12, 9, 13], dtype=float)
    low = high - 1
    thr = np.full(len(high), 3.0)
    pv = zigzag(high, low, thr)
    assert [p.kind for p in pv] == [-1, 1, -1, 1, -1]
    for a, b in zip(pv[:-1], pv[1:]):
        assert a.kind != b.kind and a.idx < b.idx
    # cada pivote se conoce despues de su barra, nunca antes
    assert all(p.confirmed > p.idx for p in pv)
    hi = pv[1]
    assert hi.idx == 2 and hi.price == 15.0 and hi.confirmed == 3  # 15 - 12 = 3 = umbral
    assert pv[3].idx == 6 and pv[3].price == 18.0


def test_impulse_setup_applies_hard_rules():
    p = ElliottParams(waves=(2, 4))
    ok = [Pivot(0, 100, -1, 1), Pivot(5, 110, 1, 6), Pivot(8, 105, -1, 9)]  # onda 2 retrocede el 50 %
    s = impulse_setup(ok, p)
    assert s and s["side"] == 1 and s["wave"] == 2 and s["trigger"] == 110 and s["invalidation"] == 100
    too_deep = [Pivot(0, 100, -1, 1), Pivot(5, 110, 1, 6), Pivot(8, 99, -1, 9)]  # mas del 100 %: invalido
    assert impulse_setup(too_deep, p) is None
    too_shallow = [Pivot(0, 100, -1, 1), Pivot(5, 110, 1, 6), Pivot(8, 109, -1, 9)]  # 10 %: fuera de la banda
    assert impulse_setup(too_shallow, p) is None
    w4 = ok + [Pivot(12, 125, 1, 13), Pivot(15, 118, -1, 16)]  # onda 3 = 20 >= onda 1; onda 4 retrocede el 35 % y no solapa
    s4 = impulse_setup(w4, p)
    assert s4 and s4["wave"] == 4 and s4["trigger"] == 125 and s4["invalidation"] == 110
    overlap = ok + [Pivot(12, 125, 1, 13), Pivot(15, 108, -1, 16)]  # la onda 4 entra en territorio de la onda 1
    assert impulse_setup(overlap, ElliottParams(waves=(4,))) is None
    short_w3 = ok + [Pivot(12, 112, 1, 13), Pivot(15, 109, -1, 16)]  # onda 3 mas corta que la 1
    assert impulse_setup(short_w3, ElliottParams(waves=(4,))) is None
    # con la onda 2 permitida, los tres ultimos pivotes forman por si solos un 1-2 valido (impulso de menor grado)
    assert impulse_setup(short_w3, p)["wave"] == 2
    bearish = [Pivot(0, 100, 1, 1), Pivot(5, 90, -1, 6), Pivot(8, 95, 1, 9)]
    assert impulse_setup(bearish, p)["side"] == -1
    assert impulse_setup(bearish, ElliottParams(allow_short=False)) is None


def test_backtest_enters_after_wave_two_and_hits_target():
    # onda 1 de 2000 a 2100, onda 2 hasta 2050, onda 3 hasta 2250
    m15 = _bars_from_path([2000, 2100, 2050, 2250, 2200], bars_per_leg=6)
    p = ElliottParams(timeframe="H4", zz_atr=0, zz_pct=0.015, waves=(2,), entry="ruptura", target_ext=1.0, stop="onda",
                      allow_short=False, atr_period=3)
    trades = backtest_symbol(m15, "XAUUSD", p, 10_000)
    assert len(trades) == 1
    t = trades[0]
    assert t.side == 1 and t.wave == 2 and t.reason == "objetivo"
    assert t.entry > 2100 and t.stop < 2050 and abs(t.target - 2150) < 1.0
    assert t.r > 0 and t.exit_time >= t.entry_time


def test_backtest_pivot_entry_and_stop():
    # tras la onda 2 el precio se desploma: la entrada "pivote" salta el stop
    m15 = _bars_from_path([2000, 2100, 2050, 2090, 1900], bars_per_leg=6)
    p = ElliottParams(timeframe="H4", zz_atr=0, zz_pct=0.015, waves=(2,), entry="pivote", target_ext=1.618, stop="inicio",
                      allow_short=False, atr_period=3)
    trades = backtest_symbol(m15, "XAUUSD", p, 10_000)
    assert len(trades) == 1 and trades[0].reason == "stop" and trades[0].r < 0
    assert trades[0].stop < 2000  # stop de invalidacion bajo el inicio de la onda 1


def test_resample_daily_drops_thin_days():
    idx = pd.date_range("2024-03-04 00:00", periods=4 * 96, freq="15min", tz="UTC")
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)
    d1 = resample(df, "D1")
    assert len(d1) >= 3 and all((d1.index.hour == 22))
