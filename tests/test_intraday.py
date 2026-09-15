import numpy as np
import pandas as pd

from autotrader.intraday import SPECS, IntradayParams, backtest_intraday, size_units


def _m15(days=15, seed=3, start=7600.0):
    rng = np.random.default_rng(seed)
    rows = []
    day0 = pd.Timestamp("2026-08-03", tz="UTC")
    px = start
    for d in range(days):
        day = day0 + pd.Timedelta(days=d)
        if day.weekday() >= 5:
            continue
        for k in range(96):
            t = day + pd.Timedelta(minutes=15 * k)
            ret = rng.normal(0.0002 if 54 <= k < 80 else 0.0, 0.0012)
            o = px; c = px * (1 + ret); hi = max(o, c) * (1 + abs(rng.normal(0, 0.0005))); lo = min(o, c) * (1 - abs(rng.normal(0, 0.0005)))
            rows.append((t, o, hi, lo, c, 1000.0)); px = c
    return pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"]).set_index("time")


def test_orb_and_ema_run_and_respect_session():
    df = _m15()
    for strat in ["orb", "ema"]:
        res = backtest_intraday({"US500": df}, IntradayParams(strategy=strat), initial=10_000)
        m = res.metrics()
        assert m["trades"] >= 1, strat
        for t in res.trades:
            assert t.exit_time.date() == t.entry_time.date()  # nunca se mantiene de un dia a otro
            h = t.exit_time.hour * 60 + t.exit_time.minute
            assert 13 * 60 + 30 <= h < 20 * 60
        assert abs(m["final"] - (10_000 + sum(t.pnl for t in res.trades))) < 0.01


def test_size_units_respects_min_lot_and_risk_cap():
    p = IntradayParams(risk_pct=0.01, max_risk_pct=0.03)
    # 200 USD, oro a 4300 con stop 0,4 %: el minimo de 1 oz arriesga 17 USD (8,6 %) -> no se opera
    assert size_units(200, 4300, 4300 * 0.996, SPECS["XAUUSD"], p) == 0.0
    # US500 a 7600: 2 USD de riesgo / 30 pts = 0.066 -> 0.06 unidades
    assert size_units(200, 7600, 7600 * 0.996, SPECS["US500"], p) == 0.06
    # con 10.000 USD el oro si es operable
    assert size_units(10_000, 4300, 4300 * 0.996, SPECS["XAUUSD"], p) >= 1
