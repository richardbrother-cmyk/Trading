import numpy as np
import pandas as pd

from autotrader.preopen import gap_verdict, projected_decision
from autotrader.strategy import StrategyParams


def test_gap_verdict_thresholds():
    assert gap_verdict(-0.02, 0.015, 0.03) is not None
    assert gap_verdict(-0.01, 0.015, 0.03) is None
    assert gap_verdict(0.035, 0.015, 0.03) is not None
    assert gap_verdict(0.02, 0.015, 0.03) is None


def _uptrend(n=200):
    close = np.linspace(100, 200, n)
    end = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None) - pd.Timedelta(days=1)
    idx = pd.bdate_range(end=end, periods=n)
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1e6}, index=idx)


def test_projected_decision_uses_live_price():
    df = _uptrend()
    assert projected_decision(df, 201.0, StrategyParams(10, 30, 14, 101.0))["action"] == "BUY"
    # con una SMA rapida de 2 barras, un desplome hoy invierte el cruce y la senal deja de ser BUY
    assert projected_decision(df, 120.0, StrategyParams(2, 30, 14, 101.0))["action"] == "HOLD"
