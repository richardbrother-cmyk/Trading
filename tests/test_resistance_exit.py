import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from autotrader.backtest import run_backtest
from autotrader.bot import new_cross_since, resistance_check, run_cycle
from autotrader.broker import Account, Position
from autotrader.config import Settings
from autotrader.risk import RiskParams
from autotrader.strategy import StrategyParams


def _today() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc).date())


def _uptrend(n: int = 80, today_high: float | None = None) -> pd.DataFrame:
    idx = pd.bdate_range(end=_today(), periods=n)
    close = np.linspace(100.0, 140.0, n) + np.where(np.arange(n) % 2 == 0, -2.0, 2.0)  # zigzag: RSI por debajo de 70
    df = pd.DataFrame({"open": close - 0.2, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": 1e6}, index=idx)
    if today_high is not None:
        df.iloc[-1, df.columns.get_loc("high")] = today_high
    return df


def _dip_then_recover(n: int = 80) -> pd.DataFrame:
    """Subida, caida fuerte (SMA rapida bajo la lenta) y recuperacion: al final la tendencia vuelve a estar arriba."""
    idx = pd.bdate_range(end=_today(), periods=n)
    close = np.concatenate([np.linspace(100, 130, 40), np.linspace(130, 95, 15), np.linspace(95, 150, 25)]) + np.where(np.arange(n) % 2 == 0, -6.0, 6.0)
    return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1e6}, index=idx)


class FakeBroker:
    name = "fake"

    def __init__(self, positions=None):
        self._positions, self.orders = positions or {}, []

    def account(self, prices=None):
        return Account(cash=100_000, equity=100_000, positions=dict(self._positions), pending_buys=set())

    def is_market_open(self):
        return True

    def submit_market_order(self, symbol, qty, side, price_hint=None):
        self.orders.append((symbol, side, qty))
        return {"id": "1", "status": "accepted"}


class FakeProvider:
    def __init__(self, frames):
        self.frames = frames

    def bars_many(self, symbols):
        return {s: self.frames[s] for s in symbols if s in self.frames}

    def quote(self, symbol):
        return None


def _settings(tmp_path, **kw):
    return Settings(broker="sim", symbols=["AAA"], state_dir=str(tmp_path), event_mode="off", resistance_lookback=20,
                    resistance_state_path=str(tmp_path / "res.json"), **kw)


def test_resistance_check_levels():
    df = _uptrend(today_high=200.0)
    hit, level, hi = resistance_check(df, 140.0, 100.0, 20, 0.0, _today())
    assert hit and abs(level - float(df["high"].iloc[-21:-1].max())) < 1e-9 and hi == 200.0
    # sin tocar la resistencia, o tocandola en perdidas, no hay salida
    assert resistance_check(_uptrend(today_high=100.0), 140.0, 100.0, 20, 0.0, _today())[0] is False
    assert resistance_check(df, 140.0, 150.0, 20, 0.0, _today())[0] is False


def test_cycle_exits_at_resistance_and_waits_for_new_cross(tmp_path):
    s = _settings(tmp_path)
    df = _uptrend(today_high=200.0)
    broker = FakeBroker({"AAA": Position("AAA", 10, 100.0)})
    summary = run_cycle(s, broker, FakeProvider({"AAA": df}), force=True)
    dec = summary["decisions"][0]
    assert dec["action"] == "SELL" and dec["reason"].startswith("resistencia de 20 dias")
    assert broker.orders == [("AAA", "sell", 10)]
    assert json.load(open(tmp_path / "res.json"))["AAA"]["date"] == _today().date().isoformat()
    assert summary["resistance_waiting"] == ["AAA"]
    # siguiente ciclo sin posicion y con la tendencia intacta: no reentra
    broker2 = FakeBroker({})
    summary2 = run_cycle(s, broker2, FakeProvider({"AAA": _uptrend()}), force=True)
    assert summary2["decisions"][0]["action"] == "HOLD" and "espera un cruce" in summary2["decisions"][0]["reason"]
    assert broker2.orders == []
    # con un cruce alcista nuevo despues de la salida, la reentrada vuelve a estar permitida y el bloqueo se borra
    state = {"AAA": {"date": (_today() - pd.Timedelta(days=60)).date().isoformat(), "level": 131.0, "price": 130.0}}
    json.dump(state, open(tmp_path / "res.json", "w"))
    broker3 = FakeBroker({})
    summary3 = run_cycle(s, broker3, FakeProvider({"AAA": _dip_then_recover()}), force=True)
    assert summary3["decisions"][0]["action"] == "BUY" and broker3.orders and broker3.orders[0][1] == "buy"
    assert json.load(open(tmp_path / "res.json")) == {}


def test_new_cross_since():
    params = StrategyParams(20, 50, 14, 70.0)
    assert new_cross_since(_dip_then_recover(), (_today() - pd.Timedelta(days=60)).date().isoformat(), params) is True
    assert new_cross_since(_uptrend(), (_today() - pd.Timedelta(days=10)).date().isoformat(), params) is False


def test_backtest_resistance_hook_reduces_exposure():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2024-01-01", periods=400)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0008, 0.01, 400)))
    df = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1e6}, index=idx)
    risk = RiskParams(0.02, 5, 0.25, 0.03, 0.05, 1.0)
    base = run_backtest({"AAA": df}, StrategyParams(20, 50, 14, 70.0), risk)
    res = run_backtest({"AAA": df}, StrategyParams(20, 50, 14, 70.0), risk, resistance_lookback=20, resistance_reentry="signal")
    reasons = {t.reason for t in res.trades}
    assert "resistance" in reasons and len(res.trades) >= len(base.trades)


def test_stop_uses_live_price_not_stale_daily_close(tmp_path):
    """cTrader no incluye la barra del dia: con un cierre de ayer un 6 % por debajo del precio real, el stop del 3 %
    saltaba nada mas comprar. Con el precio en vivo del broker no salta."""

    class LiveBroker(FakeBroker):
        def quote(self, symbol):
            return {"last": 3.22, "at": None, "prev_close": None}

    df = _uptrend()
    df.iloc[-1, df.columns.get_loc("close")] = 3.02  # cierre de ayer, muy por debajo de la entrada de hoy
    s = Settings(broker="sim", symbols=["AAA"], state_dir=str(tmp_path), event_mode="off", stop_loss_pct=0.03)
    stale = FakeBroker({"AAA": Position("AAA", 1800, 3.22)})
    out = run_cycle(s, stale, FakeProvider({"AAA": df}), force=True)
    assert out["decisions"][0]["action"] == "SELL" and out["decisions"][0]["reason"].startswith("stop loss")
    live = LiveBroker({"AAA": Position("AAA", 1800, 3.22)})
    out2 = run_cycle(s, live, FakeProvider({"AAA": df}), force=True)
    assert out2["decisions"][0]["action"] != "SELL" and out2["decisions"][0]["live"] == 3.22 and live.orders == []
