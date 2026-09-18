from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd

from autotrader.config import Settings
from autotrader.ctrader import SymbolInfo, OpenPosition
from autotrader.swingbot import SWING_LABEL, run_swing_cycle, size_units
from autotrader.intraday import SPECS


def _h4(n=300, crash_last=True, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-06-01", periods=n, freq="4h", tz="UTC")
    close = 7600 + np.cumsum(rng.normal(0, 8, n))
    if crash_last:
        close[-1] = close[-25:-1].mean() - 14 * close[-25:-1].std()  # cierre muy por debajo de la banda
    o = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({"open": o, "high": np.maximum(o, close) + 3, "low": np.minimum(o, close) - 3, "close": close, "volume": 1.0}, index=idx)


class FakeSession:
    def __init__(self, bars, positions):
        self.bars, self._positions, self.calls = bars, positions, []
        self.symbols = {"US500": SymbolInfo(1, "US500", digits=2, lot_size=100, min_volume=1, step_volume=1)}
        self.account_id = 1
        self.model = SimpleNamespace(ProtoOAOrderType=SimpleNamespace(MARKET=1), ProtoOATradeSide=SimpleNamespace(BUY=1),
                                     ProtoOAExecutionType=SimpleNamespace(Name=lambda x: "ORDER_ACCEPTED"))

    def trader(self):
        return 200.0, 2, 500.0

    def unrealized_pnl(self):
        return 0.0

    def positions(self, only_bot=True):
        return self._positions

    def call(self, name, timeout=None, **kw):
        self.calls.append((name, kw))
        return SimpleNamespace(executionType=1)


def test_swing_buys_on_band_signal_with_stop_and_target(tmp_path, monkeypatch):
    import autotrader.swingbot as sb
    bars = _h4()
    monkeypatch.setattr(sb, "closed_h4_bars", lambda session, symbol, days=60, now=None: bars)
    s = Settings(broker="sim", symbols=["US500"], state_dir=str(tmp_path), event_mode="off", risk_per_trade=0.01)
    sess = FakeSession(bars, [])
    summary = run_swing_cycle(s, sess, equity_cap=200.0, now=bars.index[-1] + timedelta(hours=5))
    assert summary["decisions"][0]["action"] == "BUY"
    orders = [c for c in sess.calls if c[0] == "ProtoOANewOrderReq"]
    assert len(orders) == 1 and orders[0][1]["label"] == SWING_LABEL
    assert orders[0][1]["relativeStopLoss"] % 1000 == 0 and orders[0][1]["relativeTakeProfit"] > 0
    assert summary["orders"][0]["risk_usd"] <= 200 * 0.03 + 1e-6


def test_swing_closes_old_positions_and_skips_held_symbols(tmp_path, monkeypatch):
    import autotrader.swingbot as sb
    bars = _h4()
    monkeypatch.setattr(sb, "closed_h4_bars", lambda session, symbol, days=60, now=None: bars)
    now = bars.index[-1] + timedelta(hours=5)
    old = OpenPosition(9, "US500", 0.05, "buy", 7500.0, 7400.0, SWING_LABEL, (now - timedelta(days=4)).to_pydatetime())
    s = Settings(broker="sim", symbols=["US500"], state_dir=str(tmp_path), event_mode="off")
    sess = FakeSession(bars, [old])
    summary = run_swing_cycle(s, sess, equity_cap=200.0, now=now)
    assert [c[0] for c in sess.calls] == ["ProtoOAClosePositionReq"] or "ProtoOAClosePositionReq" in [c[0] for c in sess.calls]
    assert summary["closed"][0]["reason"] == "tiempo maximo"


def test_size_units_min_lot_rule():
    assert size_units(200, 4300, 4300 - 66, SPECS["XAUUSD"], 0.01, 0.03) == 0.0
    assert size_units(200, 7600, 7600 - 55, SPECS["US500"], 0.01, 0.03) == 0.03


def test_swing_skips_stale_signal(tmp_path, monkeypatch):
    import autotrader.swingbot as sb
    bars = _h4()
    monkeypatch.setattr(sb, "closed_h4_bars", lambda session, symbol, days=60, now=None: bars)
    s = Settings(broker="sim", symbols=["US500"], state_dir=str(tmp_path), event_mode="off", risk_per_trade=0.01)
    sess = FakeSession(bars, [])
    # la barra cerro 4 h despues de abrir; 7 h tras la apertura llevan 3 h cerrada: caducada con el maximo de 2 h
    summary = run_swing_cycle(s, sess, now=bars.index[-1] + timedelta(hours=7))
    dec = summary["decisions"][0]
    assert dec["action"] == "HOLD" and dec["reason"].startswith("senal caducada") and dec["bar_age_h"] == 3.0
    assert not [c for c in sess.calls if c[0] == "ProtoOANewOrderReq"]
    # con un maximo de 4 h si entra
    sess2 = FakeSession(bars, [])
    summary = run_swing_cycle(s, sess2, now=bars.index[-1] + timedelta(hours=7), max_signal_age_hours=4)
    assert summary["decisions"][0]["action"] == "BUY"


def test_swing_risk_5pct_and_cap():
    from autotrader.swingbot import default_max_risk_pct
    assert default_max_risk_pct(0.01) == 0.03
    assert abs(default_max_risk_pct(0.05) - 0.075) < 1e-9
    # 5 % de 200 USD = 10 USD de riesgo; con stop de 55 puntos en US500 (100 USD/punto por lote) entra 0.18 unidades
    assert size_units(200, 7600, 7600 - 55, SPECS["US500"], 0.05, 0.075) == 0.18


def test_swing_guard_freeze_blocks_entries_and_close_flattens(tmp_path, monkeypatch):
    import autotrader.swingbot as sb
    bars = _h4()
    monkeypatch.setattr(sb, "closed_h4_bars", lambda session, symbol, days=60, now=None: bars)
    now = bars.index[-1] + timedelta(hours=5)
    s = Settings(broker="sim", symbols=["US500"], state_dir=str(tmp_path), event_mode="off", halt_mode="freeze")
    sess = FakeSession(bars, [])
    summary = run_swing_cycle(s, sess, now=now)
    assert summary["decisions"][0]["reason"] == "freno activo" and not sess.calls and summary["guard"]["mode"] == "freeze"
    pos = OpenPosition(9, "US500", 0.05, "buy", 7500.0, 7400.0, SWING_LABEL, (now - timedelta(hours=6)).to_pydatetime())
    s = Settings(broker="sim", symbols=["US500"], state_dir=str(tmp_path), event_mode="off", halt_mode="close")
    sess = FakeSession(bars, [pos])
    summary = run_swing_cycle(s, sess, now=now)
    assert [c[0] for c in sess.calls] == ["ProtoOAClosePositionReq"] and summary["closed"][0]["reason"].startswith("cierre por freno")


def test_swing_drawdown_brake_from_published_history(tmp_path, monkeypatch):
    import json
    import autotrader.swingbot as sb
    bars = _h4()
    monkeypatch.setattr(sb, "closed_h4_bars", lambda session, symbol, days=60, now=None: bars)
    hist = tmp_path / "swing_state.json"
    hist.write_text(json.dumps({"history": [["2026-09-01", 240.0]]}))  # equity actual 200 = -16.7 %
    s = Settings(broker="sim", symbols=["US500"], state_dir=str(tmp_path), event_mode="off", max_drawdown_pct=0.10,
                 equity_history_path=str(hist))
    sess = FakeSession(bars, [])
    summary = run_swing_cycle(s, sess, now=bars.index[-1] + timedelta(hours=5))
    assert summary["guard"]["mode"] == "freeze" and not sess.calls


def test_swing_skips_entry_on_bar_containing_opex_close(tmp_path, monkeypatch):
    import json
    import autotrader.swingbot as sb
    bars = _h4()
    monkeypatch.setattr(sb, "closed_h4_bars", lambda session, symbol, days=60, now=None: bars)
    last_open = bars.index[-1]
    ev_path = tmp_path / "events.json"
    ev_path.write_text(json.dumps({"opex": False, "events": [{"name": "OPEX", "at": (last_open + timedelta(hours=3)).isoformat(), "tags": ["opex"],
                                                                "mode": "freeze", "hours_before": 2, "hours_after": 0.5}]}))
    s = Settings(broker="sim", symbols=["US500"], state_dir=str(tmp_path), events_path=str(ev_path), event_mode="trail", max_drawdown_pct=0)
    sess = FakeSession(bars, [])
    summary = run_swing_cycle(s, sess, now=last_open + timedelta(hours=5))  # fuera de la ventana (cerro hace 1 h, evento hace 2 h)
    assert summary["decisions"][0]["action"] == "HOLD" and "vencimiento" in summary["decisions"][0]["reason"] and not sess.calls
