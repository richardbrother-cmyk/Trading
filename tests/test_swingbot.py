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

    def amend_stop(self, position_id, stop_price, take_profit=0.0):
        self.calls.append(("ProtoOAAmendPositionSLTPReq", {"positionId": position_id, "stopLoss": stop_price, "takeProfit": take_profit}))


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


def _h4_breakout(n=300, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-06-01", periods=n, freq="4h", tz="UTC")
    close = 7000 + np.cumsum(rng.normal(0.8, 6, n))  # tendencia alcista: cierre sobre la EMA200
    close[-1] = close[-25:-1].max() + 40  # ultima barra rompe el maximo de 20 barras
    o = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({"open": o, "high": np.maximum(o, close) + 3, "low": np.minimum(o, close) - 3, "close": close, "volume": 1.0}, index=idx)


def test_aggressive_profile_breakout_with_fixed_6r_target(tmp_path, monkeypatch):
    import autotrader.swingbot as sb
    from autotrader.swing import SwingParams
    bars = _h4_breakout()
    monkeypatch.setattr(sb, "closed_h4_bars", lambda session, symbol, days=60, now=None: bars)
    s = Settings(broker="sim", symbols=["US500"], state_dir=str(tmp_path), event_mode="off", risk_per_trade=0.03, max_drawdown_pct=0)
    p = SwingParams("breakout", "H4", stop_atr=0.75, tp_atr=4.5, pure_rr=True, allow_short=False, max_hold_days=7.0, risk_pct=0.03, max_risk_pct=0.045)
    sess = FakeSession(bars, [])
    summary = run_swing_cycle(s, sess, params=p, now=bars.index[-1] + timedelta(hours=5), label="autotrader-aggr", max_positions=3)
    dec = summary["decisions"][0]
    assert dec["action"] == "BUY", dec
    order = summary["orders"][0]
    stop_dist = order["entry_ref"] - order["stop"]
    assert abs((order["target"] - order["entry_ref"]) / stop_dist - 6.0) < 0.05  # objetivo 6R
    assert order["risk_usd"] <= 200 * 0.045 + 1e-6
    sent = [c for c in sess.calls if c[0] == "ProtoOANewOrderReq"]
    assert len(sent) == 1 and sent[0][1]["label"] == "autotrader-aggr" and "posiciones" not in summary.get("strategy", "")
    assert "ruptura" in summary["strategy"] or "maximo" in summary["strategy"]


def test_aggressive_profile_respects_max_positions(tmp_path, monkeypatch):
    import autotrader.swingbot as sb
    from autotrader.swing import SwingParams
    bars = _h4_breakout()
    monkeypatch.setattr(sb, "closed_h4_bars", lambda session, symbol, days=60, now=None: bars)
    now = bars.index[-1] + timedelta(hours=5)
    held = [OpenPosition(i, sym, 0.05, "buy", 7000.0, 6950.0, "autotrader-aggr", (now - timedelta(hours=6)).to_pydatetime()) for i, sym in enumerate(["EURUSD", "GBPUSD", "XAUUSD"])]
    s = Settings(broker="sim", symbols=["US500"], state_dir=str(tmp_path), event_mode="off", risk_per_trade=0.03, max_drawdown_pct=0)
    p = SwingParams("breakout", "H4", stop_atr=0.75, tp_atr=4.5, pure_rr=True, allow_short=False, max_hold_days=7.0, risk_pct=0.03, max_risk_pct=0.045)
    sess = FakeSession(bars, held)
    summary = run_swing_cycle(s, sess, params=p, now=now, label="autotrader-aggr", max_positions=3)
    assert summary["decisions"][0]["action"] == "HOLD" and "maximo de 3" in summary["decisions"][0]["reason"]
    assert not [c for c in sess.calls if c[0] == "ProtoOANewOrderReq"]


def test_breakeven_moves_stop_to_entry_once_and_keeps_target(tmp_path, monkeypatch):
    import autotrader.swingbot as sb
    from autotrader.swing import SwingParams
    bars = _h4(crash_last=False)
    monkeypatch.setattr(sb, "closed_h4_bars", lambda session, symbol, days=60, now=None: bars)
    now = bars.index[-1] + timedelta(hours=5)
    # entrada 7000, stop 6950 (R = 50), objetivo 7300; precio actual 7060 = 1,2 R ganados
    pos = OpenPosition(7, "US500", 0.05, "buy", 7000.0, 6950.0, "autotrader-aggr", (now - timedelta(hours=8)).to_pydatetime(), 7300.0)
    s = Settings(broker="sim", symbols=["US500"], state_dir=str(tmp_path), event_mode="off", risk_per_trade=0.06, max_drawdown_pct=0)
    p = SwingParams("breakout", "H4", stop_atr=0.75, tp_atr=4.5, pure_rr=True, allow_short=False, max_hold_days=7.0, risk_pct=0.06,
                    max_risk_pct=0.09, breakeven_r=1.0, breakeven_lock_r=0.1)
    sess = FakeSession(bars, [pos])
    sess.last_price = lambda symbol, now=None: 7060.0
    summary = run_swing_cycle(s, sess, params=p, now=now, label="autotrader-aggr", max_positions=3)
    amends = [c for c in sess.calls if c[0] == "ProtoOAAmendPositionSLTPReq"]
    assert len(amends) == 1
    assert amends[0][1]["positionId"] == 7 and amends[0][1]["stopLoss"] == 7005.0 and amends[0][1]["takeProfit"] == 7300.0
    assert summary["stops_moved"][0]["gained_r"] == 1.2 and summary["stops_moved"][0]["status"] == "amended"
    # segunda pasada: el stop ya esta en break even, no se vuelve a tocar
    pos2 = OpenPosition(7, "US500", 0.05, "buy", 7000.0, 7005.0, "autotrader-aggr", pos.opened_at, 7300.0)
    sess2 = FakeSession(bars, [pos2])
    sess2.last_price = lambda symbol, now=None: 7100.0
    summary2 = run_swing_cycle(s, sess2, params=p, now=now, label="autotrader-aggr", max_positions=3)
    assert not [c for c in sess2.calls if c[0] == "ProtoOAAmendPositionSLTPReq"] and "stops_moved" not in summary2


def test_breakeven_waits_until_gain_reaches_threshold(tmp_path, monkeypatch):
    import autotrader.swingbot as sb
    from autotrader.swing import SwingParams
    bars = _h4(crash_last=False)
    monkeypatch.setattr(sb, "closed_h4_bars", lambda session, symbol, days=60, now=None: bars)
    now = bars.index[-1] + timedelta(hours=5)
    pos = OpenPosition(7, "US500", 0.05, "buy", 7000.0, 6950.0, "autotrader-aggr", (now - timedelta(hours=8)).to_pydatetime(), 7300.0)
    s = Settings(broker="sim", symbols=["US500"], state_dir=str(tmp_path), event_mode="off", risk_per_trade=0.06, max_drawdown_pct=0)
    p = SwingParams("breakout", "H4", stop_atr=0.75, tp_atr=4.5, pure_rr=True, allow_short=False, max_hold_days=7.0, risk_pct=0.06,
                    max_risk_pct=0.09, breakeven_r=1.0, breakeven_lock_r=0.1)
    sess = FakeSession(bars, [pos])
    sess.last_price = lambda symbol, now=None: 7040.0  # 0,8 R: todavia no
    summary = run_swing_cycle(s, sess, params=p, now=now, label="autotrader-aggr", max_positions=3)
    assert not [c for c in sess.calls if c[0] == "ProtoOAAmendPositionSLTPReq"] and "stops_moved" not in summary
    # perfil sin break even (swing de 200 USD): nunca toca el stop aunque vaya muy en ganancia
    sess3 = FakeSession(bars, [pos])
    sess3.last_price = lambda symbol, now=None: 7200.0
    p0 = SwingParams("breakout", "H4", stop_atr=0.75, tp_atr=4.5, pure_rr=True, allow_short=False, max_hold_days=7.0, risk_pct=0.06, max_risk_pct=0.09)
    run_swing_cycle(s, sess3, params=p0, now=now, label="autotrader-aggr", max_positions=3)
    assert not [c for c in sess3.calls if c[0] == "ProtoOAAmendPositionSLTPReq"]


def test_params_from_env_reads_breakeven(monkeypatch):
    from autotrader.swingbot import params_from_env, describe
    monkeypatch.setenv("SWING_BREAKEVEN_R", "1.5")
    monkeypatch.setenv("SWING_BREAKEVEN_LOCK_R", "0.1")
    s = Settings(broker="sim", symbols=["US500"], risk_per_trade=0.06)
    p = params_from_env(s)
    assert p.breakeven_r == 1.5 and p.breakeven_lock_r == 0.1 and "break even tras 1.5 R" in describe(p)
