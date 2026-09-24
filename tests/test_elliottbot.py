from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd

from autotrader.config import Settings
from autotrader.ctrader import OpenPosition, SymbolInfo
from autotrader.elliott import ElliottParams
from autotrader.elliottbot import run_elliott_cycle


def _h4_path(path, bars_per_leg=12, start="2026-08-01 01:00"):
    """Barras H4 que recorren en linea recta los puntos de `path`; la ultima barra es el ultimo punto."""
    closes = []
    for a, b in zip(path[:-1], path[1:]):
        closes += list(np.linspace(a, b, bars_per_leg, endpoint=False))
    closes.append(path[-1])
    idx = pd.date_range(start, periods=len(closes), freq="4h", tz="UTC")
    c = np.array(closes, dtype=float)
    o = np.concatenate([[c[0]], c[:-1]])
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + 0.5, "low": np.minimum(o, c) - 0.5, "close": c, "volume": 1.0}, index=idx)


class FakeSession:
    def __init__(self, positions):
        self._positions, self.calls = positions, []
        self.symbols = {"XAUUSD": SymbolInfo(1, "XAUUSD", digits=2, lot_size=100, min_volume=100, step_volume=100)}
        self.account_id = 1
        self.model = SimpleNamespace(ProtoOAOrderType=SimpleNamespace(MARKET=1), ProtoOATradeSide=SimpleNamespace(BUY=1, SELL=2),
                                     ProtoOAExecutionType=SimpleNamespace(Name=lambda x: "ORDER_ACCEPTED"))

    def trader(self):
        return 10_000.0, 2, 500.0  # con 500 USD el lote minimo de oro (1 oz, stop de ~100 USD) supera el tope de riesgo

    def unrealized_pnl(self):
        return 0.0

    def positions(self, only_bot=True):
        return self._positions

    def deals(self, days=14):
        return []

    def call(self, name, timeout=None, **kw):
        self.calls.append((name, kw))
        return SimpleNamespace(executionType=1)


# onda 1 de 2000 a 2100, onda 2 hasta 2050, y la ultima barra cierra sobre 2100: ruptura
PATH_READY = [2000, 2100, 2050, 2101]
P = ElliottParams(timeframe="H4", zz_atr=0, zz_pct=0.015, waves=(2,), entry="ruptura", target_ext=1.618, stop="inicio",
                  allow_short=False, atr_period=3, risk_pct=0.06, max_risk_pct=0.09)


def _settings(tmp_path):
    return Settings(broker="sim", symbols=["XAUUSD"], state_dir=str(tmp_path), event_mode="off", risk_per_trade=0.06, max_drawdown_pct=0)


def _run(tmp_path, monkeypatch, bars, positions, now=None, state="elliott_state.json", p=P, **kw):
    import autotrader.elliottbot as eb
    monkeypatch.setattr(eb, "closed_h4_bars", lambda session, symbol, days=60, now=None: bars)
    now = now or (bars.index[-1].to_pydatetime() + timedelta(hours=4, minutes=30))
    sess = FakeSession(positions)
    summary = run_elliott_cycle(_settings(tmp_path), sess, p=p, now=now, state_path=str(tmp_path / state), **kw)
    return summary, sess


def test_enters_on_breakout_bar_once(tmp_path, monkeypatch):
    bars = _h4_path(PATH_READY)
    summary, sess = _run(tmp_path, monkeypatch, bars, [])
    dec = summary["decisions"][0]
    assert dec["action"] == "BUY" and dec["phase"] == "ready" and dec["wave"] == 2
    orders = [c for c in sess.calls if c[0] == "ProtoOANewOrderReq"]
    assert len(orders) == 1 and orders[0][1]["tradeSide"] == 1 and orders[0][1]["label"] == "autotrader-elliott"
    o = summary["orders"][0]
    assert o["stop"] < 2000 and o["target"] > o["entry_ref"] and o["risk_usd"] <= 10_000 * 0.09 + 0.01
    assert orders[0][1]["relativeStopLoss"] > 0 and orders[0][1]["relativeTakeProfit"] > 0
    assert summary["publish"] is True
    # mismo ciclo una hora despues: la senal ya se opero, no se duplica
    summary2, sess2 = _run(tmp_path, monkeypatch, bars, [], now=bars.index[-1].to_pydatetime() + timedelta(hours=5, minutes=30))
    assert summary2["decisions"][0]["action"] == "HOLD" and summary2["decisions"][0]["reason"] == "senal ya operada"
    assert not [c for c in sess2.calls if c[0] == "ProtoOANewOrderReq"]


def test_stale_signal_and_open_position_block_entry(tmp_path, monkeypatch):
    bars = _h4_path(PATH_READY)
    late = bars.index[-1].to_pydatetime() + timedelta(hours=7)  # la barra cerro hace 3 h
    summary, sess = _run(tmp_path, monkeypatch, bars, [], now=late)
    assert summary["decisions"][0]["action"] == "HOLD" and "caducada" in summary["decisions"][0]["reason"]
    assert not [c for c in sess.calls if c[0] == "ProtoOANewOrderReq"]
    pos = OpenPosition(7, "XAUUSD", 1.0, "buy", 2105.0, 1990.0, "autotrader-elliott", bars.index[-1].to_pydatetime(), 2200.0)
    summary2, sess2 = _run(tmp_path, monkeypatch, bars, [pos], state="s2.json")
    assert summary2["decisions"][0]["reason"] == "posicion abierta" and summary2["positions"] == {"XAUUSD": 1.0}


def test_no_pattern_means_hold(tmp_path, monkeypatch):
    bars = _h4_path([2000, 2100, 2050, 2090])  # onda 2 confirmada, sin ruptura todavia
    summary, sess = _run(tmp_path, monkeypatch, bars, [])
    assert summary["decisions"][0]["action"] == "HOLD" and summary["decisions"][0]["phase"] == "waiting_break"
    assert not [c for c in sess.calls if c[0] == "ProtoOANewOrderReq"]
    bars2 = _h4_path([2000, 2100, 2050, 2070])  # el minimo de la onda 2 aun no esta confirmado: sin patron, pero con pivotes
    summary2, _ = _run(tmp_path, monkeypatch, bars2, [], state="s3.json")
    assert summary2["decisions"][0]["phase"] == "sin patron" and len(summary2["decisions"][0]["pivots"]) == 2


def test_min_lot_over_risk_cap_is_skipped(tmp_path, monkeypatch):
    bars = _h4_path(PATH_READY)
    import autotrader.elliottbot as eb
    monkeypatch.setattr(eb, "closed_h4_bars", lambda session, symbol, days=60, now=None: bars)
    sess = FakeSession([]); sess.trader = lambda: (500.0, 2, 500.0)
    now = bars.index[-1].to_pydatetime() + timedelta(hours=4, minutes=30)
    summary = run_elliott_cycle(_settings(tmp_path), sess, p=P, now=now, state_path=str(tmp_path / "s4.json"))
    assert summary["decisions"][0]["action"] == "BUY" and not [c for c in sess.calls if c[0] == "ProtoOANewOrderReq"]
    assert any("lote minimo" in x for x in summary["skipped"])


def test_closes_position_after_max_hold(tmp_path, monkeypatch):
    bars = _h4_path([2000, 2100, 2050, 2090])
    opened = bars.index[-1].to_pydatetime() - timedelta(days=11)
    pos = OpenPosition(9, "XAUUSD", 1.0, "buy", 2050.0, 1990.0, "autotrader-elliott", opened, 2200.0)
    other = OpenPosition(10, "XAUUSD", 1.0, "buy", 2050.0, 1990.0, "autotrader-aggr", opened, 2200.0)
    summary, sess = _run(tmp_path, monkeypatch, bars, [pos, other])
    closes = [c for c in sess.calls if c[0] == "ProtoOAClosePositionReq"]
    assert len(closes) == 1 and closes[0][1]["positionId"] == 9
    assert summary["closed"][0]["reason"] == "tiempo maximo"
