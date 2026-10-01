"""Soportes y resistencias: niveles, senales sin mirar al futuro, backtest y ciclo en vivo."""
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd

from autotrader.config import Settings
from autotrader.ctrader import SymbolInfo
from autotrader.sr import cluster_levels, find_pivots
from autotrader.swing import SwingParams, backtest_symbol, indicators, signal
from autotrader.swingbot import describe, history_days, run_swing_cycle


def _range_series(seed=3, mirror=False, start="2026-01-05 01:00") -> pd.DataFrame:
    """Tendencia alcista hasta ~100 y despues rango 100-110 con ciclos de 20 barras H4 (soporte ~100, resistencia ~110)."""
    rng = np.random.default_rng(seed)
    close = list(np.linspace(60, 100, 260) + rng.normal(0, 0.15, 260))
    for t in range(120):
        phase = (t % 20) / 20
        close.append(105 - 5 * np.sin(2 * np.pi * phase - np.pi / 2) + rng.normal(0, 0.15))
    close = np.array(close)
    if mirror:
        close = 160 - close
    o = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(o, close) + 0.25
    low = np.minimum(o, close) - 0.25
    idx = pd.date_range(start, periods=len(close), freq="4h", tz="UTC")
    return pd.DataFrame({"open": o, "high": high, "low": low, "close": close, "volume": 1.0}, index=idx)


def _params(**kw) -> SwingParams:
    base = dict(strategy="sr", timeframe="H4", sr_mode="bounce", sr_target="level", sr_min_rr=1.0, sr_lookback=300, sr_trend=True,
                allow_short=False, risk_pct=0.03, max_risk_pct=0.045, max_hold_days=7.0)
    base.update(kw)
    return SwingParams(**base)


def _bounce_end(df: pd.DataFrame) -> pd.DataFrame:
    """Recorta la serie en su ultima barra con senal de rebote (la serie es determinista: semilla fija)."""
    d = indicators(df, _params())
    sigs = d.index[d["sr_signal"] != 0]
    assert len(sigs) >= 3
    return df.loc[:sigs[-1]]


def test_find_pivots_need_k_bars_on_each_side():
    high = np.array([1, 2, 5, 2, 1, 2, 4, 2, 1], dtype=float)
    low = high - 0.5
    idx, price, kind = find_pivots(high, low, k=2)
    assert list(idx[kind == 1]) == [2, 6] and list(price[kind == 1]) == [5.0, 4.0]
    assert list(idx[kind == -1]) == [4]  # el minimo de la barra 4 (0.5) es un valle
    # el ultimo pivote posible necesita k barras a la derecha: ninguno en las dos ultimas barras
    assert idx.max() <= len(high) - 1 - 2


def test_cluster_levels_groups_by_width_and_drops_single_touches():
    levels = cluster_levels(np.array([100.0, 100.2, 99.9, 110.0, 120.0, 120.3]), width=0.6, min_touches=2)
    assert [(round(p, 2), n) for p, n in levels] == [(100.03, 3), (120.15, 2)]
    assert cluster_levels(np.array([100.0, 105.0]), width=0.5, min_touches=2) == []
    assert cluster_levels(np.array([]), width=1.0, min_touches=1) == []


def test_signals_do_not_look_ahead():
    df = _range_series()
    full = indicators(df, _params())
    cut = 330
    part = indicators(df.iloc[:cut], _params())
    cols = ["sr_signal", "sr_stop", "sr_target", "sr_level"]
    a, b = full.iloc[:cut][cols], part[cols]
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_bounce_signal_geometry():
    df = _bounce_end(_range_series())
    p = _params()
    d = indicators(df, p)
    last = d.iloc[-1]
    assert last["sr_signal"] == 1 and signal(d, len(d) - 1, p) == 1
    assert last["sr_stop"] < last["sr_level"] < last["close"] < last["sr_target"]
    risk = last["close"] - last["sr_stop"]
    assert 0.4 * last["atr"] <= risk <= 3.0 * last["atr"] + 1e-9
    assert (last["sr_target"] - last["close"]) >= p.sr_min_rr * risk


def test_min_rr_filters_out_trades_without_room():
    df = _bounce_end(_range_series())
    strict = indicators(df, _params(sr_min_rr=50.0))
    assert (strict["sr_signal"] == 0).all()


def test_fixed_rr_target_leaves_target_to_the_bot():
    d = indicators(_bounce_end(_range_series()), _params(sr_target="rr", sr_rr=2.0))
    assert d["sr_signal"].iloc[-1] == 1 and np.isnan(d["sr_target"].iloc[-1])


def test_longs_only_blocks_shorts_but_mirror_series_has_them():
    mirror = _range_series(mirror=True)
    p_long = _params(sr_trend=True)
    d = indicators(mirror, p_long)
    assert all(signal(d, i, p_long) != -1 for i in range(len(d)))
    p_both = _params(allow_short=True)
    d2 = indicators(mirror, p_both)
    shorts = [i for i in range(len(d2)) if signal(d2, i, p_both) == -1]
    assert shorts
    i = shorts[-1]
    assert d2["sr_stop"].iloc[i] > d2["sr_level"].iloc[i] > 0 and d2["sr_stop"].iloc[i] > d2["close"].iloc[i]


def _to_m15(h4: pd.DataFrame) -> pd.DataFrame:
    rows, idx = [], []
    for t, r in h4.iterrows():
        path = np.linspace(r.open, r.close, 16)
        for k in range(16):
            hi, lo = path[k] + 0.01, path[k] - 0.01
            if k == 4:
                hi = max(hi, r.high)
            if k == 8:
                lo = min(lo, r.low)
            o = path[k - 1] if k else r.open
            rows.append((o, max(hi, o, path[k]), min(lo, o, path[k]), path[k], 1.0))
            idx.append(t + pd.Timedelta(minutes=15 * k))
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx), columns=["open", "high", "low", "close", "volume"])


def test_backtest_uses_level_stop_and_target():
    h4 = _range_series()
    trades = backtest_symbol(_to_m15(h4), "US500", _params(), 10_000.0)
    assert trades, "la serie sintetica debe producir operaciones"
    assert all(t.side == 1 and t.stop < t.entry for t in trades)
    assert {t.reason for t in trades} <= {"stop", "objetivo", "tiempo maximo", "break even"}
    assert any(t.reason == "objetivo" for t in trades)  # el rango sube hasta la resistencia
    assert all(abs(t.entry - t.stop) > 0 for t in trades)


def test_history_days_and_description():
    assert history_days(_params(sr_lookback=300)) > history_days(SwingParams("breakout", "H4"))
    assert history_days(_params(sr_lookback=300)) >= 80
    text = describe(_params())
    assert "soporte" in text and "resistencia" in text


class FakeSession:
    def __init__(self):
        self.calls = []
        self.symbols = {"US500": SymbolInfo(1, "US500", digits=2, lot_size=100, min_volume=1, step_volume=1)}
        self.account_id = 1
        self.model = SimpleNamespace(ProtoOAOrderType=SimpleNamespace(MARKET=1), ProtoOATradeSide=SimpleNamespace(BUY=1),
                                     ProtoOAExecutionType=SimpleNamespace(Name=lambda x: "ORDER_ACCEPTED"))

    def trader(self):
        return 500.0, 2, 500.0

    def unrealized_pnl(self):
        return 0.0

    def positions(self, only_bot=True):
        return []

    def call(self, name, timeout=None, **kw):
        self.calls.append((name, kw))
        return SimpleNamespace(executionType=1)


def test_live_cycle_buys_bounce_with_level_stop_and_target(tmp_path, monkeypatch):
    import autotrader.swingbot as sb
    bars = _bounce_end(_range_series())
    monkeypatch.setattr(sb, "closed_h4_bars", lambda session, symbol, days=60, now=None: bars)
    s = Settings(broker="sim", symbols=["US500"], state_dir=str(tmp_path), event_mode="off", risk_per_trade=0.03)
    sess = FakeSession()
    summary = run_swing_cycle(s, sess, params=_params(), dry_run=False, now=bars.index[-1] + timedelta(hours=4, minutes=30),
                              label="autotrader-sr", max_positions=2)
    dec = summary["decisions"][0]
    assert dec["action"] == "BUY" and dec["sr_level"] < dec["close"] < dec["sr_target"] and dec["sr_stop"] < dec["sr_level"]
    orders = [c for c in sess.calls if c[0] == "ProtoOANewOrderReq"]
    assert len(orders) == 1 and orders[0][1]["label"] == "autotrader-sr"
    o = summary["orders"][0]
    assert o["stop"] == dec["sr_stop"] and abs(o["target"] - dec["sr_target"]) < 0.02
    assert o["risk_usd"] <= 500 * 0.045 + 1e-6 and o["risk_usd"] > 0
    assert orders[0][1]["relativeTakeProfit"] > orders[0][1]["relativeStopLoss"]


def test_live_cycle_holds_when_signal_is_stale(tmp_path, monkeypatch):
    import autotrader.swingbot as sb
    bars = _bounce_end(_range_series())
    monkeypatch.setattr(sb, "closed_h4_bars", lambda session, symbol, days=60, now=None: bars)
    s = Settings(broker="sim", symbols=["US500"], state_dir=str(tmp_path), event_mode="off", risk_per_trade=0.03)
    sess = FakeSession()
    summary = run_swing_cycle(s, sess, params=_params(), now=bars.index[-1] + timedelta(hours=9), label="autotrader-sr")
    assert summary["decisions"][0]["action"] == "HOLD" and "caducada" in summary["decisions"][0]["reason"]
    assert not [c for c in sess.calls if c[0] == "ProtoOANewOrderReq"]


def test_last_levels_reports_nearest_support_and_resistance():
    from autotrader.sr import last_levels
    df = _bounce_end(_range_series())
    p = _params()
    d = indicators(df, p)
    near = last_levels(d, p)
    assert near["levels"] >= 2
    assert near["support"] is not None and near["resistance"] is not None
    assert near["support"] <= d["close"].iloc[-1] < near["resistance"]
    assert 98 < near["support"] < 102 and 108 < near["resistance"] < 112  # soporte ~100, resistencia ~110 de la serie


def test_live_decision_carries_levels_even_without_signal(tmp_path, monkeypatch):
    import autotrader.swingbot as sb
    bars = _range_series().iloc[:300]
    monkeypatch.setattr(sb, "closed_h4_bars", lambda session, symbol, days=60, now=None: bars)
    s = Settings(broker="sim", symbols=["US500"], state_dir=str(tmp_path), event_mode="off", risk_per_trade=0.03)
    summary = run_swing_cycle(s, FakeSession(), params=_params(sr_min_rr=50.0), now=bars.index[-1] + timedelta(hours=4, minutes=20), label="autotrader-sr")
    dec = summary["decisions"][0]
    assert dec["action"] == "HOLD" and dec["levels"] >= 1 and "support" in dec and "resistance" in dec


def test_sr_state_script_keeps_last_cycle_and_events(tmp_path):
    import importlib.util
    import json
    spec = importlib.util.spec_from_file_location("sr_state", "scripts/sr_state.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    log = tmp_path / "run_log.jsonl"
    cycle = {"kind": "swing", "label": "autotrader-sr", "timestamp": "2026-10-02T07:15:00+00:00", "strategy": "x", "equity": 500.0,
             "guard": {"mode": "off"}, "positions": {}, "skipped": [],
             "decisions": [{"symbol": "GBPUSD", "action": "BUY", "close": 1.3, "sr_level": 1.29, "sr_stop": 1.28, "sr_target": 1.33, "bar": "b"}],
             "orders": [{"symbol": "GBPUSD", "status": "order_accepted"}], "closed": []}
    other = dict(cycle, label="autotrader-aggr", equity=999.0)
    log.write_text("\n".join(json.dumps(x) for x in (cycle, other)) + "\n", encoding="utf-8")
    rec = mod.last_cycle(str(log), "autotrader-sr")
    assert rec["equity"] == 500.0
    st = mod.build_state({}, rec)
    assert st["at"] == "2026-10-02T07:15Z" and len(st["events"]) == 1 and st["events"][0]["signals"][0]["symbol"] == "GBPUSD"
    st2 = mod.build_state(st, dict(cycle, decisions=[{"symbol": "US500", "action": "HOLD"}], orders=[]))
    assert len(st2["events"]) == 1  # un ciclo sin senal ni orden no anade eventos
    assert mod.last_cycle(str(log), "otra") is None
