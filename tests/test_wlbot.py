from types import SimpleNamespace

import pandas as pd
import pytest

from autotrader.config import Settings
from autotrader.ctrader import OpenPosition, SymbolInfo
from autotrader.wlbot import NY, WLParams, daily_atr, run_wl_cycle, size_units, trading_week


def _bars(end_ny, days=16, base=4400.0, bar_range=4.0):
    """Barras M15 de lunes a viernes (dia de 18:00 a 17:00 NY, rotura 17:00-18:00) hasta `end_ny`; cada barra con rango `bar_range`."""
    idx = pd.date_range(end=pd.Timestamp(end_ny, tz=NY) - pd.Timedelta(minutes=15), periods=days * 96, freq="15min")
    idx = [t for t in idx if not (t.hour == 17) and not (t.weekday() == 5) and not (t.weekday() == 4 and t.hour >= 17)
           and not (t.weekday() == 6 and t.hour < 18)]
    df = pd.DataFrame({"open": base, "high": base + bar_range / 2, "low": base - bar_range / 2, "close": base, "volume": 1.0}, index=idx)
    return df.tz_convert("UTC")


def _now(ts):
    return pd.Timestamp(ts, tz=NY).tz_convert("UTC").to_pydatetime()


SUN = "2026-09-27 18:30"  # domingo, ya en la semana de negociacion del lunes 28
MON = "2026-09-28 09:00"
FRI = "2026-10-02 16:10"


def test_trading_week_boundaries():
    assert trading_week(pd.Timestamp("2026-09-27 17:59", tz=NY)) == ("2026-09-21", 6)  # aun semana anterior (fin de semana)
    assert trading_week(pd.Timestamp("2026-09-27 18:00", tz=NY)) == ("2026-09-28", 0)
    assert trading_week(pd.Timestamp("2026-09-28 16:59", tz=NY)) == ("2026-09-28", 0)
    assert trading_week(pd.Timestamp("2026-10-02 16:30", tz=NY)) == ("2026-09-28", 4)


def test_daily_atr_uses_prior_complete_days_only():
    bars = _bars("2026-09-27 18:30")
    atr = daily_atr(bars, _now(SUN), 10)
    assert atr is not None and atr == pytest.approx(4.0)  # rango constante: el dia entero mide lo mismo que una barra
    assert daily_atr(_bars("2026-09-27 18:30", days=3), _now(SUN), 10) is None  # historia insuficiente


def test_size_units_min_lot_and_cap():
    p = WLParams(stop_atr=0.5, risk_pct=0.06, max_risk_pct=0.15)
    one_oz = SymbolInfo(1, "XAUUSD", digits=2, lot_size=100, min_volume=100, step_volume=100)  # minimo y paso 1 onza (valores x100)
    assert size_units(440.0, 53.0, one_oz, p) == pytest.approx(1.0)  # 26,4/53 < 1 -> minimo; riesgo 53 <= 66
    assert size_units(440.0, 80.0, one_oz, p) == 0.0  # el minimo arriesga 80 > 66 (15 %)
    assert size_units(0.0, 53.0, one_oz, p) == 0.0


class FakeSession:
    def __init__(self, positions=()):
        self._positions, self.calls = list(positions), []
        self.symbols = {"XAUUSD": SymbolInfo(1, "XAUUSD", digits=2, lot_size=100, min_volume=100, step_volume=100)}
        self.account_id = 1
        self.model = SimpleNamespace(ProtoOAOrderType=SimpleNamespace(MARKET=1), ProtoOATradeSide=SimpleNamespace(BUY=1, SELL=2),
                                     ProtoOAExecutionType=SimpleNamespace(Name=lambda x: "ORDER_ACCEPTED"))

    def trader(self):
        return 440.0, 2, 500.0

    def unrealized_pnl(self):
        return 0.0

    def positions(self, only_bot=True):
        return self._positions

    def deals(self, days=14):
        return []

    def call(self, name, timeout=None, **kw):
        self.calls.append((name, kw))
        return SimpleNamespace(executionType=1)


def _settings(tmp_path):
    return Settings(broker="sim", symbols=["XAUUSD"], state_dir=str(tmp_path), event_mode="off", risk_per_trade=0.06, max_drawdown_pct=0)


def _patch_bars(monkeypatch, end):
    import autotrader.wlbot as wb
    bars = _bars(end)
    monkeypatch.setattr(wb, "m15_bars", lambda session, symbol, days=21, now=None: bars)


def test_enters_once_per_week_long_with_stop_and_no_target(tmp_path, monkeypatch):
    _patch_bars(monkeypatch, SUN)
    s, state = _settings(tmp_path), str(tmp_path / "wl.json")
    sess = FakeSession()
    out = run_wl_cycle(s, sess, WLParams(), now=_now(SUN), state_path=state)
    orders = [c for c in sess.calls if c[0] == "ProtoOANewOrderReq"]
    assert len(orders) == 1 and orders[0][1]["label"] == "autotrader-wl" and orders[0][1]["tradeSide"] == 1
    assert orders[0][1]["relativeStopLoss"] > 0 and "relativeTakeProfit" not in orders[0][1]
    assert out["publish"] is True and out["orders"][0]["units"] == pytest.approx(13.0) and out["orders"][0]["stop"] == pytest.approx(4400 - 2.0)
    # siguiente ciclo, misma semana: no repite aunque ya no haya posicion (salto del stop)
    sess2 = FakeSession()
    out2 = run_wl_cycle(s, sess2, WLParams(), now=_now(MON), state_path=state)
    assert not [c for c in sess2.calls if c[0] == "ProtoOANewOrderReq"] and any("ya se opero esta semana" in x for x in out2["skipped"])


def test_closes_friday_four_only_own_label_and_does_nothing_midweek(tmp_path, monkeypatch):
    _patch_bars(monkeypatch, FRI)
    s, state = _settings(tmp_path), str(tmp_path / "wl.json")
    mine = OpenPosition(5, "XAUUSD", 1.0, "buy", 4400.0, 4350.0, "autotrader-wl", None, None)
    other = OpenPosition(6, "XAUUSD", 2.0, "sell", 4400.0, 4420.0, "autotrader-asia", None, None)
    sess = FakeSession([mine, other])
    out = run_wl_cycle(s, sess, WLParams(), now=_now(FRI), state_path=state)
    closes = [c for c in sess.calls if c[0] == "ProtoOAClosePositionReq"]
    assert len(closes) == 1 and closes[0][1]["positionId"] == 5 and out["closed"][0]["reason"] == "cierre viernes 16:00 NY"
    mid = FakeSession([mine])
    run_wl_cycle(s, mid, WLParams(), now=_now("2026-09-30 12:00"), state_path=str(tmp_path / "b.json"))
    assert not mid.calls  # miercoles: ni compra ni cierra


def test_stale_position_from_previous_week_is_closed(tmp_path, monkeypatch):
    _patch_bars(monkeypatch, SUN)
    s, state = _settings(tmp_path), tmp_path / "wl.json"
    state.write_text('{"last_trade_week": "2026-09-21"}')
    mine = OpenPosition(5, "XAUUSD", 1.0, "buy", 4400.0, 4350.0, "autotrader-wl", None, None)
    sess = FakeSession([mine])
    out = run_wl_cycle(s, sess, WLParams(), now=_now(SUN), state_path=str(state))
    assert [c[1]["positionId"] for c in sess.calls if c[0] == "ProtoOAClosePositionReq"] == [5]
    assert out["closed"][0]["reason"] == "posicion de la semana anterior"
    assert len([c for c in sess.calls if c[0] == "ProtoOANewOrderReq"]) == 1  # y entra en la semana nueva


def test_weekend_and_late_week_do_not_enter(tmp_path, monkeypatch):
    _patch_bars(monkeypatch, "2026-09-26 12:00")
    s = _settings(tmp_path)
    for ts in ["2026-09-26 12:00", "2026-09-30 10:00", "2026-09-27 12:00"]:
        sess = FakeSession()
        run_wl_cycle(s, sess, WLParams(), now=_now(ts), state_path=str(tmp_path / "c.json"))
        assert not sess.calls
