from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from autotrader.asiabot import NY, AsiaParams, day_frames, plan_day, run_asia_cycle, size_units
from autotrader.config import Settings
from autotrader.ctrader import OpenPosition, SymbolInfo


def _bars(rows):
    """rows: lista de (hora NY 'HH:MM', open, high, low, close) del mismo dia 2026-09-21 (lunes)."""
    idx = [pd.Timestamp(f"2026-09-21 {hm}", tz=NY) for hm, *_ in rows]
    return pd.DataFrame([r[1:] for r in rows], index=idx, columns=["open", "high", "low", "close"]).assign(volume=1.0)


ASIA_HI, ASIA_LO = 4400.0, 4380.0


def test_plan_waiting_then_break_retest_ready():
    p = AsiaParams()
    nym = _bars([("08:00", 4390, 4398, 4388, 4395), ("08:15", 4395, 4399, 4392, 4397)])
    assert plan_day(ASIA_HI, ASIA_LO, nym, p)["phase"] == "waiting_break"
    nym = _bars([("08:00", 4390, 4398, 4388, 4395), ("08:15", 4395, 4405, 4394, 4403)])  # cierra fuera
    plan = plan_day(ASIA_HI, ASIA_LO, nym, p)
    assert plan["phase"] == "broke" and plan["side"] == 1 and plan["level"] == ASIA_HI
    nym = _bars([("08:00", 4390, 4398, 4388, 4395), ("08:15", 4395, 4405, 4394, 4403), ("08:30", 4403, 4406, 4399.5, 4402)])  # toca el nivel
    plan = plan_day(ASIA_HI, ASIA_LO, nym, p)
    assert plan["phase"] == "ready" and plan["retest_at"] == "08:30" and plan["stop"] == 4390.0
    # una vela mas tarde el retest ya no es la ultima: no se persigue
    nym2 = pd.concat([nym, _bars([("08:45", 4402, 4408, 4401, 4407)])])
    assert plan_day(ASIA_HI, ASIA_LO, nym2, p)["phase"] == "retest"


def test_plan_failed_break_and_expiry():
    p = AsiaParams(max_wait_bars=2)
    nym = _bars([("08:00", 4390, 4398, 4388, 4395), ("08:15", 4395, 4405, 4394, 4403), ("08:30", 4403, 4404, 4396, 4397)])  # cierra dentro
    assert plan_day(ASIA_HI, ASIA_LO, nym, p)["phase"] == "failed"
    nym = _bars([("08:00", 4390, 4398, 4388, 4395), ("08:15", 4395, 4405, 4394, 4403), ("08:30", 4403, 4410, 4402, 4408),
                 ("08:45", 4408, 4412, 4405, 4410)])  # nunca vuelve al nivel en 2 velas
    assert plan_day(ASIA_HI, ASIA_LO, nym, p)["phase"] == "expired"
    nym = _bars([("08:00", 4390, 4392, 4375, 4378), ("08:15", 4378, 4381, 4376, 4379)])  # ruptura a la baja y retest
    plan = plan_day(ASIA_HI, ASIA_LO, nym, p)
    assert plan["phase"] == "ready" and plan["side"] == -1 and plan["level"] == ASIA_LO


def test_size_units_min_lot():
    info = SymbolInfo(1, "XAUUSD", digits=2, lot_size=100, min_volume=100, step_volume=100)  # 1 unidad
    p = AsiaParams(risk_pct=0.03, max_risk_pct=0.045)
    assert size_units(500.0, 10.0, info, p) == 1.0  # 15 USD de riesgo: 1 unidad a 10 USD
    assert size_units(500.0, 30.0, info, p) == 0.0  # el lote minimo arriesga 30 > 22,5


class FakeSession:
    def __init__(self, bars_utc, positions):
        self.bars, self._positions, self.calls = bars_utc, positions, []
        self.symbols = {"XAUUSD": SymbolInfo(1, "XAUUSD", digits=2, lot_size=100, min_volume=100, step_volume=100)}
        self.account_id = 1
        self.model = SimpleNamespace(ProtoOAOrderType=SimpleNamespace(MARKET=1), ProtoOATradeSide=SimpleNamespace(BUY=1, SELL=2),
                                     ProtoOAExecutionType=SimpleNamespace(Name=lambda x: "ORDER_ACCEPTED"))

    def trader(self):
        return 500.0, 2, 500.0

    def unrealized_pnl(self):
        return 0.0

    def positions(self, only_bot=True):
        return self._positions

    def deals(self, days=14):
        return []

    def call(self, name, timeout=None, **kw):
        self.calls.append((name, kw))
        if name == "ProtoOAGetTrendbarsReq":
            raise AssertionError("las barras se inyectan por monkeypatch")
        return SimpleNamespace(executionType=1)


def _full_day_bars():
    rows = []
    t = pd.Timestamp("2026-09-20 19:00", tz=NY)  # Asia de la vispera
    px = 4390.0
    while t < pd.Timestamp("2026-09-21 03:00", tz=NY):
        first = not rows
        rows.append((t, px, ASIA_HI if first else px + 3, ASIA_LO if first else px - 3, px)); t += pd.Timedelta(minutes=15)
    for hm, o, h, l, c in [("08:00", 4390, 4398, 4388, 4395), ("08:15", 4395, 4405, 4394, 4403), ("08:30", 4403, 4406, 4399.5, 4402)]:
        rows.append((pd.Timestamp(f"2026-09-21 {hm}", tz=NY), o, h, l, c))
    df = pd.DataFrame([r[1:] for r in rows], index=[r[0] for r in rows], columns=["open", "high", "low", "close"]).assign(volume=1.0)
    return df.tz_convert("UTC")


def test_cycle_enters_on_ready_and_closes_at_noon(tmp_path, monkeypatch):
    import autotrader.asiabot as ab
    bars = _full_day_bars()
    monkeypatch.setattr(ab, "m15_bars", lambda session, symbol="XAUUSD", days=3, now=None: bars)
    s = Settings(broker="sim", symbols=["XAUUSD"], state_dir=str(tmp_path), event_mode="off", risk_per_trade=0.03, max_drawdown_pct=0)
    now = pd.Timestamp("2026-09-21 08:46", tz=NY).tz_convert("UTC").to_pydatetime()
    sess = FakeSession(bars, [])
    state = tmp_path / "asia_state.json"
    summary = run_asia_cycle(s, sess, AsiaParams(risk_pct=0.03, max_risk_pct=0.045), now=now, state_path=str(state))
    assert summary["plan"]["phase"] == "ready"
    orders = [c for c in sess.calls if c[0] == "ProtoOANewOrderReq"]
    assert len(orders) == 1 and orders[0][1]["tradeSide"] == 1 and orders[0][1]["label"] == "autotrader-asia"
    assert orders[0][1]["relativeTakeProfit"] == 2 * orders[0][1]["relativeStopLoss"]
    assert summary["orders"][0]["stop"] == 4390.0 and summary["orders"][0]["risk_usd"] == pytest.approx(12.0, abs=0.01)
    assert summary["publish"] is True
    # segundo ciclo el mismo dia: ya se opero
    sess2 = FakeSession(bars, [])
    summary2 = run_asia_cycle(s, sess2, AsiaParams(risk_pct=0.03), now=now, state_path=str(state))
    assert not [c for c in sess2.calls if c[0] == "ProtoOANewOrderReq"] and "ya se opero hoy" in summary2["skipped"]
    # a las 12:00 NY se cierra lo abierto
    pos = OpenPosition(5, "XAUUSD", 1.0, "buy", 4402.0, 4390.0, "autotrader-asia", None, 4426.0)
    sess3 = FakeSession(bars, [pos])
    noon = pd.Timestamp("2026-09-21 12:01", tz=NY).tz_convert("UTC").to_pydatetime()
    summary3 = run_asia_cycle(s, sess3, AsiaParams(risk_pct=0.03), now=noon, state_path=str(state))
    assert [c[0] for c in sess3.calls if c[0] == "ProtoOAClosePositionReq"] == ["ProtoOAClosePositionReq"]
    assert summary3["closed"][0]["reason"] == "cierre 12:00 NY"


def test_cycle_outside_hours_does_nothing(tmp_path):
    s = Settings(broker="sim", symbols=["XAUUSD"], state_dir=str(tmp_path), event_mode="off", risk_per_trade=0.03, max_drawdown_pct=0)
    sess = FakeSession(pd.DataFrame(), [])
    evening = datetime(2026, 9, 21, 23, 0, tzinfo=timezone.utc)
    summary = run_asia_cycle(s, sess, AsiaParams(), now=evening, state_path=str(tmp_path / "s.json"))
    assert summary["plan"]["phase"] == "fuera de horario" and not [c for c in sess.calls if c[0] == "ProtoOANewOrderReq"]


def test_day_frames_uses_only_closed_bars():
    bars = _full_day_bars()
    now = pd.Timestamp("2026-09-21 08:40", tz=NY).tz_convert("UTC").to_pydatetime()  # la vela de 08:30 no ha cerrado
    _day, asia, nym = day_frames(bars, now, AsiaParams())
    assert len(asia) == 32 and len(nym) == 2
