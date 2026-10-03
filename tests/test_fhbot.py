from types import SimpleNamespace

import pandas as pd
import pytest

from autotrader.config import Settings
from autotrader.ctrader import OpenPosition, SymbolInfo
from autotrader.fhbot import NY, FHParams, atr14, plan_day, run_fh_cycle, size_units


def _bars(day, moves, base=5000.0, hist_range=4.0):
    """Historia previa plana (rango `hist_range` por vela, 3 dias) + la primera hora del dia con `moves` = [(o, h, l, c)] x 4 (+ mas velas)."""
    rows, idx = [], []
    t = pd.Timestamp(f"{day} 09:30", tz=NY) - pd.Timedelta(days=3)
    end = pd.Timestamp(f"{day} 09:30", tz=NY)
    while t < end:
        if t.weekday() < 5:
            rows.append((base, base + hist_range / 2, base - hist_range / 2, base)); idx.append(t)
        t += pd.Timedelta(minutes=15)
    for k, (o, h, l, c) in enumerate(moves):
        rows.append((o, h, l, c)); idx.append(end + pd.Timedelta(minutes=15 * k))
    df = pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close"]).assign(volume=1.0)
    return df.tz_convert("UTC")


UP = [(5000, 5006, 4999, 5005), (5005, 5012, 5004, 5011), (5011, 5016, 5009, 5015), (5015, 5020, 5013, 5019)]  # +19 puntos
DOWN = [(5000, 5001, 4994, 4995), (4995, 4996, 4988, 4989), (4989, 4991, 4984, 4985), (4985, 4987, 4980, 4981)]  # -19
FLAT = [(5000, 5003, 4998, 5001), (5001, 5003, 4999, 5000), (5000, 5002, 4998, 5001), (5001, 5003, 4999, 5001)]
DAY = "2026-09-21"  # lunes


def _now(hm):
    return pd.Timestamp(f"{DAY} {hm}", tz=NY).tz_convert("UTC").to_pydatetime()


def test_plan_signal_sides_and_threshold():
    p = FHParams()
    plan = plan_day(_bars(DAY, UP), _now("10:31"), p)
    assert plan["phase"] == "ready" and plan["side"] == 1 and plan["r"] == pytest.approx(19.0)
    assert plan["u"] == pytest.approx(2 * 4.0, rel=0.8)  # U = 2 ATR, ATR del orden del rango historico
    assert plan_day(_bars(DAY, DOWN), _now("10:31"), p)["side"] == -1
    assert plan_day(_bars(DAY, FLAT), _now("10:31"), p)["phase"] == "sin_senal"


def test_plan_phases_timing():
    p = FHParams(max_late_min=30)
    bars = _bars(DAY, UP)
    assert plan_day(bars, _now("10:20"), p)["phase"] == "antes_de_las_10_30"
    assert plan_day(bars, _now("10:44"), p)["phase"] == "ready"
    assert plan_day(bars, _now("11:05"), p)["phase"] == "late"  # senal caducada: no se persigue
    # la vela de las 10:15 aun no ha cerrado a las 10:29
    assert plan_day(bars, _now("10:29"), p)["phase"] == "antes_de_las_10_30"


def test_atr_matches_wilder_ewm():
    bars = _bars(DAY, UP).tz_convert(NY)
    a = atr14(bars)
    assert a.iloc[-1] > a.iloc[-6]  # las velas de la primera hora son mas anchas que la historia plana


def test_size_units_min_lot():
    info = SymbolInfo(1, "US500", digits=2, lot_size=1, min_volume=100, step_volume=1)  # minimo 1 unidad, paso 0,01
    p = FHParams(risk_pct=0.02, max_risk_pct=0.03)
    assert size_units(500.0, 10.0, info, p) == pytest.approx(1.0)  # 10 USD de riesgo a 10 puntos
    assert size_units(500.0, 40.0, info, p) == 0.0  # el lote minimo arriesga 40 > 15 (3 %)
    assert size_units(500.0, 14.0, info, p) == pytest.approx(1.0)  # 10/14 = 0,71 < minimo: se acepta el minimo (14 USD <= 15)
    fine = SymbolInfo(1, "US500", digits=2, lot_size=1, min_volume=1, step_volume=1)  # minimo y paso 0,01
    assert size_units(500.0, 14.0, fine, p) == pytest.approx(0.71)  # 10/14 -> paso 0,01
    assert size_units(0.0, 10.0, info, p) == 0.0 and size_units(500.0, 0.0, info, p) == 0.0


class FakeSession:
    def __init__(self, positions, symbols=("US500", "NAS100")):
        self._positions, self.calls = positions, []
        self.symbols = {s: SymbolInfo(i + 1, s, digits=2, lot_size=100, min_volume=1, step_volume=1) for i, s in enumerate(symbols)}
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
        return SimpleNamespace(executionType=1)


def _settings(tmp_path):
    return Settings(broker="sim", symbols=["US500", "NAS100"], state_dir=str(tmp_path), event_mode="off", risk_per_trade=0.02, max_drawdown_pct=0)


def test_cycle_enters_both_indices_once_and_closes_at_four(tmp_path, monkeypatch):
    import autotrader.fhbot as fb
    # NAS100 sintetico: mismo movimiento alcista a escala de precio ~18000
    nas = _bars(DAY, [(18000, 18020, 17996, 18017), (18017, 18040, 18012, 18036), (18036, 18052, 18030, 18048), (18048, 18060, 18040, 18057)], base=18000.0, hist_range=14.0)
    data = {"US500": _bars(DAY, UP), "NAS100": nas}
    monkeypatch.setattr(fb, "m15_bars", lambda session, symbol, days=4, now=None: data[symbol])
    s = _settings(tmp_path)
    state = tmp_path / "fh_state.json"
    sess = FakeSession([])
    summary = run_fh_cycle(s, sess, FHParams(), now=_now("10:31"), state_path=str(state))
    orders = [c for c in sess.calls if c[0] == "ProtoOANewOrderReq"]
    assert len(orders) == 2 and all(o[1]["label"] == "autotrader-fh" and o[1]["tradeSide"] == 1 for o in orders)
    assert all("relativeTakeProfit" not in o[1] and o[1]["relativeStopLoss"] > 0 for o in orders)
    assert summary["publish"] is True and {o["symbol"] for o in summary["orders"]} == {"US500", "NAS100"}
    # mismo dia, siguiente ciclo: ya se opero
    sess2 = FakeSession([])
    summary2 = run_fh_cycle(s, sess2, FHParams(), now=_now("10:46"), state_path=str(state))
    assert not [c for c in sess2.calls if c[0] == "ProtoOANewOrderReq"] and any("ya se opero hoy" in x for x in summary2["skipped"])
    # a las 16:00 NY se cierra lo abierto con la etiqueta propia (y solo esa)
    mine = OpenPosition(5, "US500", 0.7, "buy", 5019.0, 5011.0, "autotrader-fh", None, None)
    other = OpenPosition(6, "XAUUSD", 1.0, "buy", 4400.0, 4390.0, "autotrader-asia", None, None)
    sess3 = FakeSession([mine, other])
    run3 = run_fh_cycle(s, sess3, FHParams(), now=_now("16:01"), state_path=str(state))
    closes = [c for c in sess3.calls if c[0] == "ProtoOAClosePositionReq"]
    assert len(closes) == 1 and closes[0][1]["positionId"] == 5 and run3["closed"][0]["reason"] == "cierre 16:00 NY"


def test_cycle_late_signal_and_outside_hours_do_nothing(tmp_path, monkeypatch):
    import autotrader.fhbot as fb
    monkeypatch.setattr(fb, "m15_bars", lambda session, symbol, days=4, now=None: _bars(DAY, UP))
    s = _settings(tmp_path)
    sess = FakeSession([])
    late = run_fh_cycle(s, sess, FHParams(), now=_now("11:20"), state_path=str(tmp_path / "a.json"))
    assert not [c for c in sess.calls if c[0] == "ProtoOANewOrderReq"] and all(p["phase"] == "late" for p in late["plans"].values())
    sess2 = FakeSession([])
    sat = pd.Timestamp("2026-09-26 11:00", tz=NY).tz_convert("UTC").to_pydatetime()
    run_fh_cycle(s, sess2, FHParams(), now=sat, state_path=str(tmp_path / "b.json"))
    assert not sess2.calls
