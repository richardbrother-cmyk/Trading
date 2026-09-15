from datetime import datetime, timezone
from types import SimpleNamespace

from autotrader.ctrader import cfd_market_open, decode_trendbars, money, relative_stop, round_volume
from autotrader.risk import RiskParams, position_size


def test_decode_trendbars_prices_and_dates():
    bars = [
        SimpleNamespace(low=405000000, deltaOpen=100000, deltaHigh=300000, deltaClose=200000, volume=10, utcTimestampInMinutes=29_000_000),
        SimpleNamespace(low=406000000, deltaOpen=0, deltaHigh=50000, deltaClose=20000, volume=12, utcTimestampInMinutes=29_001_440),
    ]
    df = decode_trendbars(bars)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.iloc[0]["low"] == 4050.0 and df.iloc[0]["open"] == 4051.0 and df.iloc[0]["high"] == 4053.0 and df.iloc[0]["close"] == 4052.0
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert df.index.is_monotonic_increasing and df.index[0].hour == 0


def test_round_volume_respects_min_and_step():
    # minimo 0.01 lote de 100 oz = 1 unidad -> 100 en protocolo; paso 100
    assert round_volume(1.0, 100, 100) == 100
    assert round_volume(2.57, 100, 100) == 200
    assert round_volume(0.5, 100, 100) == 0
    assert round_volume(1000, 100, 100, max_volume=50000) == 50000


def test_money_and_market_hours():
    assert money(1000000, 2) == 10000.0
    assert cfd_market_open(datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc))  # lunes
    assert not cfd_market_open(datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc))  # sabado
    assert not cfd_market_open(datetime(2026, 9, 11, 21, 30, tzinfo=timezone.utc))  # viernes noche
    assert cfd_market_open(datetime(2026, 9, 13, 22, 30, tzinfo=timezone.utc))  # domingo noche


def test_position_size_with_leverage_keeps_risk_constant():
    p1 = RiskParams(risk_per_trade=0.02, max_position_pct=0.15, stop_loss_pct=0.05, exposure_leverage=1)
    p3 = RiskParams(risk_per_trade=0.02, max_position_pct=0.15, stop_loss_pct=0.05, exposure_leverage=3)
    # equity 10k, oro a 4000: por riesgo 200/(4000*0.05) = 1 oz; por tope 1x: 1500/4000 = 0.375 oz; 3x: 1.125 oz
    assert position_size(10_000, 10_000, 4000, p1, step=0.01) == 0.37
    assert position_size(10_000, 10_000, 4000, p3, step=0.01) == 1.0  # el riesgo (2 %) sigue mandando
    assert position_size(10_000, 10_000, 100, p1) == 15  # acciones enteras por defecto


def test_relative_stop_matches_symbol_precision():
    # XAUUSD (2 decimales): el stop debe ser multiplo de 0.01 -> multiplo de 1000 en 1e-5
    assert relative_stop(4348.67, 0.03, 2) % 1000 == 0
    assert abs(relative_stop(4348.67, 0.03, 2) / 100000 - 130.46) < 1e-6
    # XAGUSD (3 decimales): multiplo de 100
    assert relative_stop(64.477, 0.03, 3) % 100 == 0
    # indices con 2 decimales y precio grande
    assert relative_stop(29369.88, 0.03, 2) % 1000 == 0
    # nunca cero
    assert relative_stop(0.0001, 0.03, 5) == 1


def test_ctrader_ensure_stops_amends_only_unprotected_positions():
    from autotrader.ctrader import CTraderBroker, OpenPosition, SymbolInfo

    class Sess:
        demo = True
        symbols = {"COFARA": SymbolInfo(1, "COFARA", digits=2), "WHEAT": SymbolInfo(2, "WHEAT", digits=2)}
        amended = []

        def load_symbols(self, names):
            return self.symbols

        def positions(self):
            return [OpenPosition(11, "COFARA", 80, "buy", 298.46, 0.0), OpenPosition(12, "WHEAT", 8, "buy", 702.9, 681.68)]

        def amend_stop(self, pid, price):
            self.amended.append((pid, price))

    b = CTraderBroker.__new__(CTraderBroker)
    b.session = Sess(); b.stop_loss_pct = 0.03; b.name = "ctrader-demo"
    placed = b.ensure_stops()
    assert [p["symbol"] for p in placed] == ["COFARA"]
    assert b.session.amended == [(11, round(298.46 * 0.97, 2))]


def test_positions_filter_by_bot_label():
    """Solo se gestionan las posiciones con la etiqueta del bot; las manuales se ignoran salvo que se pidan."""
    from types import SimpleNamespace
    from autotrader.ctrader import BOT_LABEL, CTraderSession, SymbolInfo

    class Model:
        class ProtoOAPositionStatus: POSITION_STATUS_OPEN = 1
        class ProtoOATradeSide: BUY = 1
    def td(symbol_id, volume, label):
        t = SimpleNamespace(symbolId=symbol_id, volume=volume, tradeSide=1, label=label)
        t.HasField = lambda f: f == "label" and label != ""
        return t
    def pos(pid, symbol_id, volume, label):
        p = SimpleNamespace(positionId=pid, positionStatus=1, tradeData=td(symbol_id, volume, label), price=300.0, stopLoss=0.0)
        p.HasField = lambda f: False
        return p
    sess = CTraderSession.__new__(CTraderSession)
    sess.model = Model; sess.account_id = 1
    sess.symbols = {"COFARA": SymbolInfo(7, "COFARA")}
    sess.call = lambda *a, **k: SimpleNamespace(position=[pos(1, 7, 2000, BOT_LABEL), pos(2, 7, 8000, "")])
    own = sess.positions()
    assert [p.position_id for p in own] == [1] and own[0].is_bot
    assert len(sess.positions(only_bot=False)) == 2
