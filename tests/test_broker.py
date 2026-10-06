import pytest

from autotrader.broker import AlpacaBroker, SimulatedBroker
from autotrader.config import PAPER_URL, ConfigError, Settings


def test_sim_broker_roundtrip(tmp_path):
    b = SimulatedBroker(initial_cash=1000, state_dir=str(tmp_path))
    b.submit_market_order("XYZ", 5, "buy", price_hint=100)
    assert b.account().cash == 500
    assert b.account({"XYZ": 120}).equity == 500 + 5 * 120
    b.submit_market_order("XYZ", 5, "sell", price_hint=120)
    assert b.account().cash == 1100 and not b.positions
    # persistencia
    b2 = SimulatedBroker(initial_cash=0, state_dir=str(tmp_path))
    assert b2.cash == 1100


def test_sim_broker_rejects_overspend(tmp_path):
    b = SimulatedBroker(initial_cash=100, state_dir=str(tmp_path))
    with pytest.raises(ValueError):
        b.submit_market_order("XYZ", 5, "buy", price_hint=100)


def test_alpaca_broker_refuses_live_url():
    with pytest.raises(ValueError):
        AlpacaBroker("k", "s", base_url="https://api.alpaca.markets")
    AlpacaBroker("k", "s", base_url=PAPER_URL)  # paper permitido


def test_settings_refuse_live_url(monkeypatch):
    monkeypatch.setenv("BROKER", "alpaca")
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://api.alpaca.markets")
    with pytest.raises(ConfigError):
        Settings.from_env(dotenv_path="/nonexistent")


def test_settings_defaults(monkeypatch):
    for k in ["BROKER", "DATA_PROVIDER", "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_BASE_URL", "SYMBOLS",
              "CTRADER_CLIENT_ID", "CTRADER_CLIENT_SECRET", "CTRADER_DEMO", "MAX_POSITIONS", "MAX_POSITION_PCT"]:
        monkeypatch.delenv(k, raising=False)
    s = Settings.from_env(dotenv_path="/nonexistent")
    assert s.broker == "sim" and s.symbols[0] == "SPY"


class _FakeResp:
    def __init__(self, data, status=200):
        self._data, self.status_code, self.text = data, status, str(data)

    def json(self):
        return self._data

    def raise_for_status(self):
        pass


class _FakeSession:
    def __init__(self, open_orders):
        self.headers = {}
        self.posted, self.deleted = [], []
        self.open_orders = open_orders

    def get(self, url, params=None, timeout=None):
        return _FakeResp(self.open_orders if "/v2/orders" in url else {})

    def post(self, url, json=None, timeout=None):
        self.posted.append(json)
        return _FakeResp({"id": "o1", "status": "accepted", **json})

    def delete(self, url, timeout=None):
        self.deleted.append(url)
        # como en Alpaca, la orden cancelada deja de aparecer entre las abiertas
        self.open_orders = [o for o in self.open_orders if not url.endswith(f"/v2/orders/{o.get('id')}")]
        return _FakeResp({}, 204)


def test_alpaca_buy_attaches_oto_stop():
    sess = _FakeSession([])
    b = AlpacaBroker("k", "s", session=sess, stop_loss_pct=0.05)
    b.submit_market_order("GLD", 10, "buy", price_hint=400.0)
    p = sess.posted[0]
    assert p["order_class"] == "oto" and p["stop_loss"] == {"stop_price": "380.00"} and p["type"] == "market"
    assert p["time_in_force"] == "gtc"  # el stop hereda la vigencia: "day" caducaria al cierre


def test_alpaca_sell_cancels_open_orders_first():
    sess = _FakeSession([{"id": "stop1", "symbol": "GLD", "side": "sell", "type": "stop"}])
    b = AlpacaBroker("k", "s", session=sess, stop_loss_pct=0.05)
    b.submit_market_order("GLD", 10, "sell", price_hint=390.0)
    assert sess.deleted == [f"{PAPER_URL}/v2/orders/stop1"]
    assert "order_class" not in sess.posted[0] and sess.posted[0]["side"] == "sell"


def test_ensure_stops_places_missing_gtc_stops():
    class Sess(_FakeSession):
        def get(self, url, params=None, timeout=None):
            if "/v2/positions" in url:
                return _FakeResp([{"symbol": "GLD", "qty": "38", "avg_entry_price": "391.76"},
                                  {"symbol": "SPY", "qty": "19", "avg_entry_price": "758.79"}])
            return _FakeResp(self.open_orders)

    sess = Sess([{"id": "s1", "symbol": "SPY", "side": "sell", "type": "stop"}])  # SPY ya tiene stop
    b = AlpacaBroker("k", "s", session=sess, stop_loss_pct=0.05)
    placed = b.ensure_stops()
    assert [p["symbol"] for p in placed] == ["GLD"]
    assert sess.posted[0]["type"] == "stop" and sess.posted[0]["time_in_force"] == "gtc" and sess.posted[0]["stop_price"] == "372.17"


def test_alpaca_sell_waits_for_pending_cancel(monkeypatch):
    """La cancelacion del stop es asincrona: no se vende hasta que el simbolo no tiene ordenes abiertas."""
    import autotrader.broker as broker_mod

    class Sess(_FakeSession):
        def __init__(self):
            super().__init__([{"id": "stop1", "symbol": "GLD", "side": "sell", "type": "stop"}])
            self.polls = 0

        def delete(self, url, timeout=None):
            self.deleted.append(url)  # queda en pending_cancel: sigue apareciendo como abierta un rato
            return _FakeResp({}, 204)

        def get(self, url, params=None, timeout=None):
            if "/v2/orders" in url:
                self.polls += 1
                # la orden sigue abierta (pending_cancel) en las dos primeras consultas tras el delete
                return _FakeResp(self.open_orders if self.polls <= 3 else [])
            return _FakeResp({})

    sleeps = []
    monkeypatch.setattr(broker_mod.time, "sleep", lambda s: sleeps.append(s))
    sess = Sess()
    b = AlpacaBroker("k", "s", session=sess, stop_loss_pct=0.05)
    b.submit_market_order("GLD", 10, "sell", price_hint=390.0)
    assert sess.deleted == [f"{PAPER_URL}/v2/orders/stop1"]
    assert sleeps and sess.posted[0]["side"] == "sell"  # espero antes de vender


def test_alpaca_sell_retries_once_when_qty_still_held(monkeypatch):
    import autotrader.broker as broker_mod

    class Sess(_FakeSession):
        def __init__(self):
            super().__init__([])
            self.attempts = 0

        def post(self, url, json=None, timeout=None):
            self.attempts += 1
            self.posted.append(json)
            if self.attempts == 1:
                return _FakeResp({"code": 40310000, "message": "insufficient qty available for order"}, 403)
            return _FakeResp({"id": "o2", "status": "accepted", **json})

    monkeypatch.setattr(broker_mod.time, "sleep", lambda s: None)
    sess = Sess()
    b = AlpacaBroker("k", "s", session=sess, stop_loss_pct=0.05)
    order = b.submit_market_order("GLD", 10, "sell", price_hint=390.0)
    assert sess.attempts == 2 and order["status"] == "accepted"
