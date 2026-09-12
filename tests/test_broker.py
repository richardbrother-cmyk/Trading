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
    for k in ["BROKER", "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_BASE_URL", "SYMBOLS"]:
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
        return _FakeResp({}, 204)


def test_alpaca_buy_attaches_oto_stop():
    sess = _FakeSession([])
    b = AlpacaBroker("k", "s", session=sess, stop_loss_pct=0.05)
    b.submit_market_order("GLD", 10, "buy", price_hint=400.0)
    p = sess.posted[0]
    assert p["order_class"] == "oto" and p["stop_loss"] == {"stop_price": "380.00"} and p["type"] == "market"


def test_alpaca_sell_cancels_open_orders_first():
    sess = _FakeSession([{"id": "stop1", "symbol": "GLD", "side": "sell", "type": "stop"}])
    b = AlpacaBroker("k", "s", session=sess, stop_loss_pct=0.05)
    b.submit_market_order("GLD", 10, "sell", price_hint=390.0)
    assert sess.deleted == [f"{PAPER_URL}/v2/orders/stop1"]
    assert "order_class" not in sess.posted[0] and sess.posted[0]["side"] == "sell"
