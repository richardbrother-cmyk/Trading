import json

import pandas as pd
import pytest

from autotrader.broker import Account, AlpacaBroker, Position
from autotrader.config import Settings
from autotrader.gem import GemParams, gem_signal, run_gem, tbill_index


def _series(values, start="2024-01-31"):
    idx = pd.date_range(start, periods=len(values), freq="ME")
    return pd.Series(values, index=idx, dtype=float)


def _flat_tbill(n=14, rate=4.0):
    return tbill_index(_series([rate] * n))


def test_signal_picks_us_when_us_beats_intl_and_tbill():
    us = _series([100 + i * 2 for i in range(14)])  # +24 % en 12 meses
    intl = _series([100 + i for i in range(14)])  # +12 %
    sig = gem_signal(us, intl, _flat_tbill())
    assert sig["choice"] == "us"
    assert sig["momentum"]["us"] > sig["momentum"]["intl"] > sig["momentum"]["tbill"] > 0


def test_signal_picks_intl_when_it_leads():
    us = _series([100 + i for i in range(14)])
    intl = _series([100 + i * 3 for i in range(14)])
    assert gem_signal(us, intl, _flat_tbill())["choice"] == "intl"


def test_signal_goes_to_bonds_when_us_below_tbills():
    us = _series([100 - i for i in range(14)])  # negativo
    intl = _series([100 + i * 3 for i in range(14)])  # aunque internacional suba mucho
    sig = gem_signal(us, intl, _flat_tbill())
    assert sig["choice"] == "bond" and "absoluto" in sig["reason"]


def test_signal_needs_enough_history():
    with pytest.raises(ValueError):
        gem_signal(_series([1] * 5), _series([1] * 5), _flat_tbill(5))


class FakeBroker:
    name = "fake"

    def __init__(self, positions=None, cash=50_000.0, open_=True):
        self.positions = positions or {}
        self.cash = cash
        self.open_ = open_
        self.orders = []

    def account(self, prices=None):
        return Account(cash=self.cash, equity=100_000.0, positions=dict(self.positions), pending_buys=set())

    def is_market_open(self):
        return self.open_

    def submit_market_order(self, symbol, qty, side, price_hint=None):
        self.orders.append((symbol, qty, side))
        return {"status": "accepted"}


def _settings(tmp_path):
    s = Settings()
    s.state_dir = str(tmp_path)
    return s


def _params(tmp_path):
    return GemParams(notional=20_000.0, state_path=str(tmp_path / "gem_state.json"))


SIGNAL_US = {"as_of": "2026-09-30", "choice": "us", "reason": "x", "momentum": {"us": 0.15, "intl": 0.1, "tbill": 0.04}}
PRICES = {"IVV": 500.0, "VEU": 80.0, "AGG": 95.0}


def test_first_run_buys_target_with_notional(tmp_path):
    broker = FakeBroker()
    out = run_gem(_settings(tmp_path), broker, _params(tmp_path), PRICES, SIGNAL_US)
    assert broker.orders == [("IVV", 40, "buy")]  # 20.000 / 500
    state = json.loads((tmp_path / "gem_state.json").read_text())
    assert state["holding"] == "IVV" and state["capital"] == 20_000.0 and state["history"][-1]["choice"] == "us"
    assert out["orders"][0]["status"] == "accepted"


def test_switch_sells_old_and_buys_new_with_proceeds(tmp_path):
    broker = FakeBroker(positions={"VEU": Position("VEU", 250, 78.0), "SPY": Position("SPY", 10, 600.0)})
    out = run_gem(_settings(tmp_path), broker, _params(tmp_path), PRICES, SIGNAL_US)
    assert broker.orders == [("VEU", 250, "sell"), ("IVV", 40, "buy")]  # 250 * 80 = 20.000 -> 40 IVV
    assert "SPY" not in str(broker.orders)  # nunca toca posiciones ajenas
    assert out["capital"] == 20_000.0


def test_already_in_target_does_nothing(tmp_path):
    broker = FakeBroker(positions={"IVV": Position("IVV", 40, 490.0)})
    out = run_gem(_settings(tmp_path), broker, _params(tmp_path), PRICES, SIGNAL_US)
    assert broker.orders == [] and any("sin cambios" in s for s in out["skipped"])
    assert out["gem_equity"] == 20_000.0


def test_dry_run_sends_nothing_and_saves_nothing(tmp_path):
    broker = FakeBroker(positions={"AGG": Position("AGG", 200, 94.0)})
    out = run_gem(_settings(tmp_path), broker, _params(tmp_path), PRICES, SIGNAL_US, dry_run=True)
    assert broker.orders == [] and [o["status"] for o in out["orders"]] == ["dry_run", "dry_run"]
    assert not (tmp_path / "gem_state.json").exists()


def test_buy_is_capped_by_available_cash(tmp_path):
    broker = FakeBroker(cash=7_900.0)  # el bot SMA tiene el resto invertido: sin margen
    out = run_gem(_settings(tmp_path), broker, _params(tmp_path), PRICES, SIGNAL_US)
    assert broker.orders == [("IVV", 15, "buy")]  # 7.900 / 500
    assert any("efectivo limitado" in x for x in out["skipped"]) and out["capital"] == 7_500.0


def test_switch_counts_sale_proceeds_as_cash(tmp_path):
    broker = FakeBroker(cash=100.0, positions={"VEU": Position("VEU", 250, 78.0)})
    run_gem(_settings(tmp_path), broker, _params(tmp_path), PRICES, SIGNAL_US)
    assert broker.orders == [("VEU", 250, "sell"), ("IVV", 40, "buy")]


def test_market_closed_skips_orders(tmp_path):
    broker = FakeBroker(open_=False)
    out = run_gem(_settings(tmp_path), broker, _params(tmp_path), PRICES, SIGNAL_US)
    assert broker.orders == [] and "mercado cerrado" in out["skipped"]


def test_halt_freeze_blocks_buy_but_allows_switch_out(tmp_path):
    s = _settings(tmp_path)
    s.halt_mode = "freeze"
    broker = FakeBroker(positions={"VEU": Position("VEU", 250, 78.0)})
    out = run_gem(s, broker, _params(tmp_path), PRICES, SIGNAL_US)
    assert broker.orders == [("VEU", 250, "sell")] and any("freno" in x for x in out["skipped"])


def test_alpaca_ensure_stops_only_covers_bot_universe():
    class _Resp:
        def __init__(self, data):
            self._data, self.status_code, self.text = data, 200, ""

        def json(self):
            return self._data

        def raise_for_status(self):
            pass

    class Sess:
        def __init__(self):
            self.posted = []
            self.headers = {}

        def get(self, url, params=None, timeout=None):
            if "/v2/positions" in url:
                return _Resp([{"symbol": "GLD", "qty": "38", "avg_entry_price": "391.76"},
                              {"symbol": "IVV", "qty": "40", "avg_entry_price": "500.00"}])
            return _Resp([])

        def post(self, url, json=None, timeout=None):
            self.posted.append(json)
            return _Resp({"status": "accepted"})

    sess = Sess()
    b = AlpacaBroker("k", "s", session=sess, stop_loss_pct=0.05)
    placed = b.ensure_stops(only={"GLD", "SPY"})
    assert [p["symbol"] for p in placed] == ["GLD"]  # IVV (GEM) queda sin stop
