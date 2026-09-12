from autotrader.bot import run_cycle
from autotrader.broker import SimulatedBroker
from autotrader.config import Settings
from autotrader.data import DataProvider


def test_run_cycle_sim_synthetic(tmp_path):
    s = Settings(broker="sim", data_provider="synthetic", symbols=["AAA", "BBB", "CCC", "DDD"],
                 fast_sma=10, slow_sma=30, state_dir=str(tmp_path))
    broker = SimulatedBroker(initial_cash=s.initial_cash, state_dir=s.state_dir)
    summary = run_cycle(s, broker, DataProvider("synthetic", days=300), force=True)
    assert len(summary["decisions"]) == 4
    acct = broker.account()
    assert acct.cash <= s.initial_cash
    assert (tmp_path / "run_log.jsonl").exists()
    # segundo ciclo: no debe recomprar lo que ya tiene
    before = {k: v.qty for k, v in broker.positions.items()}
    summary2 = run_cycle(s, broker, DataProvider("synthetic", days=300), force=True)
    bought_again = [o for o in summary2["orders"] if o["side"] == "buy" and o["symbol"] in before]
    assert not bought_again


def test_run_cycle_skips_symbols_with_pending_buy(tmp_path):
    from autotrader.broker import Account

    class FakeBroker:
        name = "fake"

        def __init__(self):
            self.orders = []

        def account(self, prices=None):
            return Account(cash=100_000, equity=100_000, positions={}, pending_buys={"AAA", "BBB", "CCC", "DDD"})

        def is_market_open(self):
            return True

        def submit_market_order(self, symbol, qty, side, price_hint=None):
            self.orders.append(symbol)
            return {"id": "x", "status": "accepted"}

    s = Settings(broker="sim", data_provider="synthetic", symbols=["AAA", "BBB", "CCC", "DDD"],
                 fast_sma=10, slow_sma=30, state_dir=str(tmp_path))
    broker = FakeBroker()
    summary = run_cycle(s, broker, DataProvider("synthetic", days=300))
    assert broker.orders == []
    assert len(summary["skipped"]) == 4
