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


def test_gap_filter_uses_last_completed_close(tmp_path):
    """Durante la sesion, la barra de hoy es parcial: el hueco se mide contra el cierre anterior."""
    import pandas as pd
    from autotrader.broker import Account

    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    idx = pd.bdate_range(end=today, periods=120)
    close = [100.0 + i * 0.5 for i in range(120)]
    close[-1] = close[-2] * 0.97  # hoy cae un 3 % respecto a ayer
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1e6}, index=idx)

    class Prov:
        def bars_many(self, symbols):
            return {"AAA": df}

        def quote(self, symbol):
            return {"last": close[-1], "at": None, "prev_close": None}

    class FakeBroker:
        name = "fake"
        orders = []

        def account(self, prices=None):
            return Account(cash=100_000, equity=100_000)

        def is_market_open(self):
            return True

        def submit_market_order(self, symbol, qty, side, price_hint=None):
            self.orders.append(symbol)
            return {"id": "x", "status": "accepted"}

    s = Settings(broker="sim", data_provider="yahoo", symbols=["AAA"], fast_sma=10, slow_sma=30,
                 rsi_max_entry=101.0, state_dir=str(tmp_path))
    broker = FakeBroker()
    summary = run_cycle(s, broker, Prov())
    assert broker.orders == []
    assert any("cae" in x for x in summary["skipped"])


def test_exposure_leverage_allows_several_cfd_positions(tmp_path):
    """Con 5x de exposicion y 10k de equity caben varias posiciones de ~5k nominales."""
    from autotrader.broker import Account

    class FakeBroker:
        name = "fake"

        def __init__(self):
            self.orders = []

        def account(self, prices=None):
            return Account(cash=10_000, equity=10_000)

        def is_market_open(self):
            return True

        def qty_step(self, symbol):
            return 1.0

        def submit_market_order(self, symbol, qty, side, price_hint=None):
            self.orders.append((symbol, qty))
            return {"id": "x", "status": "order_accepted"}

    syms = ["A", "B", "C", "D", "E", "F"]
    s = Settings(broker="sim", data_provider="synthetic", symbols=syms, fast_sma=10, slow_sma=30, rsi_max_entry=101.0,
                 max_positions=8, max_position_pct=0.12, exposure_leverage=5.0, stop_loss_pct=0.03, state_dir=str(tmp_path))
    broker = FakeBroker()

    class Prov:
        def bars_many(self, symbols):
            from autotrader.data import synthetic_bars
            import numpy as np, pandas as pd
            out = {}
            for sym in symbols:
                n = 200
                close = np.linspace(4000, 4400, n)
                idx = pd.bdate_range(end=pd.Timestamp.now(tz="UTC").normalize().tz_localize(None), periods=n)
                out[sym] = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1e6}, index=idx)
            return out

        def quote(self, symbol):
            return None

    summary = run_cycle(s, broker, Prov())
    assert len(broker.orders) >= 5, summary["skipped"]
