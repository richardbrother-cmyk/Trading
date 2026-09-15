from datetime import datetime, timezone

from autotrader.events import Event, active_events, load_events, trailed_stop, upcoming_events


def test_load_and_windows():
    evs = load_events()
    assert evs and all(e.at.tzinfo is not None for e in evs)
    fomc = [e for e in evs if "FOMC" in e.name][0]
    before = fomc.at.replace(hour=fomc.at.hour - 2)
    assert active_events(evs, before, hours_before=3, hours_after=1)
    far = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
    assert not active_events(evs, far, hours_before=3, hours_after=1)
    assert upcoming_events(evs, datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc), days=7)


def test_trailed_stop_never_lowers():
    assert trailed_stop(372.17, 393.40, 0.015) == round(393.40 * 0.985, 2)
    assert trailed_stop(400.0, 393.40, 0.015) == 400.0


def test_bot_trails_winners_and_blocks_buys_in_event_window(tmp_path):
    import json
    import numpy as np
    import pandas as pd
    from datetime import timedelta
    from autotrader.bot import run_cycle
    from autotrader.broker import Account, Position
    from autotrader.config import Settings

    now = datetime.now(timezone.utc)
    ev_path = tmp_path / "events.json"
    ev_path.write_text(json.dumps({"events": [{"name": "FOMC test", "at": (now + timedelta(hours=1)).isoformat(), "impact": "high"}]}))

    class Prov:
        def bars_many(self, symbols):
            out = {}
            for sym in symbols:
                close = np.linspace(100, 120, 200)
                idx = pd.bdate_range(end=pd.Timestamp.now(tz="UTC").normalize().tz_localize(None), periods=200)
                out[sym] = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1e6}, index=idx)
            return out

        def quote(self, symbol):
            return None

    class Broker:
        name = "fake"

        def __init__(self):
            self.updates, self.orders = [], []

        def account(self, prices=None):
            return Account(cash=50_000, equity=100_000, positions={"WIN": Position("WIN", 10, 100.0), "LOSE": Position("LOSE", 10, 122.0)})

        def is_market_open(self):
            return True

        def current_stops(self):
            return {"WIN": 95.0, "LOSE": 115.9}

        def update_stop(self, symbol, price):
            self.updates.append((symbol, price))

        def submit_market_order(self, symbol, qty, side, price_hint=None):
            self.orders.append((symbol, side))
            return {"id": "x", "status": "accepted"}

    s = Settings(broker="sim", data_provider="synthetic", symbols=["WIN", "LOSE", "NEW"], fast_sma=10, slow_sma=30,
                 rsi_max_entry=101.0, event_mode="trail", events_path=str(ev_path), state_dir=str(tmp_path))
    b = Broker()
    summary = run_cycle(s, b, Prov())
    assert summary["event_window"]
    assert b.updates == [("WIN", round(120 * 0.985, 2))]  # gana +20 %: stop sube; LOSE pierde: no se toca
    assert not any(sym == "NEW" for sym, _ in b.orders)  # sin entradas nuevas
    assert any("ventana de evento" in x for x in summary["skipped"])

    s.event_mode = "close"
    b2 = Broker()
    run_cycle(s, b2, Prov())
    assert ("WIN", "sell") in b2.orders and ("LOSE", "sell") not in b2.orders
