from datetime import datetime, timedelta, timezone

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


def test_opex_events_third_friday_and_freeze_mode():
    from datetime import date
    from autotrader.events import effective_mode, opex_events, third_friday
    assert third_friday(2026, 9) == date(2026, 9, 18) and third_friday(2026, 10) == date(2026, 10, 16)
    evs = opex_events(date(2026, 9, 1), months=4)
    sep = evs[0]
    assert sep.at == datetime(2026, 9, 18, 20, 0, tzinfo=timezone.utc) and sep.mode == "freeze" and "quarterly" in sep.tags
    dec = [e for e in evs if e.at.month == 12][0]
    assert dec.at == datetime(2026, 12, 18, 21, 0, tzinfo=timezone.utc)  # horario de invierno: 16:00 ET = 21:00 UTC
    # ventana propia: 2 h antes, 30 min despues
    assert active_events(evs, datetime(2026, 9, 18, 18, 30, tzinfo=timezone.utc), hours_before=3, hours_after=1) == [sep]
    assert not active_events(evs, datetime(2026, 9, 18, 17, 30, tzinfo=timezone.utc), hours_before=3, hours_after=1)
    assert effective_mode([sep], "trail") == "freeze"
    assert effective_mode([sep, Event("FOMC", sep.at)], "trail") == "trail"
    loaded = load_events(now=datetime(2026, 9, 18, tzinfo=timezone.utc))
    assert any(e.is_opex for e in loaded) and any("FOMC" in e.name for e in loaded)


def test_bot_freezes_without_trailing_on_opex(tmp_path):
    import json
    import numpy as np
    import pandas as pd
    from autotrader.bot import run_cycle
    from autotrader.broker import Account, Position
    from autotrader.config import Settings

    now = datetime.now(timezone.utc)
    ev_path = tmp_path / "events.json"
    ev_path.write_text(json.dumps({"opex": False, "events": [{"name": "OPEX test", "at": (now + timedelta(hours=1)).isoformat(),
                                                                "impact": "high", "tags": ["opex"], "mode": "freeze", "hours_before": 2, "hours_after": 0.5}]}))

    class Prov:
        def bars_many(self, symbols):
            close = np.linspace(100, 120, 200)
            idx = pd.bdate_range(end=pd.Timestamp.now(tz="UTC").normalize().tz_localize(None), periods=200)
            df = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1}, index=idx)
            return {s: df for s in symbols}

        def quote(self, symbol):
            return None

    class Brk:
        name = "sim-test"
        trailed = []

        def account(self, prices=None):
            return Account(cash=1000.0, equity=13400.0, positions={"WIN": Position("WIN", 100, 100.0)})

        def is_market_open(self):
            return True

        def current_stops(self):
            return {"WIN": 95.0}

        def update_stop(self, symbol, stop):
            self.trailed.append((symbol, stop))

        def submit_market_order(self, *a, **k):
            raise AssertionError("no debe operar")

    s = Settings(broker="sim", symbols=["WIN", "NEW"], state_dir=str(tmp_path), events_path=str(ev_path), event_mode="trail", max_drawdown_pct=0)
    brk = Brk()
    summary = run_cycle(s, brk, Prov(), dry_run=False)
    assert summary["event_mode"] == "freeze" and not brk.trailed and summary["protection"] == []
    assert not summary["orders"] and summary["event_window"]
