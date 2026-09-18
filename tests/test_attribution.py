from datetime import datetime, timezone

from autotrader.attribution import bot_metrics, classify, cutoff_for, fifo_trades, parse_ts


def test_classify_manual_excluded_bot():
    cutoff = parse_ts("2026-09-15T00:00Z")
    assert classify(False, "2026-09-16T10:00Z", cutoff) == "manual"
    assert classify(True, "2026-09-14T15:00Z", cutoff) == "excluded"
    assert classify(True, "2026-09-16T10:00Z", cutoff) == "bot"
    assert classify(True, None, cutoff) == "bot"
    assert classify(True, "2026-09-14T15:00Z", None) == "bot"
    assert cutoff_for("alpaca", {"alpaca": {"entries_before": "2026-09-15"}}) == datetime(2026, 9, 15, tzinfo=timezone.utc)
    assert cutoff_for("alpaca", {}) is None


def test_fifo_trades_and_open_lots():
    fills = [
        {"symbol": "GLD", "side": "buy", "qty": 38, "price": 391.76, "at": "2026-09-14 13:35"},
        {"symbol": "GLD", "side": "sell", "qty": 38, "price": 392.86, "at": "2026-09-16 18:33"},
        {"symbol": "GLD", "side": "buy", "qty": 30, "price": 390.67, "at": "2026-09-16 19:35"},
        {"symbol": "NVDA", "side": "buy", "qty": 10, "price": 100.0, "at": "2026-09-15 13:35"},
        {"symbol": "NVDA", "side": "buy", "qty": 10, "price": 110.0, "at": "2026-09-16 13:35"},
        {"symbol": "NVDA", "side": "sell", "qty": 15, "price": 120.0, "at": "2026-09-17 13:35"},
    ]
    trades, lots = fifo_trades(fills)
    assert [(t["symbol"], t["units"], t["net"]) for t in trades] == [("GLD", 38, 41.8), ("NVDA", 10, 200.0), ("NVDA", 5, 50.0)]
    assert sorted((l["symbol"], l["qty"], l["entry"]) for l in lots) == [("GLD", 30, 390.67), ("NVDA", 5, 110.0)]


def test_bot_metrics_adjusted_equity():
    trades = [{"origin": "bot", "net": 50.0}, {"origin": "manual", "net": -326.9}, {"origin": "excluded", "net": -289.5}]
    open_items = [{"origin": "bot", "pnl": 20.0}, {"origin": "excluded", "pnl": 100.0}]
    m = bot_metrics(trades, open_items, 10_000)
    assert m["adjusted_equity"] == 10_070.0 and abs(m["adjusted_return"] - 0.007) < 1e-9
    assert m["realized"]["manual"] == -326.9 and m["open"]["excluded"] == 100.0 and m["counts"]["excluded"] == 1
