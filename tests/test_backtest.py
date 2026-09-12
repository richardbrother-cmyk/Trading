from autotrader.backtest import run_backtest
from autotrader.data import synthetic_bars
from autotrader.risk import RiskParams
from autotrader.strategy import StrategyParams


def test_backtest_runs_and_metrics_consistent():
    data = {s: synthetic_bars(s, days=400, seed=i) for i, s in enumerate(["AAA", "BBB", "CCC"])}
    res = run_backtest(data, StrategyParams(10, 30), RiskParams(), initial_cash=50_000)
    m = res.metrics()
    assert m["initial_cash"] == 50_000
    assert abs(m["final_equity"] / 50_000 - 1 - m["total_return"]) < 1e-3
    assert -1 <= m["max_drawdown"] <= 0
    assert m["trades"] == len(res.trades)
    assert all(t.exit_price is not None for t in res.trades)


def test_backtest_respects_max_positions_and_cash():
    data = {f"S{i}": synthetic_bars(f"S{i}", days=300, seed=100 + i) for i in range(8)}
    res = run_backtest(data, StrategyParams(5, 20), RiskParams(max_positions=2, max_position_pct=0.5), initial_cash=10_000)
    # nunca mas de 2 posiciones abiertas simultaneamente
    open_by_day = {}
    for t in res.trades:
        for d in res.equity.index:
            if t.entry_date <= d < t.exit_date:
                open_by_day[d] = open_by_day.get(d, 0) + 1
    assert max(open_by_day.values(), default=0) <= 2
    assert (res.equity > 0).all()
