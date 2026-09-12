from autotrader.risk import RiskParams, daily_loss_breached, position_size, stop_hit


def test_position_size_respects_all_caps():
    p = RiskParams(risk_per_trade=0.02, max_position_pct=0.25, stop_loss_pct=0.05)
    # riesgo: 2000 / (100*0.05) = 400; cap: 25000/100 = 250; cash: 30000/100 = 300 -> 250
    assert position_size(100_000, 30_000, 100, p) == 250
    assert position_size(100_000, 5_000, 100, p) == 50
    assert position_size(100_000, 30_000, 0, p) == 0


def test_daily_loss_and_stop():
    p = RiskParams(max_daily_loss_pct=0.03, stop_loss_pct=0.05)
    assert daily_loss_breached(96_999, 100_000, p)
    assert not daily_loss_breached(97_500, 100_000, p)
    assert stop_hit(100, 95, p) and not stop_hit(100, 95.5, p)
