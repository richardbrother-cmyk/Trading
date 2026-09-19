import json

from autotrader.guard import evaluate, peak_equity


def test_peak_uses_history_state_and_now(tmp_path):
    hist = tmp_path / "state.json"
    hist.write_text(json.dumps({"initial": 200.0, "history": [["2026-09-01", 210.0], ["2026-09-02", 205.0]]}))
    assert peak_equity(198.0, str(hist), str(tmp_path)) == 210.0
    hist.write_text(json.dumps({"initial": 220.0, "history": [["2026-09-01", 210.0]]}))
    assert peak_equity(198.0, str(hist), str(tmp_path)) == 220.0
    # el maximo persiste aunque el historial desaparezca
    hist.unlink()
    assert peak_equity(199.0, str(hist), str(tmp_path)) == 220.0
    assert peak_equity(250.0, str(hist), str(tmp_path)) == 250.0


def test_drawdown_freezes_and_manual_modes(tmp_path):
    hist = tmp_path / "state.json"
    hist.write_text(json.dumps({"history": [["2026-09-01", 200.0]]}))
    g = evaluate(175.0, "off", 0.10, str(hist), str(tmp_path))
    assert g.mode == "freeze" and g.blocks_entries and not g.closes_positions and "drawdown" in g.reason
    g = evaluate(195.0, "off", 0.10, str(hist), str(tmp_path))
    assert g.mode == "off" and not g.blocks_entries and abs(g.drawdown + 0.025) < 1e-9
    g = evaluate(195.0, "close", 0.10, str(hist), str(tmp_path))
    assert g.closes_positions and g.blocks_entries
    g = evaluate(195.0, "off", 0.0, str(hist), str(tmp_path))  # sin freno automatico
    assert g.mode == "off"


def test_peak_resets_after_withdrawal(tmp_path):
    import json
    from autotrader.guard import evaluate, withdrawal_status
    hist = tmp_path / "aggr_state.json"
    # historial: subio a 1000, retiro (base 700) y despues 690: sin la marca de retiro seria una caida del 31 %
    hist.write_text(json.dumps({"initial": 500, "history": [["2026-01-01T00:00Z", 1000.0], ["2026-02-01T00:00Z", 690.0]],
                                "withdrawal": {"base": 700.0, "last_at": "2026-01-15T00:00Z"}}), encoding="utf-8")
    g = evaluate(690.0, "off", 0.30, str(hist), str(tmp_path))
    assert g.mode == "off" and g.peak == 700.0
    # un maximo persistido de antes del retiro se descarta
    (tmp_path / "peak_equity.json").write_text(json.dumps({"peak": 1000.0, "reset_at": ""}), encoding="utf-8")
    assert evaluate(690.0, "off", 0.30, str(hist), str(tmp_path)).peak == 700.0
    # sin retiro, el maximo historico manda
    hist.write_text(json.dumps({"initial": 500, "history": [["2026-01-01T00:00Z", 1000.0], ["2026-02-01T00:00Z", 690.0]]}), encoding="utf-8")
    assert evaluate(690.0, "off", 0.30, str(hist), str(tmp_path)).mode == "freeze"
    w = withdrawal_status(720.0, 400.0, 0.80, 0.30)
    assert w["alert"] and w["target"] == 720.0 and w["suggested_amount"] == 216.0 and w["equity_after"] == 504.0
    w2 = withdrawal_status(600.0, 500.0, 0.80, 0.30, "2026-01-15T00:00Z")
    assert not w2["alert"] and w2["target"] == 900.0 and abs(w2["progress"] - 0.25) < 1e-9 and w2["last_at"] == "2026-01-15T00:00Z"


def test_detect_cash_flow_from_balance_reconciliation():
    from autotrader.guard import detect_cash_flow
    assert detect_cash_flow(None, 500.0, 0.0) == 0.0
    assert detect_cash_flow(500.0, 500.4, 0.0) == 0.0  # redondeos y comisiones sueltas no cuentan
    assert detect_cash_flow(900.0, 630.0, 0.0) == -270.0  # retiro
    assert detect_cash_flow(900.0, 660.0, 30.0) == -270.0  # retiro con una operacion ganadora entre medias
    assert detect_cash_flow(500.0, 700.0, 0.0) == 200.0  # deposito
