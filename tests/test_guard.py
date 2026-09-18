import json

from autotrader.guard import evaluate, peak_equity


def test_peak_uses_history_state_and_now(tmp_path):
    hist = tmp_path / "state.json"
    hist.write_text(json.dumps({"history": [["2026-09-01", 210.0], ["2026-09-02", 205.0]]}))
    assert peak_equity(198.0, str(hist), str(tmp_path)) == 210.0
    # el maximo persiste aunque el historial desaparezca
    hist.unlink()
    assert peak_equity(199.0, str(hist), str(tmp_path)) == 210.0
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
