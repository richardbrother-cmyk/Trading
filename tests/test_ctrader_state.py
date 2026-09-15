import importlib.util, os, sys

spec = importlib.util.spec_from_file_location("ctrader_state", os.path.join(os.path.dirname(__file__), "..", "scripts", "ctrader_state.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


def test_merge_history_accumulates_and_caps():
    snap = {"at": "2026-09-15T16:05Z", "equity": 10010.0, "positions": []}
    m = mod.merge_history(None, snap, {"at": "2026-09-15 16:05", "equity": 10010.0})
    assert m["history"] == [["2026-09-15T16:05Z", 10010.0]] and len(m["cycles"]) == 1
    prev = {"history": [[f"t{i}", float(i)] for i in range(mod.MAX_HISTORY)], "cycles": [{"at": str(i)} for i in range(mod.MAX_CYCLES)]}
    m2 = mod.merge_history(prev, snap, {"at": "x"})
    assert len(m2["history"]) == mod.MAX_HISTORY and m2["history"][-1][1] == 10010.0
    assert len(m2["cycles"]) == mod.MAX_CYCLES and m2["cycles"][-1]["at"] == "x"
