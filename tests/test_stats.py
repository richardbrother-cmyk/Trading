import numpy as np

from autotrader.stats import benjamini_hochberg, bootstrap_ci, p_value_mean_positive, summarize_variant


def test_bootstrap_ci_contains_mean_and_narrows_with_n():
    rng = np.random.default_rng(1)
    small, big = rng.normal(0.2, 1.0, 30), rng.normal(0.2, 1.0, 3000)
    lo_s, hi_s = bootstrap_ci(small); lo_b, hi_b = bootstrap_ci(big)
    assert lo_s <= small.mean() <= hi_s and lo_b <= big.mean() <= hi_b
    assert (hi_b - lo_b) < (hi_s - lo_s)


def test_p_value_detects_real_edge_and_not_noise():
    rng = np.random.default_rng(2)
    edge = rng.normal(0.5, 1.0, 200)   # media claramente positiva
    noise = rng.normal(0.0, 1.0, 200)  # sin borde
    assert p_value_mean_positive(edge) < 0.01
    assert p_value_mean_positive(noise) > 0.05
    assert p_value_mean_positive([0.1]) == 1.0


def test_benjamini_hochberg_orders_and_clips():
    q = benjamini_hochberg([0.01, 0.04, 0.03, 0.20])
    assert [round(v, 3) for v in q] == [0.04, 0.053, 0.053, 0.2]
    assert benjamini_hochberg([]) == []
    assert all(0 <= v <= 1 for v in benjamini_hochberg([0.9, 0.95, 1.0]))


def test_summarize_variant_fields():
    s = summarize_variant([1.0, -1.0, 2.0, -1.0, 3.0])
    assert s["n"] == 5 and s["sum_r"] == 4.0 and s["profit_factor"] == 3.0 and s["win_rate"] == 0.6
    assert s["ci95"][0] <= s["mean_r"] <= s["ci95"][1]
    assert summarize_variant([]) == {"n": 0}
