"""Estadistica para comparar variantes de estrategia sin engañarse: intervalo de confianza por bootstrap, p-valor de
"la ganancia media por operacion es mayor que cero" y correccion de Benjamini-Hochberg cuando se prueban muchas
variantes a la vez (idea tomada de research_common.py de HKUDS/AI-Trader, reimplementada con numpy).

Lectura: con veinte variantes probadas, alguna saldra "buena" por azar; la correccion ajusta los p-valores para que la
tasa de falsos descubrimientos entre las que se declaran buenas quede por debajo del nivel elegido (q).
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci(values, stat=np.mean, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Intervalo de confianza (percentil) de un estadistico por remuestreo con reemplazo."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = stat(x[idx], axis=1) if stat is np.mean else np.array([stat(x[i]) for i in idx])
    return (float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2))))


def p_value_mean_positive(values, n_boot: int = 20_000, seed: int = 0) -> float:
    """p-valor unilateral de H0: media <= 0, por bootstrap centrado (fraccion de medias remuestreadas bajo H0 que igualan
    o superan la media observada). Con muestras pequenas se acota por abajo a 1/n_boot."""
    x = np.asarray(values, dtype=float)
    if x.size < 2:
        return 1.0
    rng = np.random.default_rng(seed)
    centered = x - x.mean()  # distribucion bajo H0 (media cero)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = centered[idx].mean(axis=1)
    p = float((boots >= x.mean()).mean())
    return max(p, 1.0 / n_boot)


def benjamini_hochberg(pvalues) -> list[float]:
    """q-valores (p ajustados) de Benjamini-Hochberg: controlan la tasa de falsos descubrimientos entre las hipotesis
    declaradas significativas. Devuelve la lista en el mismo orden que la entrada."""
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # monotonia: cada q es el minimo de los q de rango superior
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return [float(v) for v in out]


def summarize_variant(r_values, alpha: float = 0.05, seed: int = 0) -> dict:
    """Resumen de una variante a partir de su lista de resultados en R por operacion."""
    x = np.asarray(r_values, dtype=float)
    if x.size == 0:
        return {"n": 0}
    lo, hi = bootstrap_ci(x, alpha=alpha, seed=seed)
    wins, losses = x[x > 0].sum(), -x[x <= 0].sum()
    return {"n": int(x.size), "mean_r": round(float(x.mean()), 3), "ci95": [round(lo, 3), round(hi, 3)],
            "p_value": round(p_value_mean_positive(x, seed=seed), 4), "win_rate": round(float((x > 0).mean()), 3),
            "profit_factor": round(float(wins / losses), 2) if losses > 0 else None, "sum_r": round(float(x.sum()), 1)}
