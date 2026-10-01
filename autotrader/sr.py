"""Soportes y resistencias sobre barras H4: deteccion de niveles y senales de rebote / ruptura con retest.

Niveles
-------
- Pivote: maximo (o minimo) que supera a `k` barras a cada lado. Se confirma `k` barras despues, asi que en la barra `i` solo
  se usan pivotes con `j + k <= i` (sin mirar al futuro) y como mucho de las ultimas `lookback` barras.
- Los pivotes (maximos y minimos juntos: un soporte roto pasa a ser resistencia y al reves) se agrupan en zonas de ancho
  `cluster_atr` x ATR. Una zona cuenta como nivel cuando reune al menos `min_touches` pivotes.

Senales (se evaluan al cierre de la barra `i`; el bot entra en la apertura siguiente)
------
- "bounce": la barra se mete en la zona de un soporte (minimo <= nivel + zona) y cierra por encima de la zona con vela alcista
  (largo); simetrico en una resistencia (corto).
- "retest": una barra reciente (hasta `retest_bars` atras) cerro con decision por encima de una resistencia, ninguna barra
  posterior cerro de vuelta debajo, y la barra actual vuelve a tocar el nivel (ahora soporte) y cierra por encima (largo);
  simetrico para cortos.
- Stop: al otro lado del nivel (o del extremo de la barra de senal) mas `stop_buf_atr` x ATR; la distancia al stop debe estar
  entre `min_stop_atr` y `max_stop_atr` ATR.
- Objetivo: `target="rr"` => `rr` veces el riesgo; `target="level"` => la siguiente zona contraria (hay que ganar al menos
  `min_rr` veces el riesgo, si no se descarta la operacion).
- `trend=True`: solo largos sobre la EMA200 y solo cortos bajo ella.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def find_pivots(high: np.ndarray, low: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pivotes con `k` barras a cada lado. Devuelve (indice de la barra pivote, precio, tipo +1 maximo / -1 minimo) ordenados por barra."""
    n = len(high)
    idx, price, kind = [], [], []
    for j in range(k, n - k):
        hl, hr = high[j - k:j], high[j + 1:j + k + 1]
        if high[j] > hl.max() and high[j] >= hr.max():
            idx.append(j); price.append(high[j]); kind.append(1)
        ll, lr = low[j - k:j], low[j + 1:j + k + 1]
        if low[j] < ll.min() and low[j] <= lr.min():
            idx.append(j); price.append(low[j]); kind.append(-1)
    return np.array(idx, dtype=int), np.array(price, dtype=float), np.array(kind, dtype=int)


def cluster_levels(prices: np.ndarray, width: float, min_touches: int) -> list[tuple[float, int]]:
    """Agrupa precios ordenados en zonas de ancho maximo `width`; devuelve [(precio medio, numero de pivotes)] con >= min_touches."""
    if len(prices) == 0 or width <= 0:
        return []
    ps = np.sort(prices)
    out: list[tuple[float, int]] = []
    start = 0
    for t in range(1, len(ps) + 1):
        if t == len(ps) or ps[t] - ps[start] > width:
            n = t - start
            if n >= min_touches:
                out.append((float(ps[start:t].mean()), n))
            start = t
    return out


def levels_at(i: int, piv_idx: np.ndarray, piv_price: np.ndarray, k: int, lookback: int, width: float, min_touches: int) -> list[tuple[float, int]]:
    lo = int(np.searchsorted(piv_idx, i - lookback, side="left"))
    hi = int(np.searchsorted(piv_idx, i - k, side="right"))
    return cluster_levels(piv_price[lo:hi], width, min_touches)


def sr_columns(d: pd.DataFrame, p) -> pd.DataFrame:
    """Anade sr_signal (+1/-1/0), sr_stop y sr_target (NaN si el objetivo es por multiplo de R) a un DataFrame con indicadores."""
    n = len(d)
    o, h, l, c = d["open"].to_numpy(), d["high"].to_numpy(), d["low"].to_numpy(), d["close"].to_numpy()
    atr, ema200 = d["atr"].to_numpy(), d["ema200"].to_numpy()
    sig = np.zeros(n, dtype=int)
    stop = np.full(n, np.nan)
    target = np.full(n, np.nan)
    kind_out = np.array([""] * n, dtype=object)
    level_out = np.full(n, np.nan)
    piv_idx, piv_price, _ = find_pivots(h, l, p.sr_pivot_k)
    modes = {"bounce": ("bounce",), "retest": ("retest",), "both": ("bounce", "retest")}[p.sr_mode]
    start = max(210, p.sr_lookback // 4)
    for i in range(start, n):
        a = atr[i]
        if not np.isfinite(a) or a <= 0 or not np.isfinite(ema200[i]):
            continue
        levels = levels_at(i, piv_idx, piv_price, p.sr_pivot_k, p.sr_lookback, p.sr_cluster_atr * a, p.sr_min_touches)
        if not levels:
            continue
        zone = p.sr_zone_atr * a
        close, op, hi_, lo_ = c[i], o[i], h[i], l[i]
        best = None  # (score, side, level, kind)
        for lv, touches in levels:
            for mode in modes:
                # ---- largos
                if (not p.sr_trend) or close > ema200[i]:
                    ok = False
                    if mode == "bounce":
                        ok = lo_ <= lv + zone and lo_ >= lv - p.sr_max_pierce_atr * a and close >= lv + zone and close > op
                    else:
                        j0 = max(1, i - p.sr_retest_bars)
                        broke = [j for j in range(j0, i) if c[j] > lv + zone and c[j - 1] <= lv + zone]
                        if broke:
                            jb = broke[-1]
                            held = all(c[j] > lv for j in range(jb, i))
                            ok = held and lo_ <= lv + zone and close >= lv + zone and close > op
                    if ok:
                        score = touches
                        if best is None or score > best[0]:
                            best = (score, 1, lv, mode)
                # ---- cortos
                if p.allow_short and ((not p.sr_trend) or close < ema200[i]):
                    ok = False
                    if mode == "bounce":
                        ok = hi_ >= lv - zone and hi_ <= lv + p.sr_max_pierce_atr * a and close <= lv - zone and close < op
                    else:
                        j0 = max(1, i - p.sr_retest_bars)
                        broke = [j for j in range(j0, i) if c[j] < lv - zone and c[j - 1] >= lv - zone]
                        if broke:
                            jb = broke[-1]
                            held = all(c[j] < lv for j in range(jb, i))
                            ok = held and hi_ >= lv - zone and close <= lv - zone and close < op
                    if ok:
                        score = touches
                        if best is None or score > best[0]:
                            best = (score, -1, lv, mode)
        if best is None:
            continue
        _, side, lv, mode = best
        if side == 1:
            sl = min(lo_, lv) - p.sr_stop_buf_atr * a
            risk = close - sl
        else:
            sl = max(hi_, lv) + p.sr_stop_buf_atr * a
            risk = sl - close
        if risk < p.sr_min_stop_atr * a or risk > p.sr_max_stop_atr * a:
            continue
        tp = np.nan
        if p.sr_target == "level":
            if side == 1:
                above = [x for x, _t in levels if x > close + 0.25 * a]
                tp = min(above) if above else np.nan
                if not np.isfinite(tp) or tp - close < p.sr_min_rr * risk:
                    continue
            else:
                below = [x for x, _t in levels if x < close - 0.25 * a]
                tp = max(below) if below else np.nan
                if not np.isfinite(tp) or close - tp < p.sr_min_rr * risk:
                    continue
        sig[i], stop[i], target[i], kind_out[i], level_out[i] = side, sl, tp, mode, lv
    out = d.copy()
    out["sr_signal"], out["sr_stop"], out["sr_target"], out["sr_kind"], out["sr_level"] = sig, stop, target, kind_out, level_out
    return out


def last_levels(d: pd.DataFrame, p) -> dict:
    """Soporte y resistencia mas cercanos al ultimo cierre (para el estado publicado). {"support": x|None, "resistance": y|None, "levels": n}."""
    n = len(d)
    if n < 20:
        return {"support": None, "resistance": None, "levels": 0}
    a = float(d["atr"].iloc[-1])
    piv_idx, piv_price, _ = find_pivots(d["high"].to_numpy(), d["low"].to_numpy(), p.sr_pivot_k)
    levels = levels_at(n - 1, piv_idx, piv_price, p.sr_pivot_k, p.sr_lookback, p.sr_cluster_atr * a, p.sr_min_touches)
    close = float(d["close"].iloc[-1])
    below = [x for x, _t in levels if x <= close]
    above = [x for x, _t in levels if x > close]
    return {"support": max(below) if below else None, "resistance": min(above) if above else None, "levels": len(levels)}
