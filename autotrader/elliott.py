"""Estrategia mecanica basada en ondas de Elliott sobre barras de 4 h o diarias (CFDs).

La lectura de Elliott es subjetiva; aqui se convierte en reglas comprobables:

1. Pivotes por ZigZag causal: un maximo (minimo) solo se confirma cuando el precio se aleja de el al menos
   `zz_atr` ATR (o `zz_pct` del precio). El pivote queda fechado en su barra, pero solo se conoce en la barra
   de confirmacion, y la estrategia solo usa pivotes ya confirmados (sin mirar al futuro).
2. Etiquetado de un impulso alcista sobre los ultimos pivotes L0 < H1 > L2 (< H3 > L4) con las tres reglas duras:
   la onda 2 no retrocede mas del 100 % de la 1 (L2 > L0), la onda 3 no es la mas corta (H3 - L2 >= w3_min_ratio * onda 1)
   y la onda 4 no solapa con la 1 (L4 > H1). Ademas, bandas de Fibonacci con tolerancia: retroceso de la onda 2
   entre `w2_min` y `w2_max` de la onda 1, y de la onda 4 entre `w4_min` y `w4_max` de la onda 3. El impulso
   bajista es el espejo.
3. Operativa: al final de la onda 2 (entrando en la 3) o al final de la onda 4 (entrando en la 5).
   - entrada "pivote": en la apertura siguiente a la barra que confirma el minimo de la onda 2 (o 4);
   - entrada "ruptura": cuando un cierre supera el extremo de la onda 1 (o 3), siempre que el minimo de la
     correccion no se haya perdido y sin esperar mas de `max_wait_bars`.
   Stop bajo el minimo de la correccion ("onda") o en el nivel que invalida el recuento ("inicio": L0 para la
   onda 2, H1 para la onda 4), con un colchon de `stop_buffer_atr` ATR. Objetivo: `target_ext` veces la onda 1
   medido desde el final de la correccion. Cierre forzoso a los `max_hold_days`.

Costes y tamano como en el swing: spread a medias en cada lado, comision proporcional, swap diario, riesgo fijo
por operacion con lote minimo del simbolo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .intraday import SPECS, ITrade
from .swing import SwingParams, _size, resample as resample_h

BARS_PER_DAY = {"H4": 6, "D1": 1}


@dataclass(frozen=True)
class ElliottParams:
    timeframe: str = "H4"  # "H4" o "D1"
    zz_atr: float = 3.0  # umbral de giro del ZigZag en ATR; 0 para usar zz_pct
    zz_pct: float = 0.0
    atr_period: int = 14
    waves: tuple[int, ...] = (2,)  # (2,), (4,) o (2, 4): fin de que onda se opera
    entry: str = "ruptura"  # "ruptura" o "pivote"
    max_wait_bars: int = 20  # ruptura: barras maximas entre la confirmacion del pivote y la ruptura
    w2_min: float = 0.382
    w2_max: float = 0.786
    w3_min_ratio: float = 1.0
    w4_min: float = 0.236
    w4_max: float = 0.5
    target_ext: float = 1.618  # objetivo = fin de la correccion + target_ext * onda 1
    stop: str = "onda"  # "onda" o "inicio"
    stop_buffer_atr: float = 0.25
    max_hold_days: float = 10.0
    allow_short: bool = True
    risk_pct: float = 0.01
    max_risk_pct: float = 0.03
    commission_side: float = 0.000025
    swap_daily: float = 0.0001
    max_notional_mult: float = 20.0

    def max_hold_bars(self) -> int:
        return max(1, int(self.max_hold_days * BARS_PER_DAY[self.timeframe]))

    def swing_like(self) -> SwingParams:
        """Parametros de tamano compatibles con `swing._size`."""
        return SwingParams(risk_pct=self.risk_pct, max_risk_pct=self.max_risk_pct, max_notional_mult=self.max_notional_mult)


@dataclass(frozen=True)
class Pivot:
    idx: int  # barra del extremo
    price: float
    kind: int  # +1 maximo, -1 minimo
    confirmed: int  # barra en la que se supo


def resample(df15: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe == "H4":
        return resample_h(df15, "H4")
    if timeframe == "D1":
        # dia de negociacion de 22:00 a 22:00 UTC (cierre de Nueva York en verano); se descartan dias con pocas barras
        agg = df15.resample("24h", label="left", closed="left", offset="22h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        n = df15["close"].resample("24h", label="left", closed="left", offset="22h").count()
        return agg[n >= 20].dropna(subset=["open", "close"])
    raise ValueError(timeframe)


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    c = df["close"]
    tr = pd.concat([df["high"] - df["low"], (df["high"] - c.shift()).abs(), (df["low"] - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def zigzag(high: np.ndarray, low: np.ndarray, thr: np.ndarray) -> list[Pivot]:
    """ZigZag causal. `thr[i]` es la distancia minima de giro vigente en la barra i (en precio).
    Devuelve los pivotes en orden, cada uno con la barra en la que quedo confirmado."""
    n = len(high)
    pivots: list[Pivot] = []
    if n == 0:
        return pivots
    hi_i, lo_i = 0, 0
    direction = 0  # 0 sin definir, +1 buscando maximo, -1 buscando minimo
    for i in range(1, n):
        if np.isnan(thr[i]) or thr[i] <= 0:
            if high[i] > high[hi_i]:
                hi_i = i
            if low[i] < low[lo_i]:
                lo_i = i
            continue
        if direction >= 0 and high[i] > high[hi_i]:
            hi_i = i
        if direction <= 0 and low[i] < low[lo_i]:
            lo_i = i
        if direction == 0:
            if high[hi_i] - low[i] >= thr[i] and hi_i <= i:
                pivots.append(Pivot(hi_i, float(high[hi_i]), 1, i)); direction = -1; lo_i = i
            elif high[i] - low[lo_i] >= thr[i]:
                pivots.append(Pivot(lo_i, float(low[lo_i]), -1, i)); direction = 1; hi_i = i
        elif direction == 1:
            if high[hi_i] - low[i] >= thr[i]:
                pivots.append(Pivot(hi_i, float(high[hi_i]), 1, i)); direction = -1; lo_i = i
        else:
            if high[i] - low[lo_i] >= thr[i]:
                pivots.append(Pivot(lo_i, float(low[lo_i]), -1, i)); direction = 1; hi_i = i
    return pivots


def _ratio(a: float, b: float) -> float:
    return a / b if b > 0 else float("nan")


def impulse_setup(piv: list[Pivot], p: ElliottParams) -> dict | None:
    """Comprueba si los ultimos pivotes forman el final de una onda 2 o 4 (alcista o bajista). Devuelve el
    plan de la operacion (lado, onda, nivel de ruptura, stop de invalidacion, longitud de la onda 1) o None."""
    if not piv:
        return None
    last = piv[-1]
    side = -last.kind  # el ultimo pivote es un minimo (-1) en un impulso alcista (side +1) y viceversa
    if side == -1 and not p.allow_short:
        return None
    s = side
    if 2 in p.waves and len(piv) >= 3:
        l0, h1, l2 = piv[-3], piv[-2], piv[-1]
        if l0.kind == -s and h1.kind == s and l2.kind == -s:
            w1 = s * (h1.price - l0.price)
            r2 = _ratio(s * (h1.price - l2.price), w1)
            if w1 > 0 and s * (l2.price - l0.price) > 0 and p.w2_min <= r2 <= p.w2_max:
                return {"side": s, "wave": 2, "pivot": l2, "trigger": h1.price, "wave1": w1, "correction_low": l2.price,
                        "invalidation": l0.price, "retrace": round(r2, 3)}
    if 4 in p.waves and len(piv) >= 5:
        l0, h1, l2, h3, l4 = piv[-5:]
        if l0.kind == -s and h1.kind == s and l2.kind == -s and h3.kind == s and l4.kind == -s:
            w1 = s * (h1.price - l0.price)
            w3 = s * (h3.price - l2.price)
            r2 = _ratio(s * (h1.price - l2.price), w1)
            r4 = _ratio(s * (h3.price - l4.price), w3)
            if (w1 > 0 and w3 >= p.w3_min_ratio * w1 and s * (h3.price - h1.price) > 0 and s * (l2.price - l0.price) > 0
                    and 0 < r2 < 1 and s * (l4.price - h1.price) > 0 and p.w4_min <= r4 <= p.w4_max):
                return {"side": s, "wave": 4, "pivot": l4, "trigger": h3.price, "wave1": w1, "correction_low": l4.price,
                        "invalidation": h1.price, "retrace": round(r4, 3)}
    return None


def entry_signals(high: np.ndarray, low: np.ndarray, close: np.ndarray, atr_values: np.ndarray, p: ElliottParams) -> list[dict]:
    """Senales de entrada evaluadas al cierre de cada barra, sin mirar al futuro y sin depender de si hay una posicion
    abierta (eso lo decide quien las consume). Cada senal es el plan de `impulse_setup` mas `bar` (indice de la barra
    cuyo cierre dispara la entrada; se ejecuta en la apertura de la siguiente) y `stop`/`target` en precio."""
    n = len(close)
    thr = atr_values * p.zz_atr if p.zz_atr > 0 else close * p.zz_pct
    pivots = zigzag(high, low, thr)
    confirmed_at = np.array([pv.confirmed for pv in pivots])
    out: list[dict] = []
    used: set[int] = set()  # pivotes (barra) ya disparados o caducados
    pending: dict | None = None  # ruptura: plan a la espera del cierre por encima del nivel
    for i in range(p.atr_period + 1, n):
        k = int(np.searchsorted(confirmed_at, i, side="right"))  # pivotes conocidos al cierre de la barra i
        plan = None
        setup = impulse_setup(pivots[:k], p) if k else None
        if setup and setup["pivot"].idx not in used and setup["pivot"].confirmed <= i:
            if p.entry == "pivote":
                if setup["pivot"].confirmed == i:
                    plan = setup
                else:
                    used.add(setup["pivot"].idx)
            else:
                pending = pending if (pending and pending["pivot"].idx == setup["pivot"].idx) else setup
        if pending is not None and plan is None:
            s = pending["side"]
            if i - pending["pivot"].confirmed > p.max_wait_bars or s * (low[i] if s == 1 else high[i]) < s * pending["correction_low"]:
                used.add(pending["pivot"].idx); pending = None
            elif s * (close[i] - pending["trigger"]) > 0:
                plan = pending; pending = None
        if plan is None:
            continue
        used.add(plan["pivot"].idx)
        s = plan["side"]
        stop_level = plan["correction_low"] if p.stop == "onda" else plan["invalidation"]
        sig = dict(plan)
        sig.update({"bar": i, "stop": stop_level - s * p.stop_buffer_atr * atr_values[i],
                    "target": plan["correction_low"] + s * p.target_ext * plan["wave1"], "atr": float(atr_values[i]),
                    "pivots": pivots[:k]})
        out.append(sig)
    return out


def pending_setup(high: np.ndarray, low: np.ndarray, close: np.ndarray, atr_values: np.ndarray, p: ElliottParams) -> dict | None:
    """Estado del recuento al cierre de la ultima barra, para el panel: el plan vigente (a la espera de la ruptura o
    disparado en la ultima barra) con su fase, o None si no hay patron valido."""
    n = len(close)
    if n <= p.atr_period + 1:
        return None
    thr = atr_values * p.zz_atr if p.zz_atr > 0 else close * p.zz_pct
    pivots = zigzag(high, low, thr)
    setup = impulse_setup(pivots, p)
    if setup is None:
        return None
    s, i = setup["side"], n - 1
    conf = setup["pivot"].confirmed
    if p.entry == "pivote":
        phase = "ready" if conf == i else "expired"
    else:
        phase = "waiting_break"
        for j in range(conf, n):
            if j - conf > p.max_wait_bars or s * (low[j] if s == 1 else high[j]) < s * setup["correction_low"]:
                phase = "expired"; break
            if s * (close[j] - setup["trigger"]) > 0:
                phase = "ready" if j == i else "broke_earlier"; break
    out = {k: v for k, v in setup.items() if k != "pivot"}
    out.update({"phase": phase, "pivot_bar": setup["pivot"].idx, "pivot_confirmed": conf, "pivots": pivots[-5:]})
    return out


def backtest_symbol(df15: pd.DataFrame, sym: str, p: ElliottParams, initial: float = 10_000.0, bars: pd.DataFrame | None = None) -> list[ITrade]:
    """`bars`: barras ya en el marco temporal de `p` (p.ej. diarias de Yahoo); si no se dan, se agregan desde `df15`."""
    spec = SPECS[sym]
    d = bars if bars is not None else resample(df15, p.timeframe)
    a = atr(d, p.atr_period).to_numpy()
    o, h, l, c = d["open"].to_numpy(), d["high"].to_numpy(), d["low"].to_numpy(), d["close"].to_numpy()
    idx = d.index
    n = len(d)
    sp = p.swing_like()
    trades: list[ITrade] = []
    equity = initial
    last_exit = -1  # una posicion a la vez: las senales que saltan con una operacion abierta se ignoran
    for plan in entry_signals(h, l, c, a, p):
        i = plan["bar"]
        if i <= last_exit or i >= n - 1:
            continue
        s = plan["side"]
        entry = o[i + 1] + s * spec.spread / 2
        stop, tp = plan["stop"], plan["target"]
        if s * (entry - stop) <= 0 or s * (tp - entry) <= 0:
            continue
        units = _size(equity, entry, stop, spec, sp)
        if units <= 0:
            continue
        t = ITrade(sym, s, idx[i + 1], entry, stop, units)
        t.costs = entry * units * p.commission_side
        j_end = min(n - 1, i + 1 + p.max_hold_bars())
        exit_price, reason, j_exit = None, "", None
        for j in range(i + 1, j_end + 1):
            if (s == 1 and l[j] <= stop) or (s == -1 and h[j] >= stop):
                exit_price, reason, j_exit = stop, "stop", j; break
            if (s == 1 and h[j] >= tp) or (s == -1 and l[j] <= tp):
                exit_price, reason, j_exit = tp, "objetivo", j; break
        if exit_price is None:
            exit_price, reason, j_exit = c[j_end], "tiempo maximo", j_end
        t.exit_time, t.reason = idx[j_exit], reason
        t.exit = exit_price - s * spec.spread / 2
        days_held = max((idx[j_exit] - idx[i + 1]).total_seconds() / 86400, 0)
        t.costs += t.exit * units * p.commission_side + entry * units * p.swap_daily * days_held
        t.wave = plan["wave"]  # type: ignore[attr-defined]
        t.retrace = plan["retrace"]  # type: ignore[attr-defined]
        t.target = tp  # type: ignore[attr-defined]
        trades.append(t)
        equity += t.pnl
        if equity <= 0:
            break
        last_exit = j_exit
    return trades
