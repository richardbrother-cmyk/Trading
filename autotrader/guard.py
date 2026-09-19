"""Freno global del bot: interruptor manual y freno por drawdown acumulado.

- BOT_HALT=off     operativa normal.
- BOT_HALT=freeze  no se abren posiciones nuevas; las abiertas siguen con su stop.
- BOT_HALT=close   ademas se cierran a mercado las posiciones del bot (nunca las manuales).
- MAX_DRAWDOWN_PCT freno automatico: si el equity cae ese porcentaje desde su maximo historico, el bot
  pasa a `freeze` hasta que alguien lo revise. El maximo se toma del historial publicado del panel
  (docs/*_state.json), del fichero state/peak_equity.json y del equity actual, lo que sea mayor.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

HALT_MODES = ("off", "freeze", "close")


@dataclass(frozen=True)
class Guard:
    mode: str  # off | freeze | close
    reason: str
    peak: float
    drawdown: float  # fraccion negativa o cero

    @property
    def blocks_entries(self) -> bool:
        return self.mode != "off"

    @property
    def closes_positions(self) -> bool:
        return self.mode == "close"

    def as_dict(self) -> dict:
        return {"mode": self.mode, "reason": self.reason, "peak": round(self.peak, 2), "drawdown": round(self.drawdown, 4)}


def _load(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _history_peak(path: str) -> tuple[float, str]:
    """(maximo del historial publicado, marca del ultimo retiro). Tras un retiro solo cuenta el historial posterior y la
    base (saldo tras el retiro) sustituye al capital inicial: sacar dinero no es una caida."""
    raw = _load(path)
    if not raw:
        return 0.0, ""
    wd = raw.get("withdrawal") or {}
    since = str(wd.get("last_at") or "")
    values = [float(v) for t, v in raw.get("history", []) if v is not None and (not since or str(t) >= since)]
    if since and wd.get("base"):
        values.append(float(wd["base"]))
    elif raw.get("initial"):  # capital inicial de la cuenta, por si el historial empieza tras las primeras perdidas
        values.append(float(raw["initial"]))
    return (max(values) if values else 0.0), since


def peak_equity(equity_now: float, history_path: str, state_dir: str) -> float:
    """Maximo historico del equity y lo persiste en state/peak_equity.json (se descarta si es anterior a un retiro)."""
    peak_file = os.path.join(state_dir, "peak_equity.json")
    hist_peak, since = _history_peak(history_path)
    stored = 0.0
    if os.path.exists(peak_file):
        try:
            with open(peak_file, encoding="utf-8") as fh:
                raw = json.load(fh)
            if str(raw.get("reset_at") or "") == since:
                stored = float(raw.get("peak", 0.0))
        except (OSError, ValueError):
            stored = 0.0
    peak = max(stored, hist_peak, equity_now)
    try:
        os.makedirs(state_dir, exist_ok=True)
        with open(peak_file, "w", encoding="utf-8") as fh:
            json.dump({"peak": round(peak, 2), "reset_at": since}, fh)
    except OSError:
        pass
    return peak


def detect_cash_flow(prev_balance: float | None, balance: float, deals_net_since: float, tolerance: float = 1.0) -> float:
    """Movimiento de caja entre dos lecturas del saldo: lo que no explican las operaciones cerradas entre medias.
    Negativo = retiro, positivo = deposito; 0 si la diferencia cabe en la tolerancia (comisiones sueltas, redondeos)."""
    if prev_balance is None:
        return 0.0
    flow = balance - prev_balance - deals_net_since
    return round(flow, 2) if abs(flow) > tolerance else 0.0


def withdrawal_status(equity: float, base: float, trigger_pct: float, withdraw_pct: float, last_at: str = "") -> dict:
    """Regla de retiros: al ganar `trigger_pct` sobre la base (saldo tras el ultimo retiro, o el inicial) toca retirar
    `withdraw_pct` del saldo. Devuelve base, objetivo, progreso y, si toca, el importe sugerido."""
    target = base * (1 + trigger_pct) if base > 0 else 0.0
    progress = (equity / base - 1) / trigger_pct if base > 0 and trigger_pct > 0 else 0.0
    alert = bool(target > 0 and equity >= target)
    return {"base": round(base, 2), "last_at": last_at or None, "trigger_pct": trigger_pct, "withdraw_pct": withdraw_pct,
            "target": round(target, 2), "progress": round(progress, 3), "alert": alert,
            "suggested_amount": round(equity * withdraw_pct, 2) if alert else 0.0,
            "equity_after": round(equity * (1 - withdraw_pct), 2) if alert else None}


def evaluate(equity_now: float, halt_mode: str, max_drawdown_pct: float, history_path: str, state_dir: str) -> Guard:
    mode = (halt_mode or "off").lower()
    if mode not in HALT_MODES:
        raise ValueError(f"BOT_HALT debe ser uno de {HALT_MODES}, no {halt_mode!r}")
    peak = peak_equity(equity_now, history_path, state_dir)
    dd = equity_now / peak - 1 if peak > 0 else 0.0
    if mode != "off":
        return Guard(mode, f"interruptor manual BOT_HALT={mode}", peak, dd)
    if max_drawdown_pct > 0 and dd <= -max_drawdown_pct:
        return Guard("freeze", f"drawdown {dd * 100:.1f} % desde el maximo de {peak:,.2f} (limite {max_drawdown_pct * 100:.0f} %)", peak, dd)
    return Guard("off", "", peak, dd)
