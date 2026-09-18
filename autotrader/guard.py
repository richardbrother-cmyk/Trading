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


def _history_peak(path: str) -> float:
    if not path or not os.path.exists(path):
        return 0.0
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return 0.0
    values = [float(v) for _t, v in raw.get("history", []) if v is not None]
    if raw.get("initial"):  # capital inicial de la cuenta, por si el historial empieza tras las primeras perdidas
        values.append(float(raw["initial"]))
    return max(values) if values else 0.0


def peak_equity(equity_now: float, history_path: str, state_dir: str) -> float:
    """Maximo historico del equity y lo persiste en state/peak_equity.json."""
    peak_file = os.path.join(state_dir, "peak_equity.json")
    stored = 0.0
    if os.path.exists(peak_file):
        try:
            with open(peak_file, encoding="utf-8") as fh:
                stored = float(json.load(fh).get("peak", 0.0))
        except (OSError, ValueError):
            stored = 0.0
    peak = max(stored, _history_peak(history_path), equity_now)
    try:
        os.makedirs(state_dir, exist_ok=True)
        with open(peak_file, "w", encoding="utf-8") as fh:
            json.dump({"peak": round(peak, 2)}, fh)
    except OSError:
        pass
    return peak


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
