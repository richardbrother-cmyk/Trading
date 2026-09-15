"""Calendario de eventos de alto impacto y reglas de proteccion alrededor de ellos.

El calendario vive en data/events.json (editable). Cada evento: {"name", "at" (ISO UTC), "impact", "tags"}.
Ventana de proteccion: desde `hours_before` horas antes hasta `hours_after` horas despues del evento.

Dentro de la ventana:
- no se abren posiciones nuevas;
- modo "trail": a las posiciones con ganancia >= min_gain se les sube el stop a max(stop actual,
  precio * (1 - trail_pct)), es decir, se asegura parte del beneficio; las posiciones en perdida
  conservan su stop original;
- modo "close": las posiciones con ganancia >= min_gain se cierran antes del evento y se reevaluan
  en el primer ciclo posterior a la ventana.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

DEFAULT_EVENTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "events.json")


@dataclass(frozen=True)
class Event:
    name: str
    at: datetime
    impact: str = "high"
    tags: tuple[str, ...] = ()

    def window(self, hours_before: float, hours_after: float) -> tuple[datetime, datetime]:
        return self.at - timedelta(hours=hours_before), self.at + timedelta(hours=hours_after)


def load_events(path: str | None = None) -> list[Event]:
    path = path or DEFAULT_EVENTS_PATH
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    out = []
    for e in raw.get("events", []):
        at = datetime.fromisoformat(e["at"].replace("Z", "+00:00"))
        if at.tzinfo is None:
            at = at.replace(tzinfo=timezone.utc)
        out.append(Event(e["name"], at.astimezone(timezone.utc), e.get("impact", "high"), tuple(e.get("tags", []))))
    return sorted(out, key=lambda x: x.at)


def active_events(events: list[Event], now: datetime, hours_before: float, hours_after: float, impact: str = "high") -> list[Event]:
    """Eventos cuya ventana de proteccion incluye `now`."""
    out = []
    for e in events:
        if impact == "high" and e.impact != "high":
            continue
        start, end = e.window(hours_before, hours_after)
        if start <= now <= end:
            out.append(e)
    return out


def upcoming_events(events: list[Event], now: datetime, days: int = 7) -> list[Event]:
    return [e for e in events if now - timedelta(hours=6) <= e.at <= now + timedelta(days=days)]


def trailed_stop(current_stop: float, price: float, trail_pct: float) -> float:
    """Nuevo stop: nunca baja, sube hasta price * (1 - trail_pct)."""
    return max(current_stop, round(price * (1 - trail_pct), 2))
