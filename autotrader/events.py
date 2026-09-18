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
- modo "freeze" (propio de los vencimientos de opciones): sin entradas nuevas y sin tocar los stops.

Vencimientos de opciones: el tercer viernes de cada mes (trimestral en marzo, junio, septiembre y
diciembre, cuando coincide con futuros de indices y el rebalanceo del S&P) se genera solo, con ventana
de 2 h antes a 30 min despues del cierre de Nueva York y modo "freeze". Se puede desactivar con
"opex": false en el JSON.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
OPEX_HOURS_BEFORE = 2.0
OPEX_HOURS_AFTER = 0.5

DEFAULT_EVENTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "events.json")


@dataclass(frozen=True)
class Event:
    name: str
    at: datetime
    impact: str = "high"
    tags: tuple[str, ...] = ()
    mode: str | None = None  # None = modo global (EVENT_MODE); "freeze" = sin entradas y sin tocar stops
    hours_before: float | None = None  # ventana propia; None = la global
    hours_after: float | None = None

    def window(self, hours_before: float, hours_after: float) -> tuple[datetime, datetime]:
        hb = self.hours_before if self.hours_before is not None else hours_before
        ha = self.hours_after if self.hours_after is not None else hours_after
        return self.at - timedelta(hours=hb), self.at + timedelta(hours=ha)

    @property
    def is_opex(self) -> bool:
        return "opex" in self.tags


def third_friday(year: int, month: int) -> date:
    d = date(year, month, 15)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def opex_events(start: date, months: int = 12) -> list[Event]:
    """Vencimientos de opciones de EE. UU.: tercer viernes de cada mes, al cierre de Nueva York (16:00 ET)."""
    out = []
    y, m = start.year, start.month
    for _ in range(months):
        d = third_friday(y, m)
        if d >= start - timedelta(days=1):
            at = datetime(d.year, d.month, d.day, 16, 0, tzinfo=NY).astimezone(timezone.utc)
            quarterly = m in (3, 6, 9, 12)
            name = "Vencimiento trimestral de opciones y futuros (triple witching) + rebalanceo S&P" if quarterly else "Vencimiento mensual de opciones"
            out.append(Event(name, at, "high", ("opex", "quarterly") if quarterly else ("opex",), "freeze", OPEX_HOURS_BEFORE, OPEX_HOURS_AFTER))
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def load_events(path: str | None = None, now: datetime | None = None) -> list[Event]:
    path = path or DEFAULT_EVENTS_PATH
    raw = {"events": []}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    out = []
    for e in raw.get("events", []):
        at = datetime.fromisoformat(e["at"].replace("Z", "+00:00"))
        if at.tzinfo is None:
            at = at.replace(tzinfo=timezone.utc)
        out.append(Event(e["name"], at.astimezone(timezone.utc), e.get("impact", "high"), tuple(e.get("tags", [])), e.get("mode"),
                         e.get("hours_before"), e.get("hours_after")))
    if raw.get("opex", True):
        start = (now or datetime.now(timezone.utc)).date() - timedelta(days=45)
        out += opex_events(start, months=14)
    return sorted(out, key=lambda x: x.at)


def effective_mode(events: list[Event], global_mode: str) -> str:
    """Modo que aplica cuando coinciden varios eventos: la proteccion global manda sobre el congelado."""
    modes = {e.mode or global_mode for e in events}
    for m in ("trail", "close"):
        if m in modes:
            return m
    return "freeze" if "freeze" in modes else global_mode


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
