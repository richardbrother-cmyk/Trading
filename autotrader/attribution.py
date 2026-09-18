"""Atribucion de operaciones: del bot, manuales o excluidas (entradas de un fallo conocido).

Las metricas "ajustadas" del panel miden solo lo que hizo el bot con la logica vigente:
- manual: la posicion no lleva la etiqueta del bot (cTrader) o no la abrio el bot.
- excluida: la abrio el bot antes de la fecha de corte de `data/exclusions.json` (fallo del filtro de huecos).
- bot: el resto.
"""

from __future__ import annotations

import json
import os
from collections import deque
from datetime import datetime, timezone

EXCLUSIONS_PATH = "data/exclusions.json"
ORIGINS = ("bot", "manual", "excluded")


def parse_ts(value) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    s = str(value).replace(" ", "T").replace("Z", "")
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[:19], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def load_exclusions(path: str = EXCLUSIONS_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def cutoff_for(broker: str, exclusions: dict | None) -> datetime | None:
    rule = (exclusions or {}).get(broker) or {}
    return parse_ts(rule.get("entries_before"))


def classify(is_bot: bool, opened_at, cutoff: datetime | None) -> str:
    if not is_bot:
        return "manual"
    opened = parse_ts(opened_at)
    if cutoff and opened and opened < cutoff:
        return "excluded"
    return "bot"


def fifo_trades(fills: list[dict]) -> tuple[list[dict], list[dict]]:
    """Reconstruye operaciones cerradas (FIFO) y lotes abiertos a partir de ejecuciones de compra/venta.

    fills: [{"symbol", "side": "buy"|"sell", "qty", "price", "at"}], en cualquier orden.
    """
    lots: dict[str, deque] = {}
    trades = []
    for f in sorted((f for f in fills if f.get("price") and f.get("qty")), key=lambda f: (parse_ts(f["at"]) or datetime.min.replace(tzinfo=timezone.utc))):
        q = lots.setdefault(f["symbol"], deque())
        qty = float(f["qty"])
        if f["side"] == "buy":
            q.append([qty, float(f["price"]), f["at"]])
            continue
        while qty > 1e-9 and q:
            lot = q[0]
            take = min(qty, lot[0])
            trades.append({"symbol": f["symbol"], "opened_at": lot[2], "at": f["at"], "units": take, "entry": lot[1],
                           "exit": float(f["price"]), "gross": round((float(f["price"]) - lot[1]) * take, 2), "swap": 0.0,
                           "commission": 0.0, "net": round((float(f["price"]) - lot[1]) * take, 2)})
            lot[0] -= take
            qty -= take
            if lot[0] <= 1e-9:
                q.popleft()
    open_lots = [{"symbol": s, "qty": lot[0], "entry": lot[1], "opened_at": lot[2]} for s, q in lots.items() for lot in q]
    return trades, open_lots


def bot_metrics(trades: list[dict], open_items: list[dict], initial: float) -> dict:
    """Suma realizado y abierto por origen y calcula el capital ajustado (solo operaciones del bot vigente)."""
    realized = {k: 0.0 for k in ORIGINS}
    open_pnl = {k: 0.0 for k in ORIGINS}
    counts = {k: 0 for k in ORIGINS}
    for t in trades:
        o = t.get("origin", "bot")
        realized[o] = realized.get(o, 0.0) + float(t.get("net") or 0.0)
        counts[o] = counts.get(o, 0) + 1
    for p in open_items:
        o = p.get("origin", "bot")
        open_pnl[o] = open_pnl.get(o, 0.0) + float(p.get("pnl") or 0.0)
    adjusted = initial + realized["bot"] + open_pnl["bot"]
    return {"initial": initial, "realized": {k: round(v, 2) for k, v in realized.items()}, "open": {k: round(v, 2) for k, v in open_pnl.items()},
            "counts": counts, "adjusted_equity": round(adjusted, 2), "adjusted_return": round(adjusted / initial - 1, 4) if initial else 0.0}
