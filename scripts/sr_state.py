"""Publica el ultimo ciclo del bot de soportes y resistencias en docs/sr_state.json (para el panel).

Lee state/run_log.jsonl (lo escribe `autotrader swing-run`), toma el ultimo ciclo con la etiqueta del bot y conserva un historial
corto de ciclos con senal u orden. Las posiciones y operaciones del bot salen del estado de la cuenta (docs/aggr_state.json,
etiqueta autotrader-sr), no de aqui.

Uso: python scripts/sr_state.py [--label autotrader-sr] [--log state/run_log.jsonl] [--out docs/sr_state.json]
"""
from __future__ import annotations

import argparse
import json
import os

KEEP_EVENTS = 30


def last_cycle(log_path: str, label: str) -> dict | None:
    if not os.path.exists(log_path):
        return None
    rec = None
    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("kind") == "swing" and row.get("label") == label:
                rec = row
    return rec


def build_state(prev: dict, rec: dict) -> dict:
    events = list(prev.get("events", []))
    acted = [d for d in rec.get("decisions", []) if d.get("action") == "BUY"] or rec.get("orders") or rec.get("closed")
    if acted:
        events.append({"at": rec["timestamp"], "orders": rec.get("orders", []), "closed": rec.get("closed", []),
                       "signals": [{k: d.get(k) for k in ("symbol", "close", "sr_level", "sr_stop", "sr_target", "bar")} for d in rec.get("decisions", [])
                                   if d.get("action") == "BUY"]})
    return {"at": rec["timestamp"][:16] + "Z", "label": rec.get("label"), "strategy": rec.get("strategy"), "equity": rec.get("equity"),
            "guard": rec.get("guard"), "dry_run": rec.get("dry_run", False), "positions": rec.get("positions", {}),
            "decisions": rec.get("decisions", []), "orders": rec.get("orders", []), "closed": rec.get("closed", []), "skipped": rec.get("skipped", []),
            "events": events[-KEEP_EVENTS:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="autotrader-sr")
    ap.add_argument("--log", default="state/run_log.jsonl")
    ap.add_argument("--out", default="docs/sr_state.json")
    args = ap.parse_args()
    rec = last_cycle(args.log, args.label)
    if rec is None:
        print("sin ciclo registrado: no se actualiza", args.out)
        return 0
    prev = {}
    if os.path.exists(args.out):
        try:
            with open(args.out, encoding="utf-8") as fh:
                prev = json.load(fh)
        except (OSError, ValueError):
            prev = {}
    state = build_state(prev, rec)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=1, default=str)
    print("->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
