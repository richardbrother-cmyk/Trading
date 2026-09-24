"""Bot "ondas de Elliott" sobre el oro (XAUUSD) en barras de 4 h, en la cuenta demo agresiva de cTrader.

Variante desplegada (la de mas operaciones entre las que baten al azar en scripts/elliott_study.py): ZigZag de 2 ATR,
entrada al final de la onda 2 cuando un cierre de 4 h supera el maximo de la onda 1 ("ruptura"), stop bajo el inicio
de la onda 1 (nivel de invalidacion) menos 0,25 ATR, objetivo 1,618 veces la onda 1 desde el minimo de la onda 2,
solo largos, cierre forzoso a los 10 dias.

Ciclo (cada hora; solo actua sobre la ultima barra H4 cerrada):
1. Cierra las posiciones propias que lleven mas de `max_hold_days` abiertas (o todas si el freno lo ordena).
2. Recalcula pivotes y senales sobre las barras cerradas (autotrader/elliott.py, sin estado en memoria). Entra solo si
   la ultima barra cerrada es la que dispara la senal, cerro hace menos de `max_signal_age_hours` y esa senal no se
   opero ya (docs/elliott_state.json guarda la barra de la ultima senal operada). Stop y objetivo van con la orden.
3. Tamano: `risk_pct` del equity con lote minimo; si el minimo arriesga mas de `max_risk_pct`, no se opera.
Respeta BOT_HALT, el freno por drawdown de la cuenta (docs/aggr_state.json) y las ventanas de evento.
Las posiciones llevan la etiqueta ELLIOTT_LABEL: no toca las del bot agresivo, las del bot de Asia ni las manuales.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

from .ctrader import PRICE_SCALE, VOLUME_SCALE, CTraderSession, round_volume
from .elliott import ElliottParams, atr, entry_signals, pending_setup, zigzag
from .events import active_events, load_events
from .guard import evaluate as evaluate_guard
from .swingbot import BAR_HOURS, closed_h4_bars, default_max_risk_pct, size_units, spec_for

ELLIOTT_LABEL = "autotrader-elliott"
DEFAULT_MAX_SIGNAL_AGE_HOURS = 2.0
HISTORY_DAYS = 120
MIN_BARS = 30

DEPLOYED = ElliottParams(timeframe="H4", zz_atr=2.0, waves=(2,), entry="ruptura", target_ext=1.618, stop="inicio",
                         allow_short=False, max_hold_days=10.0)


def params_from_env(settings, max_risk_pct: float | None = None) -> ElliottParams:
    waves = tuple(int(w) for w in os.getenv("ELLIOTT_WAVES", "2").split(",") if w.strip())
    risk = settings.risk_per_trade
    return ElliottParams(timeframe="H4", zz_atr=float(os.getenv("ELLIOTT_ZZ_ATR", "2.0")), waves=waves or (2,),
                         entry=os.getenv("ELLIOTT_ENTRY", "ruptura"), max_wait_bars=int(os.getenv("ELLIOTT_MAX_WAIT_BARS", "20")),
                         target_ext=float(os.getenv("ELLIOTT_TARGET_EXT", "1.618")), stop=os.getenv("ELLIOTT_STOP", "inicio"),
                         stop_buffer_atr=float(os.getenv("ELLIOTT_STOP_BUFFER_ATR", "0.25")),
                         max_hold_days=float(os.getenv("ELLIOTT_MAX_HOLD_DAYS", "10")),
                         allow_short=os.getenv("ELLIOTT_ALLOW_SHORT", "false").lower() in {"1", "true", "yes"},
                         risk_pct=risk, max_risk_pct=max_risk_pct or default_max_risk_pct(risk))


def describe(p: ElliottParams) -> str:
    waves = "onda 2" if p.waves == (2,) else "onda 4" if p.waves == (4,) else "ondas 2 y 4"
    entry = "ruptura del extremo de la onda anterior" if p.entry == "ruptura" else "confirmacion del pivote"
    stop = "nivel de invalidacion" if p.stop == "inicio" else "minimo de la correccion"
    return (f"ZigZag {p.zz_atr:g} ATR, fin de la {waves}, entrada por {entry}, stop en el {stop} menos {p.stop_buffer_atr:g} ATR, "
            f"objetivo {p.target_ext:g} x onda 1, {'largos y cortos' if p.allow_short else 'solo largos'}, salida a los {p.max_hold_days:g} dias")


def _load_state(path: str) -> dict:
    if path and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {}
    return {}


def _pivots_out(pivots, index, digits: int) -> list[dict]:
    return [{"bar": index[pv.idx].strftime("%Y-%m-%d %H:%M"), "price": round(pv.price, digits), "kind": "max" if pv.kind == 1 else "min",
             "confirmed": index[pv.confirmed].strftime("%Y-%m-%d %H:%M")} for pv in pivots]


def run_elliott_cycle(settings, session: CTraderSession, p: ElliottParams | None = None, dry_run: bool = False, now: datetime | None = None,
                      label: str = ELLIOTT_LABEL, state_path: str = "docs/elliott_state.json", equity_cap: float | None = None,
                      max_signal_age_hours: float = DEFAULT_MAX_SIGNAL_AGE_HOURS) -> dict:
    now = now or datetime.now(timezone.utc)
    p = p or ElliottParams(**{**DEPLOYED.__dict__, "risk_pct": settings.risk_per_trade, "max_risk_pct": default_max_risk_pct(settings.risk_per_trade)})
    state = _load_state(state_path)
    balance, _d, _lev = session.trader()
    equity = balance + session.unrealized_pnl()
    sizing_equity = min(equity, equity_cap) if equity_cap else equity
    summary = {"timestamp": now.isoformat(timespec="seconds"), "kind": "elliott", "broker": "ctrader-elliott", "label": label,
               "strategy": describe(p), "equity": round(equity, 2), "sizing_equity": round(sizing_equity, 2), "dry_run": dry_run,
               "decisions": [], "orders": [], "closed": [], "skipped": [], "publish": False}
    events_now = []
    all_events = load_events(settings.events_path or None, now) if settings.event_mode != "off" else []
    if settings.event_mode != "off":
        events_now = active_events(all_events, now, settings.event_hours_before, settings.event_hours_after)
    if events_now:
        summary["event_window"] = [e.name for e in events_now]
    guard = evaluate_guard(equity, settings.halt_mode, settings.max_drawdown_pct, settings.history_path("docs/aggr_state.json"), settings.state_dir)
    summary["guard"] = guard.as_dict()
    if guard.blocks_entries:
        summary["skipped"].append(f"freno activo: {guard.reason}")

    own = [pos for pos in session.positions(only_bot=False) if pos.label == label]
    held = {pos.symbol for pos in own}
    summary["positions"] = {pos.symbol: pos.units for pos in own}

    # 1) salida por tiempo (o por el interruptor de cierre)
    for pos in own:
        opened = getattr(pos, "opened_at", None)
        reason = None
        if guard.closes_positions:
            reason = f"cierre por freno: {guard.reason}"
        elif opened is not None and now - opened > timedelta(days=p.max_hold_days):
            reason = "tiempo maximo"
        if not reason:
            continue
        if dry_run:
            summary["closed"].append({"symbol": pos.symbol, "units": pos.units, "reason": reason + " (dry run)"})
            continue
        try:
            session.call("ProtoOAClosePositionReq", timeout=30, ctidTraderAccountId=session.account_id, positionId=pos.position_id,
                         volume=int(pos.units * VOLUME_SCALE))
            summary["closed"].append({"symbol": pos.symbol, "units": pos.units, "reason": reason})
            held.discard(pos.symbol)
            summary["publish"] = True
        except Exception as exc:  # noqa: BLE001
            summary["skipped"].append(f"{pos.symbol}: error al cerrar ({reason}): {exc}")

    # 2) recuento y entradas
    traded = state.setdefault("traded_signals", {})
    for symbol in settings.symbols:
        info = session.symbols[symbol]
        try:
            bars = closed_h4_bars(session, symbol, days=HISTORY_DAYS, now=now)
        except Exception as exc:  # noqa: BLE001
            summary["skipped"].append(f"{symbol}: sin barras: {exc}")
            continue
        if len(bars) < MIN_BARS:
            summary["skipped"].append(f"{symbol}: historial insuficiente ({len(bars)} barras H4)")
            continue
        h, l, c = bars["high"].to_numpy(), bars["low"].to_numpy(), bars["close"].to_numpy()
        a = atr(bars, p.atr_period).to_numpy()
        setup = pending_setup(h, l, c, a, p)
        signals = entry_signals(h, l, c, a, p)
        last_i = len(bars) - 1
        sig = signals[-1] if signals and signals[-1]["bar"] == last_i else None
        bar_closed = bars.index[-1].to_pydatetime() + timedelta(hours=BAR_HOURS)
        age_h = (now - bar_closed).total_seconds() / 3600
        dec = {"symbol": symbol, "bar": bars.index[-1].strftime("%Y-%m-%d %H:%M"), "bar_age_h": round(age_h, 2),
               "close": round(float(c[-1]), info.digits), "atr": round(float(a[-1]), info.digits), "action": "BUY" if sig else "HOLD",
               "phase": setup["phase"] if setup else "sin patron"}
        if setup:
            dec.update({"side": setup["side"], "wave": setup["wave"], "trigger": round(setup["trigger"], info.digits),
                        "correction_low": round(setup["correction_low"], info.digits), "invalidation": round(setup["invalidation"], info.digits),
                        "retrace": setup["retrace"], "wave1": round(setup["wave1"], info.digits),
                        "pivots": _pivots_out(setup["pivots"], bars.index, info.digits)})
        else:
            thr = a * p.zz_atr if p.zz_atr > 0 else c * p.zz_pct
            dec["pivots"] = _pivots_out(zigzag(h, l, thr)[-5:], bars.index, info.digits)
        if sig:
            sig_key = f"{symbol}@{dec['bar']}"
            if symbol in held:
                dec["action"], dec["reason"] = "HOLD", "posicion abierta"
            elif sig_key in traded:
                dec["action"], dec["reason"] = "HOLD", "senal ya operada"
            elif guard.blocks_entries:
                dec["action"], dec["reason"] = "HOLD", "freno activo"
            elif events_now:
                dec["action"], dec["reason"] = "HOLD", "ventana de evento"
            elif age_h > max_signal_age_hours:
                dec["action"], dec["reason"] = "HOLD", f"senal caducada: la barra cerro hace {age_h:.1f} h (maximo {max_signal_age_hours:g} h)"
        summary["decisions"].append(dec)
        if dec["action"] != "BUY":
            continue
        s = sig["side"]
        spec = spec_for(symbol, info)
        entry_ref = float(c[-1])
        stop, target = float(sig["stop"]), float(sig["target"])
        stop_dist, tp_dist = s * (entry_ref - stop), s * (target - entry_ref)
        if stop_dist <= 0 or tp_dist <= 0:
            summary["skipped"].append(f"{symbol}: stop u objetivo no validos (entrada {entry_ref:.2f}, stop {stop:.2f}, objetivo {target:.2f})")
            continue
        units = size_units(sizing_equity, entry_ref, stop, spec, p.risk_pct, p.max_risk_pct)
        if units <= 0:
            summary["skipped"].append(f"{symbol}: el lote minimo arriesga mas del {p.max_risk_pct:.0%} de {sizing_equity:.0f} USD")
            continue
        volume = round_volume(units, info.min_volume, info.step_volume, info.max_volume)
        tick = 10 ** max(5 - info.digits, 0)
        rel_sl = max(int(round(stop_dist * PRICE_SCALE / tick)) * tick, tick)
        rel_tp = max(int(round(tp_dist * PRICE_SCALE / tick)) * tick, tick)
        order = {"symbol": symbol, "side": "buy" if s == 1 else "sell", "units": volume / VOLUME_SCALE, "entry_ref": round(entry_ref, info.digits),
                 "stop": round(stop, info.digits), "target": round(target, info.digits), "risk_usd": round(units * stop_dist, 2),
                 "wave": sig["wave"], "retrace": sig["retrace"], "bar": dec["bar"], "bar_age_h": dec["bar_age_h"]}
        if dry_run:
            order["status"] = "dry_run"
        else:
            try:
                res = session.call("ProtoOANewOrderReq", timeout=30, ctidTraderAccountId=session.account_id, symbolId=info.symbol_id,
                                   orderType=session.model.ProtoOAOrderType.MARKET,
                                   tradeSide=session.model.ProtoOATradeSide.BUY if s == 1 else session.model.ProtoOATradeSide.SELL,
                                   volume=volume, relativeStopLoss=rel_sl, relativeTakeProfit=rel_tp, label=label)
                order["status"] = session.model.ProtoOAExecutionType.Name(res.executionType).lower() if hasattr(res, "executionType") else "sent"
                traded[sig_key] = now.strftime("%Y-%m-%dT%H:%MZ")
                summary["publish"] = True
                held.add(symbol)
            except Exception as exc:  # noqa: BLE001
                order["status"] = f"error: {exc}"
        summary["orders"].append(order)

    # 3) estado publicado
    prev = state.get("last_decision") or {}
    cur = summary["decisions"][0] if summary["decisions"] else {}
    if (cur.get("phase"), cur.get("bar")) != (prev.get("phase"), prev.get("bar")):
        summary["publish"] = True
    try:
        deals = [d for d in session.deals(days=60) if d.get("closes") and d.get("label") == label]
        trades = [{"at": d["at"].strftime("%Y-%m-%dT%H:%MZ"), "symbol": d["symbol"], "units": d["units"], "entry": d["entry_price"], "exit": d["price"],
                   "net": d["net"], "opened_at": d["opened_at"].strftime("%Y-%m-%dT%H:%MZ") if d.get("opened_at") else None} for d in deals]
    except Exception as exc:  # noqa: BLE001
        trades = state.get("trades", [])
        summary["skipped"].append(f"sin historial de operaciones: {exc}")
    # no se acumulan claves de senales antiguas: solo las de los ultimos 30 dias
    cutoff = (now - timedelta(days=30)).strftime("%Y-%m-%dT%H:%MZ")
    state["traded_signals"] = {k: v for k, v in traded.items() if v >= cutoff}
    open_pos = [pos for pos in session.positions(only_bot=False) if pos.label == label] if not dry_run else own
    state.update({"at": now.strftime("%Y-%m-%dT%H:%MZ"), "equity": summary["equity"], "label": label, "strategy": summary["strategy"],
                  "last_decision": cur, "open_positions": [{"symbol": x.symbol, "side": x.side, "units": x.units, "entry": x.price, "stop": x.stop_loss,
                                                            "target": x.take_profit, "opened_at": x.opened_at.strftime("%Y-%m-%dT%H:%MZ") if x.opened_at else None}
                                                           for x in open_pos],
                  "last_orders": summary["orders"] or state.get("last_orders", []), "trades": trades, "guard": summary["guard"],
                  "params": {"zz_atr": p.zz_atr, "waves": list(p.waves), "entry": p.entry, "target_ext": p.target_ext, "stop": p.stop,
                             "stop_buffer_atr": p.stop_buffer_atr, "max_hold_days": p.max_hold_days, "risk_pct": p.risk_pct, "max_risk_pct": p.max_risk_pct,
                             "max_signal_age_hours": max_signal_age_hours}})
    if trades:
        net = [t["net"] for t in trades]
        state["stats"] = {"trades": len(net), "wins": sum(1 for n in net if n > 0), "net_usd": round(sum(net), 2)}
    if state_path:
        os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)
        with open(state_path, "w", encoding="utf-8") as fh:
            json.dump(state, fh, ensure_ascii=False, indent=1, default=str)
    os.makedirs(settings.state_dir, exist_ok=True)
    with open(os.path.join(settings.state_dir, "run_log.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, default=str) + "\n")
    return summary
