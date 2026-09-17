"""Bot swing en vivo para cTrader: reversion en bandas de Bollinger de 4 h, solo largos, 1-3 dias.

Ciclo (cada hora; solo actua sobre la ultima barra H4 cerrada):
1. Cierra las posiciones propias que lleven mas de `max_hold_days` abiertas.
2. Para cada simbolo sin posicion propia: si la ultima barra H4 CERRADA cierra bajo la banda inferior
   con RSI < 30 (y no hay ventana de evento), compra a mercado con stop a `stop_atr` ATR y objetivo
   en la media de las bandas, ambos enviados con la orden y vigilados por el broker. La senal caduca:
   solo se entra si esa barra cerro hace menos de `max_signal_age_hours` (el planificador de GitHub
   Actions llega con retraso y una entrada tardia ya no es la que probo el backtest).
3. Tamano: `risk_pct` del equity (acotado por `equity_cap`), con lote minimo; si el minimo arriesga
   mas de `max_risk_pct`, no se opera ese simbolo.
Las posiciones llevan la etiqueta SWING_LABEL: este bot no toca las del bot tendencial ni las manuales.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

from .ctrader import PRICE_SCALE, VOLUME_SCALE, CTraderSession, decode_trendbars, relative_stop, round_volume
from .events import active_events, load_events
from .intraday import SPECS, SymbolSpec
from .swing import SwingParams, indicators

SWING_LABEL = "autotrader-swing"
PERIOD_H4 = 10
BAR_HOURS = 4
DEFAULT_MAX_SIGNAL_AGE_HOURS = 2.0


def closed_h4_bars(session: CTraderSession, symbol: str, days: int = 60, now: datetime | None = None) -> pd.DataFrame:
    """Barras H4 cerradas (descarta la barra en formacion)."""
    now = now or datetime.now(timezone.utc)
    info = session.symbols[symbol]
    frm = now - timedelta(days=days)
    res = session.call("ProtoOAGetTrendbarsReq", timeout=40, ctidTraderAccountId=session.account_id,
                       fromTimestamp=int(frm.timestamp() * 1000), toTimestamp=int(now.timestamp() * 1000),
                       period=PERIOD_H4, symbolId=info.symbol_id)
    df = decode_trendbars(list(res.trendbar), period_minutes=240)
    if df.empty:
        return df
    df.index = df.index.tz_localize("UTC")
    cutoff = pd.Timestamp(now) - pd.Timedelta(hours=4)
    return df[df.index <= cutoff]


def spec_for(symbol: str, info) -> SymbolSpec:
    """Usa la especificacion conocida o construye una a partir de los datos del broker."""
    if symbol in SPECS:
        return SPECS[symbol]
    return SymbolSpec(symbol, "00:00", "24:00", 0.0, info.step_volume / VOLUME_SCALE, info.min_volume / VOLUME_SCALE, info.digits)


def size_units(equity: float, entry: float, stop: float, spec: SymbolSpec, risk_pct: float, max_risk_pct: float) -> float:
    dist = abs(entry - stop)
    if dist <= 0:
        return 0.0
    import math
    units = math.floor(equity * risk_pct / dist / spec.step) * spec.step
    if units < spec.min_units:
        units = spec.min_units if spec.min_units * dist <= equity * max_risk_pct else 0.0
    return float(round(units, 6))


def default_max_risk_pct(risk_pct: float) -> float:
    """Tope para aceptar el lote minimo: 3 % o 1.5x el riesgo objetivo, lo que sea mayor."""
    return max(0.03, risk_pct * 1.5)


def run_swing_cycle(settings, session: CTraderSession, params: SwingParams | None = None, equity_cap: float | None = None,
                    dry_run: bool = False, now: datetime | None = None, max_risk_pct: float | None = None,
                    max_signal_age_hours: float = DEFAULT_MAX_SIGNAL_AGE_HOURS) -> dict:
    now = now or datetime.now(timezone.utc)
    p = params or SwingParams("bands", "H4", stop_atr=2.0, allow_short=False, risk_pct=settings.risk_per_trade,
                              max_risk_pct=max_risk_pct or default_max_risk_pct(settings.risk_per_trade), max_hold_days=3.0)
    balance, _d, _lev = session.trader()
    equity = balance + session.unrealized_pnl()
    sizing_equity = min(equity, equity_cap) if equity_cap else equity
    summary = {"timestamp": now.isoformat(timespec="seconds"), "kind": "swing", "broker": "ctrader-swing", "equity": round(equity, 2),
               "sizing_equity": round(sizing_equity, 2), "dry_run": dry_run, "decisions": [], "orders": [], "closed": [], "skipped": []}
    events_now = []
    if settings.event_mode != "off":
        events_now = active_events(load_events(settings.events_path or None), now, settings.event_hours_before, settings.event_hours_after)
    if events_now:
        summary["event_window"] = [e.name for e in events_now]

    own = [pos for pos in session.positions(only_bot=False) if pos.label == SWING_LABEL and pos.side == "buy"]
    held = {pos.symbol for pos in own}
    summary["positions"] = {pos.symbol: pos.units for pos in own}

    # 1) salida por tiempo
    for pos in own:
        opened = getattr(pos, "opened_at", None)
        if opened is not None and now - opened > timedelta(days=p.max_hold_days):
            if not dry_run:
                try:
                    session.call("ProtoOAClosePositionReq", timeout=30, ctidTraderAccountId=session.account_id,
                                 positionId=pos.position_id, volume=int(pos.units * VOLUME_SCALE))
                    summary["closed"].append({"symbol": pos.symbol, "units": pos.units, "reason": "tiempo maximo"})
                    held.discard(pos.symbol)
                except Exception as exc:  # noqa: BLE001
                    summary["skipped"].append(f"{pos.symbol}: error al cerrar por tiempo: {exc}")
            else:
                summary["closed"].append({"symbol": pos.symbol, "units": pos.units, "reason": "tiempo maximo (dry run)"})

    # 2) entradas
    for symbol in settings.symbols:
        info = session.symbols[symbol]
        try:
            bars = closed_h4_bars(session, symbol, now=now)
        except Exception as exc:  # noqa: BLE001
            summary["skipped"].append(f"{symbol}: sin barras: {exc}")
            continue
        if len(bars) < 210:
            summary["skipped"].append(f"{symbol}: historial insuficiente ({len(bars)} barras H4)")
            continue
        d = indicators(bars, p)
        last = d.iloc[-1]
        signal = bool(last["close"] < last["bb_lo"] and last["rsi"] < 30)
        dec = {"symbol": symbol, "close": round(float(last["close"]), info.digits), "bb_lo": round(float(last["bb_lo"]), info.digits),
               "bb_mid": round(float(last["bb_mid"]), info.digits), "rsi": round(float(last["rsi"]), 1), "atr": round(float(last["atr"]), info.digits),
               "bar": bars.index[-1].strftime("%Y-%m-%d %H:%M"), "action": "BUY" if signal else "HOLD"}
        bar_closed = bars.index[-1].to_pydatetime() + timedelta(hours=BAR_HOURS)
        age_h = (now - bar_closed).total_seconds() / 3600
        dec["bar_age_h"] = round(age_h, 2)
        if symbol in held:
            dec["action"] = "HOLD"
            dec["reason"] = "posicion abierta"
        elif signal and events_now:
            dec["action"] = "HOLD"
            dec["reason"] = "ventana de evento"
        elif signal and age_h > max_signal_age_hours:
            dec["action"] = "HOLD"
            dec["reason"] = f"senal caducada: la barra cerro hace {age_h:.1f} h (maximo {max_signal_age_hours:g} h)"
        summary["decisions"].append(dec)
        if dec["action"] != "BUY":
            continue
        spec = spec_for(symbol, info)
        entry = float(last["close"])
        stop_dist = p.stop_atr * float(last["atr"])
        units = size_units(sizing_equity, entry, entry - stop_dist, spec, p.risk_pct, p.max_risk_pct)
        if units <= 0:
            summary["skipped"].append(f"{symbol}: el lote minimo arriesga mas del {p.max_risk_pct:.0%} de {sizing_equity:.0f} USD")
            continue
        tp_dist = float(last["bb_mid"]) - entry
        if tp_dist <= 0:
            summary["skipped"].append(f"{symbol}: objetivo no valido")
            continue
        volume = round_volume(units, info.min_volume, info.step_volume, info.max_volume)
        tick = 10 ** max(5 - info.digits, 0)
        rel_sl = relative_stop(entry, stop_dist / entry, info.digits)
        rel_tp = max(int(round(tp_dist * PRICE_SCALE / tick)) * tick, tick)
        order = {"symbol": symbol, "units": volume / VOLUME_SCALE, "entry_ref": entry, "stop": round(entry - stop_dist, info.digits),
                 "target": round(float(last["bb_mid"]), info.digits), "risk_usd": round(units * stop_dist, 2)}
        if dry_run:
            order["status"] = "dry_run"
        else:
            try:
                res = session.call("ProtoOANewOrderReq", timeout=30, ctidTraderAccountId=session.account_id, symbolId=info.symbol_id,
                                   orderType=session.model.ProtoOAOrderType.MARKET, tradeSide=session.model.ProtoOATradeSide.BUY,
                                   volume=volume, relativeStopLoss=rel_sl, relativeTakeProfit=rel_tp, label=SWING_LABEL)
                order["status"] = session.model.ProtoOAExecutionType.Name(res.executionType).lower() if hasattr(res, "executionType") else "sent"
            except Exception as exc:  # noqa: BLE001
                order["status"] = f"error: {exc}"
        summary["orders"].append(order)
    os.makedirs(settings.state_dir, exist_ok=True)
    with open(os.path.join(settings.state_dir, "run_log.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, default=str) + "\n")
    return summary
