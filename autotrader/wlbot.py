"""Bot "largo semanal" sobre el oro (XAUUSD) en la cuenta demo agresiva de cTrader. PRUEBA: es la deriva alcista del oro, no una ventaja.

Regla (el "control" de scripts/gold_weekly_extremes.py, sin filtro):
1. Se compra XAUUSD a mercado en la apertura semanal (domingo 18:00 NY, primer ciclo de la semana de negociacion) y se mantiene.
2. Stop a `stop_atr` x ATR diario (media de los rangos de los 10 dias de negociacion anteriores, dia de 18:00 a 17:00 NY), enviado con la
   orden; sin objetivo. Una operacion por semana: si salta el stop no se vuelve a entrar hasta la semana siguiente.
3. El viernes a partir de las 16:00 NY se cierra a mercado (el mercado del oro cierra a las 17:00; la prueba historica salia a las 17:00).
   Si una posicion de la etiqueta propia sobrevive a la semana (ciclo perdido), se cierra en el primer ciclo de la semana siguiente.

Origen y limites: el estudio (README, "extremos semanales del oro") no encontro ventaja de calendario; esta regla cobra la subida del oro de
2023-2025 (en 2026 va negativa). Con el lote minimo de 1 onza el stop de 0,5 ATR (~50 USD) arriesga ~12 % de 440 USD: es una prueba con riesgo
alto por operacion, acotado por `max_risk_pct`. Respeta BOT_HALT y el freno por drawdown de la cuenta (docs/aggr_state.json).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .asiabot import _load_state, m15_bars
from .ctrader import PRICE_SCALE, VOLUME_SCALE, CTraderSession, round_volume
from .guard import evaluate as evaluate_guard

WL_LABEL = "autotrader-wl"
NY = ZoneInfo("America/New_York")
SYMBOL = "XAUUSD"
CLOSE_MIN = 16 * 60  # viernes, hora de Nueva York


@dataclass(frozen=True)
class WLParams:
    stop_atr: float = 0.5
    risk_pct: float = 0.06
    max_risk_pct: float = 0.15
    atr_days: int = 10


def trading_week(now_ny: pd.Timestamp) -> tuple[str, int]:
    """(clave de la semana de negociacion, dia 0=lunes..4=viernes). El dia de negociacion va de 18:00 a 17:00 NY."""
    shifted = now_ny + pd.Timedelta(hours=6)
    wd = shifted.weekday()
    start = (shifted - pd.Timedelta(days=wd)).normalize()
    return start.strftime("%Y-%m-%d"), wd


def daily_atr(bars_utc: pd.DataFrame, now: datetime, days: int = 10) -> float | None:
    """Media de los rangos de los `days` dias de negociacion completos anteriores a la semana en curso; None si no hay datos suficientes."""
    if bars_utc is None or bars_utc.empty:
        return None
    now_ny = pd.Timestamp(now).tz_convert(NY)
    ny = bars_utc.tz_convert(NY)
    closed = ny[ny.index + pd.Timedelta(minutes=15) <= now_ny]
    shifted = (closed.index + pd.Timedelta(hours=6)).tz_localize(None)  # hora local sin zona: 18:00 NY pasa a 00:00 del dia de negociacion
    closed = closed.assign(tday=shifted.normalize(), wd=shifted.weekday)
    prev = closed[(closed.tday < pd.Timestamp(trading_week(now_ny)[0])) & (closed.wd < 5)]
    if prev.empty:
        return None
    daily = prev.groupby("tday").agg(h=("high", "max"), l=("low", "min"), n=("close", "size"))
    daily = daily[daily.n >= 40].tail(days)
    if len(daily) < 8:
        return None
    return float((daily.h - daily.l).mean())


def size_units(equity: float, dist: float, info, p: WLParams) -> float:
    step, min_units = info.step_volume / VOLUME_SCALE, info.min_volume / VOLUME_SCALE
    if dist <= 0 or equity <= 0:
        return 0.0
    units = np.floor(equity * p.risk_pct / dist / step) * step
    if units < min_units:
        units = min_units if min_units * dist <= equity * p.max_risk_pct else 0.0
    return float(units)


def params_from_env(settings) -> WLParams:
    risk = settings.risk_per_trade
    return WLParams(stop_atr=float(os.getenv("WL_STOP_ATR", "0.5")), risk_pct=risk,
                    max_risk_pct=float(os.getenv("WL_MAX_RISK_PCT", "0") or 0) or max(0.03, 1.5 * risk))


def run_wl_cycle(settings, session: CTraderSession, p: WLParams | None = None, dry_run: bool = False, now: datetime | None = None,
                 label: str = WL_LABEL, state_path: str = "docs/wl_state.json", symbol: str = SYMBOL) -> dict:
    now = now or datetime.now(timezone.utc)
    p = p or WLParams(risk_pct=settings.risk_per_trade)
    now_ny = pd.Timestamp(now).tz_convert(NY)
    week, wd = trading_week(now_ny)
    state = _load_state(state_path)
    balance, _d, _lev = session.trader()
    equity = balance + session.unrealized_pnl()
    summary = {"timestamp": now.isoformat(timespec="seconds"), "kind": "wl", "label": label, "equity": round(equity, 2), "dry_run": dry_run,
               "ny_time": now_ny.strftime("%Y-%m-%d %H:%M"), "week": week, "plan": {}, "orders": [], "closed": [], "skipped": [], "publish": False}
    own = [pos for pos in session.positions(only_bot=False) if pos.label == label]
    summary["open_positions"] = [{"symbol": x.symbol, "side": x.side, "units": x.units, "entry": x.price} for x in own]
    minutes = now_ny.hour * 60 + now_ny.minute
    ny_wd = now_ny.weekday()
    last_week = state.get("last_trade_week")

    # 1) cierre: viernes desde las 16:00 NY, o posicion de una semana anterior
    time_exit = ny_wd == 4 and minutes >= CLOSE_MIN
    stale = bool(last_week) and last_week != week and wd < 5 and ny_wd != 5
    if own and (time_exit or stale):
        reason = "cierre viernes 16:00 NY" if time_exit else "posicion de la semana anterior"
        for pos in own:
            if dry_run:
                summary["closed"].append({"symbol": pos.symbol, "units": pos.units, "reason": reason + " (dry run)"})
                continue
            try:
                session.call("ProtoOAClosePositionReq", timeout=30, ctidTraderAccountId=session.account_id, positionId=pos.position_id,
                             volume=int(pos.units * VOLUME_SCALE))
                summary["closed"].append({"symbol": pos.symbol, "units": pos.units, "reason": reason})
                summary["publish"] = True
            except Exception as exc:  # noqa: BLE001
                summary["skipped"].append(f"{pos.symbol}: error al cerrar: {exc}")
        own = []

    guard = evaluate_guard(equity, settings.halt_mode, settings.max_drawdown_pct, settings.history_path("docs/aggr_state.json"), settings.state_dir,
                           reset_at=settings.peak_reset_at)
    summary["guard"] = guard.as_dict()

    # 2) entrada: primer ciclo de la semana de negociacion (lunes de negociacion = domingo 18:00 a lunes 17:00 NY)
    plan = {"week": week, "phase": "fuera_de_ventana", "atr": None}
    if wd != 0 or ny_wd in (4, 5):
        summary["skipped"].append("fuera de la ventana de entrada (domingo 18:00 a lunes 17:00 NY)")
    elif symbol not in session.symbols:
        summary["skipped"].append(f"{symbol}: simbolo no disponible en la cuenta")
    elif last_week == week or any(x.symbol == symbol for x in own):
        plan["phase"] = "operada"
        summary["skipped"].append(f"{symbol}: ya se opero esta semana")
    else:
        try:
            bars = m15_bars(session, symbol, days=21, now=now)
        except Exception as exc:  # noqa: BLE001
            bars = None
            summary["skipped"].append(f"{symbol}: sin barras: {exc}")
        atr = daily_atr(bars, now, p.atr_days) if bars is not None else None
        if atr is None:
            plan["phase"] = "sin_datos"
            summary["skipped"].append(f"{symbol}: sin ATR diario suficiente")
        else:
            plan.update({"phase": "entrada", "atr": round(atr, 2)})
            if guard.blocks_entries:
                summary["skipped"].append(f"{symbol}: freno activo: {guard.reason}")
            else:
                info = session.symbols[symbol]
                dist = p.stop_atr * atr
                units = size_units(equity, dist, info, p)
                ref = float(bars.close.iloc[-1])
                order = {"symbol": symbol, "side": "buy", "units": units, "entry_ref": round(ref, 2), "stop": round(ref - dist, 2),
                         "risk_usd": round(units * dist, 2), "atr": round(atr, 2)}
                if units <= 0:
                    summary["skipped"].append(f"{symbol}: sin tamano valido (stop {dist:.2f} USD, riesgo maximo {p.max_risk_pct:.1%})")
                elif dry_run:
                    order["status"] = "dry_run"
                    summary["orders"].append(order)
                else:
                    tick = 10 ** max(5 - info.digits, 0)
                    rel_sl = max(int(round(dist * PRICE_SCALE / tick)) * tick, tick)
                    volume = round_volume(units, info.min_volume, info.step_volume, info.max_volume)
                    try:
                        res = session.call("ProtoOANewOrderReq", timeout=30, ctidTraderAccountId=session.account_id, symbolId=info.symbol_id,
                                           orderType=session.model.ProtoOAOrderType.MARKET, tradeSide=session.model.ProtoOATradeSide.BUY,
                                           volume=volume, relativeStopLoss=rel_sl, label=label)
                        order["status"] = session.model.ProtoOAExecutionType.Name(res.executionType).lower() if hasattr(res, "executionType") else "sent"
                        last_week = week
                        plan["phase"] = "operada"
                        summary["publish"] = True
                    except Exception as exc:  # noqa: BLE001
                        order["status"] = f"error: {exc}"
                    summary["orders"].append(order)
    summary["plan"] = plan

    # 3) estado publicado
    if plan["phase"] != (state.get("plan") or {}).get("phase") or plan.get("week") != (state.get("plan") or {}).get("week"):
        summary["publish"] = True
    try:
        deals = [d for d in session.deals(days=45) if d.get("closes") and d.get("label") == label]
        trades = [{"at": d["at"].strftime("%Y-%m-%dT%H:%MZ"), "symbol": d["symbol"], "units": d["units"], "entry": d["entry_price"], "exit": d["price"],
                   "net": d["net"], "opened_at": d["opened_at"].strftime("%Y-%m-%dT%H:%MZ") if d.get("opened_at") else None} for d in deals]
    except Exception as exc:  # noqa: BLE001
        trades = state.get("trades", [])
        summary["skipped"].append(f"sin historial de operaciones: {exc}")
    state.update({"at": now.strftime("%Y-%m-%dT%H:%MZ"), "ny_time": summary["ny_time"], "equity": summary["equity"],
                  "plan": plan if plan["phase"] != "fuera_de_ventana" or not state.get("plan") else {**state["plan"], "outside": True},
                  "open_positions": summary["open_positions"], "last_orders": summary["orders"] or state.get("last_orders", []),
                  "last_trade_week": last_week, "trades": trades,
                  "params": {"stop_atr": p.stop_atr, "risk_pct": p.risk_pct, "max_risk_pct": p.max_risk_pct, "atr_days": p.atr_days},
                  "guard": summary["guard"], "label": label, "symbol": symbol})
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
