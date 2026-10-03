"""Bot "momentum de la primera hora" en indices (US500 y NAS100), intradia, en la cuenta demo agresiva de cTrader.

Regla (variante "umbral 0,5 U, stop 1 U, salida 16:00" de scripts/intraday_vwap_momentum.py, subconjunto POST HOC de indices):
1. Primera hora de la sesion de Nueva York: velas M15 de las 09:30, 09:45, 10:00 y 10:15. r = cierre de la ultima - apertura de la primera.
2. U = 2 x ATR(14) de M15 (escala de una hora, Wilder) medido en la vela de las 10:15. Si |r| < `threshold` x U no hay operacion hoy.
3. A las 10:30 se entra a mercado a favor de r (largo si sube, corto si baja). Stop a `stop_u` x U del precio de entrada, enviado con
   la orden; sin objetivo. Solo se entra si la senal esta fresca (hasta `max_late_min` minutos despues de las 10:30).
4. A las 16:00 de Nueva York se cierra lo que siga abierto. Una operacion por simbolo y dia como maximo.

Es una prueba en demo de un efecto que NO supero la correccion por multiplicidad (q 0,17, IC 95 % [-0,02, +0,21] R); se evalua con
datos nuevos y riesgo pequeno. El bot recalcula el plan en cada ciclo (cada 15 min) a partir de las velas cerradas, sin estado en
memoria. Respeta BOT_HALT y el freno por drawdown de la cuenta (docs/aggr_state.json).
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

FH_LABEL = "autotrader-fh"
NY = ZoneInfo("America/New_York")
SYMBOLS = ("US500", "NAS100")
OPEN_MIN = 9 * 60 + 30
DECISION_MIN = 10 * 60 + 30
CLOSE_MIN = 16 * 60


@dataclass(frozen=True)
class FHParams:
    threshold: float = 0.5  # |r| minimo en unidades de U
    stop_u: float = 1.0
    max_late_min: int = 30  # la senal caduca 30 min despues de las 10:30
    risk_pct: float = 0.02
    max_risk_pct: float = 0.03


def atr14(bars: pd.DataFrame) -> pd.Series:
    """ATR de Wilder (EWM alpha 1/14) sobre toda la serie continua de velas, como en el estudio."""
    tr = pd.concat([(bars.high - bars.low), (bars.high - bars.close.shift()).abs(), (bars.low - bars.close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / 14, adjust=False).mean()


def plan_day(bars_utc: pd.DataFrame, now: datetime, p: FHParams) -> dict:
    """Plan del dia para un simbolo a partir de las velas cerradas hasta `now`.

    phase: antes_de_las_10_30 | sin_datos | sin_senal | ready (entrar ahora) | late (senal caducada) | abierto (ya pasado el cierre)
    """
    now_ny = pd.Timestamp(now).tz_convert(NY)
    day = now_ny.normalize()
    out = {"day": day.strftime("%Y-%m-%d"), "phase": "sin_datos", "side": 0, "r": None, "u": None, "entry_ref": None}
    if bars_utc is None or bars_utc.empty:
        return out
    ny = bars_utc.tz_convert(NY)
    closed = ny[ny.index + pd.Timedelta(minutes=15) <= now_ny]
    if closed.empty:
        return out
    atr = atr14(closed)
    first = closed[(closed.index >= day + pd.Timedelta(minutes=OPEN_MIN)) & (closed.index < day + pd.Timedelta(minutes=DECISION_MIN))]
    if len(first) < 4 or first.index[0] != day + pd.Timedelta(minutes=OPEN_MIN):
        out["phase"] = "antes_de_las_10_30" if now_ny < day + pd.Timedelta(minutes=DECISION_MIN) else "sin_datos"
        return out
    first = first.iloc[:4]
    u = 2.0 * float(atr.loc[first.index[3]])
    r = float(first.close.iloc[3] - first.open.iloc[0])
    out.update({"r": round(r, 2), "u": round(u, 2), "entry_ref": float(first.close.iloc[3]), "first_hour_open": float(first.open.iloc[0]),
                "ratio": round(abs(r) / u, 2) if u > 0 else None})
    if u <= 0 or r == 0 or abs(r) < p.threshold * u:
        out["phase"] = "sin_senal"
        return out
    out["side"] = 1 if r > 0 else -1
    minutes = now_ny.hour * 60 + now_ny.minute
    out["phase"] = "ready" if minutes <= DECISION_MIN + p.max_late_min else "late"
    return out


def size_units(equity: float, dist: float, info, p: FHParams) -> float:
    step, min_units = info.step_volume / VOLUME_SCALE, info.min_volume / VOLUME_SCALE
    if dist <= 0 or equity <= 0:
        return 0.0
    units = np.floor(equity * p.risk_pct / dist / step) * step
    if units < min_units:
        units = min_units if min_units * dist <= equity * p.max_risk_pct else 0.0
    return float(units)


def params_from_env(settings) -> FHParams:
    risk = settings.risk_per_trade
    return FHParams(threshold=float(os.getenv("FH_THRESHOLD", "0.5")), stop_u=float(os.getenv("FH_STOP_U", "1.0")),
                    max_late_min=int(os.getenv("FH_MAX_LATE_MIN", "30")), risk_pct=risk,
                    max_risk_pct=float(os.getenv("FH_MAX_RISK_PCT", "0") or 0) or max(0.03, 1.5 * risk))


def run_fh_cycle(settings, session: CTraderSession, p: FHParams | None = None, dry_run: bool = False, now: datetime | None = None,
                 label: str = FH_LABEL, state_path: str = "docs/fh_state.json", symbols: tuple[str, ...] = SYMBOLS) -> dict:
    now = now or datetime.now(timezone.utc)
    p = p or FHParams(risk_pct=settings.risk_per_trade)
    now_ny = pd.Timestamp(now).tz_convert(NY)
    state = _load_state(state_path)
    balance, _d, _lev = session.trader()
    equity = balance + session.unrealized_pnl()
    summary = {"timestamp": now.isoformat(timespec="seconds"), "kind": "fh", "label": label, "equity": round(equity, 2), "dry_run": dry_run,
               "ny_time": now_ny.strftime("%Y-%m-%d %H:%M"), "plans": {}, "orders": [], "closed": [], "skipped": [], "publish": False}
    own = [pos for pos in session.positions(only_bot=False) if pos.label == label]
    summary["open_positions"] = [{"symbol": x.symbol, "side": x.side, "units": x.units, "entry": x.price} for x in own]
    minutes = now_ny.hour * 60 + now_ny.minute
    weekday = now_ny.weekday()

    # 1) cierre a las 16:00 NY de lo que siga abierto
    if own and minutes >= CLOSE_MIN:
        for pos in own:
            if dry_run:
                summary["closed"].append({"symbol": pos.symbol, "units": pos.units, "reason": "cierre 16:00 NY (dry run)"})
                continue
            try:
                session.call("ProtoOAClosePositionReq", timeout=30, ctidTraderAccountId=session.account_id, positionId=pos.position_id,
                             volume=int(pos.units * VOLUME_SCALE))
                summary["closed"].append({"symbol": pos.symbol, "units": pos.units, "reason": "cierre 16:00 NY"})
                summary["publish"] = True
            except Exception as exc:  # noqa: BLE001
                summary["skipped"].append(f"{pos.symbol}: error al cerrar: {exc}")
        own = []

    guard = evaluate_guard(equity, settings.halt_mode, settings.max_drawdown_pct, settings.history_path("docs/aggr_state.json"), settings.state_dir)
    summary["guard"] = guard.as_dict()

    # 2) plan y entradas
    last_trade = dict(state.get("last_trade_day") or {})
    if weekday >= 5 or minutes < DECISION_MIN or minutes >= CLOSE_MIN:
        summary["skipped"].append("fuera de la ventana de decision (10:30-16:00 NY, lunes a viernes)")
    else:
        for symbol in symbols:
            if symbol not in session.symbols:
                summary["skipped"].append(f"{symbol}: simbolo no disponible en la cuenta")
                continue
            try:
                bars = m15_bars(session, symbol, days=4, now=now)
            except Exception as exc:  # noqa: BLE001
                summary["skipped"].append(f"{symbol}: sin barras: {exc}")
                continue
            plan = plan_day(bars, now, p)
            summary["plans"][symbol] = plan
            if plan["phase"] != "ready":
                continue
            held = any(x.symbol == symbol for x in own)
            if last_trade.get(symbol) == plan["day"] or held:
                summary["skipped"].append(f"{symbol}: ya se opero hoy")
                continue
            if guard.blocks_entries:
                summary["skipped"].append(f"{symbol}: freno activo: {guard.reason}")
                continue
            info = session.symbols[symbol]
            side, entry_ref = plan["side"], plan["entry_ref"]
            dist = p.stop_u * plan["u"]
            units = size_units(equity, dist, info, p)
            order = {"symbol": symbol, "side": "buy" if side == 1 else "sell", "units": units, "entry_ref": round(entry_ref, 2),
                     "stop": round(entry_ref - side * dist, 2), "risk_usd": round(units * dist, 2), "first_hour_move": plan["r"], "u": plan["u"]}
            if units <= 0:
                summary["skipped"].append(f"{symbol}: sin tamano valido (distancia {dist:.2f}, riesgo maximo {p.max_risk_pct:.1%})")
            elif dry_run:
                order["status"] = "dry_run"
                summary["orders"].append(order)
            else:
                tick = 10 ** max(5 - info.digits, 0)
                rel_sl = max(int(round(dist * PRICE_SCALE / tick)) * tick, tick)
                volume = round_volume(units, info.min_volume, info.step_volume, info.max_volume)
                try:
                    res = session.call("ProtoOANewOrderReq", timeout=30, ctidTraderAccountId=session.account_id, symbolId=info.symbol_id,
                                       orderType=session.model.ProtoOAOrderType.MARKET,
                                       tradeSide=session.model.ProtoOATradeSide.BUY if side == 1 else session.model.ProtoOATradeSide.SELL,
                                       volume=volume, relativeStopLoss=rel_sl, label=label)
                    order["status"] = session.model.ProtoOAExecutionType.Name(res.executionType).lower() if hasattr(res, "executionType") else "sent"
                    last_trade[symbol] = plan["day"]
                    summary["publish"] = True
                except Exception as exc:  # noqa: BLE001
                    order["status"] = f"error: {exc}"
                summary["orders"].append(order)

    # 3) estado publicado
    prev_plans = {k: (v or {}).get("phase") for k, v in (state.get("plans") or {}).items()}
    if {k: v.get("phase") for k, v in summary["plans"].items()} != {k: prev_plans.get(k) for k in summary["plans"]}:
        summary["publish"] = True
    try:
        deals = [d for d in session.deals(days=45) if d.get("closes") and d.get("label") == label]
        trades = [{"at": d["at"].strftime("%Y-%m-%dT%H:%MZ"), "symbol": d["symbol"], "units": d["units"], "entry": d["entry_price"], "exit": d["price"],
                   "net": d["net"], "opened_at": d["opened_at"].strftime("%Y-%m-%dT%H:%MZ") if d.get("opened_at") else None} for d in deals]
    except Exception as exc:  # noqa: BLE001
        trades = state.get("trades", [])
        summary["skipped"].append(f"sin historial de operaciones: {exc}")
    plans = summary["plans"] or state.get("plans", {})
    state.update({"at": now.strftime("%Y-%m-%dT%H:%MZ"), "ny_time": summary["ny_time"], "equity": summary["equity"], "plans": plans,
                  "open_positions": summary["open_positions"], "last_orders": summary["orders"] or state.get("last_orders", []),
                  "last_trade_day": last_trade, "trades": trades,
                  "params": {"threshold": p.threshold, "stop_u": p.stop_u, "max_late_min": p.max_late_min, "risk_pct": p.risk_pct,
                             "max_risk_pct": p.max_risk_pct}, "guard": summary["guard"], "label": label, "symbols": list(symbols)})
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
