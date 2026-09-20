"""Bot "retest de Asia" sobre el oro (XAUUSD), intradia, en la cuenta demo agresiva de cTrader.

Regla (la unica variante consistente del estudio scripts/london_range_gold.py):
1. Rango de Asia: maximo y minimo entre las 19:00 de la vispera y las 03:00 de Nueva York.
2. En la manana de Nueva York (08:00-12:00) se espera la primera vela de 15 min que CIERRA fuera del rango.
3. Despues el precio debe volver a tocar el nivel roto (retest) en las `max_wait` velas siguientes sin que ninguna
   cierre de vuelta dentro del rango. La entrada es en la apertura de la vela siguiente al retest, a favor de la ruptura.
4. Stop en la mitad del rango de Asia; objetivo `rr` veces la distancia al stop; ambos enviados con la orden.
5. A las 12:00 de Nueva York se cierra lo que siga abierto. Una operacion por dia como maximo.

El bot corre cada 15 minutos desde GitHub Actions; en cada ciclo recalcula el plan del dia a partir de las velas cerradas,
asi que no depende de un estado en memoria. Riesgo por operacion = RISK_PER_TRADE del equity; el lote minimo se acepta si
no arriesga mas de ASIA_MAX_RISK_PCT. Respeta BOT_HALT y el freno por drawdown de la cuenta (docs/aggr_state.json).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .ctrader import PRICE_SCALE, VOLUME_SCALE, CTraderSession, decode_trendbars, round_volume
from .guard import evaluate as evaluate_guard

ASIA_LABEL = "autotrader-asia"
NY = ZoneInfo("America/New_York")
PERIOD_M15 = 7
SYMBOL = "XAUUSD"


@dataclass(frozen=True)
class AsiaParams:
    rr: float = 2.0
    max_wait_bars: int = 8
    risk_pct: float = 0.03
    max_risk_pct: float = 0.045
    asia_start: int = 19 * 60  # minutos NY de la vispera
    asia_end: int = 3 * 60
    ny_start: int = 8 * 60
    ny_end: int = 12 * 60


def m15_bars(session: CTraderSession, symbol: str = SYMBOL, days: int = 3, now: datetime | None = None) -> pd.DataFrame:
    """Barras M15 (UTC) de los ultimos dias, incluida la que esta en formacion."""
    now = now or datetime.now(timezone.utc)
    info = session.symbols[symbol]
    res = session.call("ProtoOAGetTrendbarsReq", timeout=40, ctidTraderAccountId=session.account_id,
                       fromTimestamp=int((now - timedelta(days=days)).timestamp() * 1000), toTimestamp=int(now.timestamp() * 1000),
                       period=PERIOD_M15, symbolId=info.symbol_id)
    df = decode_trendbars(list(res.trendbar), period_minutes=15)
    if not df.empty:
        df.index = df.index.tz_localize("UTC")
    return df


def day_frames(bars_utc: pd.DataFrame, now: datetime, p: AsiaParams) -> tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]:
    """(dia de negociacion NY, barras de Asia, barras cerradas de la manana de NY hasta `now`)."""
    now_ny = pd.Timestamp(now).tz_convert(NY)
    day = now_ny.normalize()
    ny = bars_utc.tz_convert(NY)
    closed = ny[ny.index + pd.Timedelta(minutes=15) <= now_ny]  # solo velas cerradas
    asia = closed[(closed.index >= day - pd.Timedelta(days=1) + pd.Timedelta(minutes=p.asia_start)) & (closed.index < day + pd.Timedelta(minutes=p.asia_end))]
    nym = closed[(closed.index >= day + pd.Timedelta(minutes=p.ny_start)) & (closed.index < day + pd.Timedelta(minutes=p.ny_end))]
    return day, asia, nym


def plan_day(asia_hi: float, asia_lo: float, nym: pd.DataFrame, p: AsiaParams) -> dict:
    """Estado del dia a partir de las velas cerradas de la manana de NY.

    phase: waiting_break | broke | failed | retest | ready (la vela del retest es la ultima cerrada: toca entrar) | expired
    """
    out = {"phase": "waiting_break", "side": 0, "level": None, "break_at": None, "retest_at": None, "stop": (asia_hi + asia_lo) / 2}
    if asia_hi <= asia_lo or nym.empty:
        return out
    side, i0 = 0, None
    for i, (t, b) in enumerate(nym.iterrows()):
        if b.close > asia_hi:
            side, i0 = 1, i; break
        if b.close < asia_lo:
            side, i0 = -1, i; break
    if side == 0:
        return out
    level = asia_hi if side == 1 else asia_lo
    out.update({"phase": "broke", "side": side, "level": level, "break_at": nym.index[i0].strftime("%H:%M")})
    for j in range(i0 + 1, len(nym)):
        b = nym.iloc[j]
        if (side == 1 and b.close < asia_hi) or (side == -1 and b.close > asia_lo):
            out["phase"] = "failed"
            return out
        if (side == 1 and b.low <= level) or (side == -1 and b.high >= level):
            out.update({"phase": "ready" if j == len(nym) - 1 else "retest", "retest_at": nym.index[j].strftime("%H:%M")})
            return out
        if j - i0 >= p.max_wait_bars:
            out["phase"] = "expired"
            return out
    return out


def size_units(equity: float, dist: float, info, p: AsiaParams) -> float:
    step, min_units = info.step_volume / VOLUME_SCALE, info.min_volume / VOLUME_SCALE
    if dist <= 0 or equity <= 0:
        return 0.0
    units = np.floor(equity * p.risk_pct / dist / step) * step
    if units < min_units:
        units = min_units if min_units * dist <= equity * p.max_risk_pct else 0.0
    return float(units)


def params_from_env(settings) -> AsiaParams:
    risk = settings.risk_per_trade
    return AsiaParams(rr=float(os.getenv("ASIA_RR", "2.0")), max_wait_bars=int(os.getenv("ASIA_MAX_WAIT_BARS", "8")), risk_pct=risk,
                      max_risk_pct=float(os.getenv("ASIA_MAX_RISK_PCT", "0") or 0) or max(0.03, 1.5 * risk))


def _load_state(path: str) -> dict:
    if path and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {}
    return {}


def run_asia_cycle(settings, session: CTraderSession, p: AsiaParams | None = None, dry_run: bool = False, now: datetime | None = None,
                   label: str = ASIA_LABEL, state_path: str = "docs/asia_state.json", symbol: str = SYMBOL) -> dict:
    now = now or datetime.now(timezone.utc)
    p = p or AsiaParams(risk_pct=settings.risk_per_trade)
    now_ny = pd.Timestamp(now).tz_convert(NY)
    state = _load_state(state_path)
    balance, _d, _lev = session.trader()
    equity = balance + session.unrealized_pnl()
    summary = {"timestamp": now.isoformat(timespec="seconds"), "kind": "asia", "label": label, "symbol": symbol, "equity": round(equity, 2),
               "dry_run": dry_run, "ny_time": now_ny.strftime("%Y-%m-%d %H:%M"), "orders": [], "closed": [], "skipped": [], "publish": False}
    own = [pos for pos in session.positions(only_bot=False) if pos.label == label]
    summary["open_position"] = {"symbol": own[0].symbol, "side": own[0].side, "units": own[0].units, "entry": own[0].price} if own else None
    weekday = now_ny.weekday()
    minutes = now_ny.hour * 60 + now_ny.minute

    # 1) cierre a las 12:00 NY de lo que siga abierto
    if own and minutes >= p.ny_end:
        for pos in own:
            if dry_run:
                summary["closed"].append({"symbol": pos.symbol, "units": pos.units, "reason": "cierre 12:00 NY (dry run)"})
                continue
            try:
                session.call("ProtoOAClosePositionReq", timeout=30, ctidTraderAccountId=session.account_id, positionId=pos.position_id,
                             volume=int(pos.units * VOLUME_SCALE))
                summary["closed"].append({"symbol": pos.symbol, "units": pos.units, "reason": "cierre 12:00 NY"})
                summary["publish"] = True
            except Exception as exc:  # noqa: BLE001
                summary["skipped"].append(f"{pos.symbol}: error al cerrar: {exc}")
        own = []

    guard = evaluate_guard(equity, settings.halt_mode, settings.max_drawdown_pct, settings.history_path("docs/aggr_state.json"), settings.state_dir)
    summary["guard"] = guard.as_dict()

    # 2) plan del dia
    if weekday >= 5 or minutes < p.ny_start or minutes >= p.ny_end:
        summary["plan"] = {"phase": "fuera de horario"}
        summary["skipped"].append("fuera de la manana de Nueva York")
    else:
        try:
            bars = m15_bars(session, symbol, now=now)
        except Exception as exc:  # noqa: BLE001
            summary["skipped"].append(f"sin barras: {exc}")
            bars = pd.DataFrame()
        if bars.empty:
            summary["plan"] = {"phase": "sin datos"}
        else:
            day, asia, nym = day_frames(bars, now, p)
            if len(asia) < 15:
                summary["plan"] = {"phase": "sin rango de Asia"}
                summary["skipped"].append(f"rango de Asia incompleto ({len(asia)} velas)")
            else:
                asia_hi, asia_lo = float(asia.high.max()), float(asia.low.min())
                plan = plan_day(asia_hi, asia_lo, nym, p)
                plan.update({"day": day.strftime("%Y-%m-%d"), "asia_hi": round(asia_hi, 2), "asia_lo": round(asia_lo, 2), "bars_seen": len(nym)})
                summary["plan"] = plan
                traded_today = state.get("last_trade_day") == plan["day"] or bool(own)
                if plan["phase"] == "ready" and not traded_today:
                    if guard.blocks_entries:
                        summary["skipped"].append(f"freno activo: {guard.reason}")
                    else:
                        info = session.symbols[symbol]
                        side = plan["side"]
                        entry_ref = float(nym.close.iloc[-1])
                        stop = plan["stop"]
                        dist = abs(entry_ref - stop)
                        units = size_units(equity, dist, info, p)
                        order = {"symbol": symbol, "side": "buy" if side == 1 else "sell", "units": units, "entry_ref": round(entry_ref, 2),
                                 "stop": round(stop, 2), "target": round(entry_ref + side * p.rr * dist, 2), "risk_usd": round(units * dist, 2),
                                 "asia_hi": round(asia_hi, 2), "asia_lo": round(asia_lo, 2), "retest_at": plan["retest_at"]}
                        if units <= 0 or (side == 1 and entry_ref <= stop) or (side == -1 and entry_ref >= stop):
                            summary["skipped"].append(f"{symbol}: sin tamano valido (distancia {dist:.2f}, riesgo maximo {p.max_risk_pct:.1%})")
                        elif dry_run:
                            order["status"] = "dry_run"; summary["orders"].append(order)
                        else:
                            tick = 10 ** max(5 - info.digits, 0)
                            rel_sl = max(int(round(dist * PRICE_SCALE / tick)) * tick, tick)
                            rel_tp = max(int(round(p.rr * dist * PRICE_SCALE / tick)) * tick, tick)
                            volume = round_volume(units, info.min_volume, info.step_volume, info.max_volume)
                            try:
                                res = session.call("ProtoOANewOrderReq", timeout=30, ctidTraderAccountId=session.account_id, symbolId=info.symbol_id,
                                                   orderType=session.model.ProtoOAOrderType.MARKET,
                                                   tradeSide=session.model.ProtoOATradeSide.BUY if side == 1 else session.model.ProtoOATradeSide.SELL,
                                                   volume=volume, relativeStopLoss=rel_sl, relativeTakeProfit=rel_tp, label=label)
                                order["status"] = session.model.ProtoOAExecutionType.Name(res.executionType).lower() if hasattr(res, "executionType") else "sent"
                                state["last_trade_day"] = plan["day"]
                                summary["publish"] = True
                            except Exception as exc:  # noqa: BLE001
                                order["status"] = f"error: {exc}"
                            summary["orders"].append(order)
                elif plan["phase"] == "ready" and traded_today:
                    summary["skipped"].append("ya se opero hoy")
    # 3) estado publicado
    prev_phase = (state.get("plan") or {}).get("phase")
    if summary.get("plan", {}).get("phase") != prev_phase:
        summary["publish"] = True
    try:
        deals = [d for d in session.deals(days=45) if d.get("closes") and d.get("label") == label]
        trades = [{"at": d["at"].strftime("%Y-%m-%dT%H:%MZ"), "symbol": d["symbol"], "units": d["units"], "entry": d["entry_price"], "exit": d["price"],
                   "net": d["net"], "opened_at": d["opened_at"].strftime("%Y-%m-%dT%H:%MZ") if d.get("opened_at") else None} for d in deals]
    except Exception as exc:  # noqa: BLE001
        trades = state.get("trades", [])
        summary["skipped"].append(f"sin historial de operaciones: {exc}")
    state.update({"at": now.strftime("%Y-%m-%dT%H:%MZ"), "ny_time": summary["ny_time"], "equity": summary["equity"], "plan": summary.get("plan"),
                  "open_position": summary["open_position"], "last_orders": summary["orders"] or state.get("last_orders", []),
                  "trades": trades, "params": {"rr": p.rr, "max_wait_bars": p.max_wait_bars, "risk_pct": p.risk_pct, "max_risk_pct": p.max_risk_pct},
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
