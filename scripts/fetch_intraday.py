"""Descarga barras M15 de cTrader (solo desde una red que alcance el puerto 5035) a data/intraday/<SYM>_M15.csv.

Uso: python scripts/fetch_intraday.py --symbols US500,NAS100 --days 120
Acumula con lo ya descargado (dedup por marca de tiempo).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autotrader.config import Settings  # noqa: E402
from autotrader.ctrader import VOLUME_SCALE, CTraderSession, decode_trendbars  # noqa: E402
from autotrader.ctrader_auth import load_access_token  # noqa: E402

PERIOD_M15 = 7
CHUNK_DAYS = 7


def fetch_m15(session: CTraderSession, symbol: str, days: int) -> pd.DataFrame:
    info = session.symbols[symbol]
    end = datetime.now(timezone.utc)
    frames = []
    start_days = days
    while start_days > 0:
        frm = end - timedelta(days=start_days)
        to = end - timedelta(days=max(start_days - CHUNK_DAYS, 0))
        res = session.call("ProtoOAGetTrendbarsReq", timeout=40, ctidTraderAccountId=session.account_id,
                           fromTimestamp=int(frm.timestamp() * 1000), toTimestamp=int(to.timestamp() * 1000),
                           period=PERIOD_M15, symbolId=info.symbol_id)
        frames.append(decode_trendbars(list(res.trendbar), period_minutes=15))
        start_days -= CHUNK_DAYS
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index = df.index.tz_localize("UTC")
    df.index.name = "time"
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="US500,NAS100,XAUUSD,XTIUSD,EURUSD,GBPUSD")
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--out", default="data/intraday")
    args = ap.parse_args()
    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    s = Settings.from_env("/dev/null")
    s.broker = "ctrader"
    s.symbols = symbols
    s.validate()
    token = load_access_token(os.path.join(s.state_dir, "ctrader_tokens.json"), s.ctrader_client_id, s.ctrader_client_secret,
                              s.ctrader_access_token, s.ctrader_refresh_token)
    session = CTraderSession(s.ctrader_client_id, s.ctrader_client_secret, token, s.ctrader_account_login or None, demo=s.ctrader_demo)
    os.makedirs(args.out, exist_ok=True)
    try:
        available = {n.upper() for n in session.all_symbol_names()}
        missing = [x for x in symbols if x not in available]
        if missing:
            hints = {m: sorted(n for n in available if m[:3] in n)[:12] for m in missing}
            print(f"AVISO: simbolos no disponibles {missing}; parecidos: {hints}")
            symbols = [x for x in symbols if x in available]
        infos = session.load_symbols(symbols)
        spec_path = os.path.join(args.out, "symbols.json")
        known = {}
        if os.path.exists(spec_path):
            with open(spec_path, encoding="utf-8") as fh:
                known = json.load(fh)
        for name, info in infos.items():
            known[name] = {"symbol_id": info.symbol_id, "digits": info.digits, "lot_size": info.lot_size / VOLUME_SCALE,
                           "min_units": info.min_volume / VOLUME_SCALE, "step_units": info.step_volume / VOLUME_SCALE}
        with open(spec_path, "w", encoding="utf-8") as fh:
            json.dump(known, fh, indent=1, sort_keys=True)
        print("especificaciones:", {k: known[k] for k in symbols})
        for sym in symbols:
            df = fetch_m15(session, sym, args.days)
            path = os.path.join(args.out, f"{sym}_M15.csv")
            if os.path.exists(path):
                old = pd.read_csv(path, parse_dates=["time"]).set_index("time")
                if old.index.tz is None:
                    old.index = old.index.tz_localize("UTC")
                df = pd.concat([old, df])
                df = df[~df.index.duplicated(keep="last")].sort_index()
            df.to_csv(path, float_format="%.5f")
            print(f"{sym}: {len(df)} barras M15, {df.index[0]} -> {df.index[-1]}")
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
