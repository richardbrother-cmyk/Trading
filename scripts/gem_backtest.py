"""Backtest de Dual Momentum (GEM, Antonacci) con datos mensuales de Yahoo.

Reglas (Antonacci, "Global Equities Momentum"):
  Al cierre de cada mes se mira el retorno total de 12 meses de la renta variable USA, la
  internacional ex-USA y las letras del Tesoro.
  - Momentum absoluto: si USA 12m <= letras 12m -> bonos agregados.
  - Momentum relativo: si no, el mejor de USA / internacional a 12 meses.
Se compara con comprar y mantener, 60/40 y un filtro de SMA de 10 meses sobre USA.

Uso:
  python scripts/gem_backtest.py            # fondos indexados (historial desde 1997)
  python scripts/gem_backtest.py --etf      # ETFs operables en Alpaca (desde 2008)
  python scripts/gem_backtest.py --out docs/gem_backtest.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CACHE_DIR = os.path.join(ROOT, "data", "research")

from autotrader.data import yahoo_monthly_adjclose  # noqa: E402
from autotrader.gem import tbill_index  # noqa: E402

FUNDS = {"us": "VFINX", "intl": "VGTSX", "bond": "VBMFX"}
ETFS = {"us": "SPY", "intl": "VEU", "bond": "AGG"}
TBILL = "^IRX"  # rendimiento anualizado de la letra a 13 semanas, en %


def yahoo_adjclose(symbol: str, session: requests.Session) -> pd.Series:
    """Cierres mensuales ajustados (meses completos), con cache de un dia en data/research/."""
    cache = os.path.join(CACHE_DIR, f"yahoo_adj_{symbol.replace('^', '_')}.csv")
    if os.path.exists(cache) and time.time() - os.path.getmtime(cache) < 86_400:
        return pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0].astype(float)
    s = yahoo_monthly_adjclose(symbol, session=session)
    os.makedirs(CACHE_DIR, exist_ok=True)
    s.to_csv(cache)
    return s


def monthly(series: pd.Series) -> pd.Series:
    return series.resample("ME").last().dropna()


def stats(equity: pd.Series, rf: pd.Series | None = None) -> dict:
    rets = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    dd = equity / equity.cummax() - 1
    excess = rets - (rf.reindex(rets.index).fillna(0.0) if rf is not None else 0.0)
    sharpe = float(excess.mean() / excess.std() * np.sqrt(12)) if excess.std() > 0 else 0.0
    worst_year = rets.groupby(rets.index.year).apply(lambda r: (1 + r).prod() - 1).min()
    return {
        "cagr": round(float(cagr) * 100, 2),
        "max_dd": round(float(dd.min()) * 100, 2),
        "sharpe": round(sharpe, 2),
        "vol": round(float(rets.std() * np.sqrt(12)) * 100, 2),
        "worst_year": round(float(worst_year) * 100, 2),
        "final_multiple": round(float(equity.iloc[-1] / equity.iloc[0]), 2),
    }


def run(symbols: dict[str, str], lookback: int = 12, sma_months: int = 10) -> dict:
    sess = requests.Session()
    px = {k: monthly(yahoo_adjclose(v, sess)) for k, v in symbols.items()}
    tb = tbill_index(monthly(yahoo_adjclose(TBILL, sess)))
    df = pd.concat({**px, "tbill": tb}, axis=1, sort=True).dropna()
    rets = df.pct_change()
    mom = df / df.shift(lookback) - 1  # retorno de 12 meses al cierre de cada mes

    # Señal al cierre del mes t, aplicada al retorno del mes t+1.
    choice = pd.Series(index=df.index, dtype=object)
    for t in df.index:
        if pd.isna(mom.loc[t, "us"]):
            continue
        if mom.loc[t, "us"] <= mom.loc[t, "tbill"]:
            choice[t] = "bond"
        else:
            choice[t] = "us" if mom.loc[t, "us"] >= mom.loc[t, "intl"] else "intl"
    choice = choice.dropna()
    gem_ret = pd.Series([rets.loc[t1, choice[t0]] for t0, t1 in zip(choice.index[:-1], choice.index[1:])], index=choice.index[1:])
    rf = rets["tbill"]

    start = gem_ret.index[0]
    base = rets.loc[start:]
    curves = {
        "gem": (1 + gem_ret).cumprod(),
        "us_buy_hold": (1 + base["us"]).cumprod(),
        "intl_buy_hold": (1 + base["intl"]).cumprod(),
        "60_40": (1 + 0.6 * base["us"] + 0.4 * base["bond"]).cumprod(),
    }
    # Filtro SMA de 10 meses sobre USA: dentro si cierre > SMA, si no bonos.
    sma = df["us"].rolling(sma_months).mean()
    in_mkt = (df["us"] > sma).shift(1).reindex(base.index).fillna(False)
    sma_ret = np.where(in_mkt, base["us"], base["bond"])
    curves["sma10_us"] = (1 + pd.Series(sma_ret, index=base.index)).cumprod()

    switches = int((choice != choice.shift(1)).sum() - 1)
    out = {
        "symbols": symbols,
        "period": [str(start.date()), str(df.index[-1].date())],
        "months": int(len(gem_ret)),
        "switches": switches,
        "stats": {k: stats(v, rf) for k, v in curves.items()},
        "allocation_months": choice.loc[start:].value_counts().to_dict(),
        "current_signal": {
            "date": str(choice.index[-1].date()),
            "choice": choice.iloc[-1],
            "mom_us": round(float(mom["us"].iloc[-1]) * 100, 2),
            "mom_intl": round(float(mom["intl"].iloc[-1]) * 100, 2),
            "mom_tbill": round(float(mom["tbill"].iloc[-1]) * 100, 2),
        },
    }
    # Sub-periodos: por decada y ultimos 10 anos.
    sub = {}
    for label, (a, b) in {
        "hasta_2009": (None, "2009-12-31"),
        "2010_2019": ("2010-01-01", "2019-12-31"),
        "2020_hoy": ("2020-01-01", None),
        "ultimos_10a": (str((df.index[-1] - pd.DateOffset(years=10)).date()), None),
    }.items():
        seg = {k: v.loc[a:b] for k, v in curves.items()}
        seg = {k: v / v.iloc[0] for k, v in seg.items() if len(v) > 12}
        if seg:
            sub[label] = {k: stats(v, rf) for k, v in seg.items()}
    out["subperiods"] = sub
    out["yearly_gem_vs_us"] = {
        str(y): [round(float(g) * 100, 1), round(float(u) * 100, 1)]
        for (y, g), u in zip(
            gem_ret.groupby(gem_ret.index.year).apply(lambda r: (1 + r).prod() - 1).items(),
            base["us"].groupby(base.index.year).apply(lambda r: (1 + r).prod() - 1),
        )
    }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--etf", action="store_true", help="usar SPY/VEU/AGG en vez de fondos indexados")
    ap.add_argument("--lookback", type=int, default=12)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    res = run(ETFS if args.etf else FUNDS, lookback=args.lookback)
    txt = json.dumps(res, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
