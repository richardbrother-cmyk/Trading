"""Interfaz de linea de comandos: backtest, run, status."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .backtest import run_backtest
from .broker import build_broker
from .config import ConfigError, Settings
from .data import DataProvider
from .risk import RiskParams
from .strategy import StrategyParams


def _settings(args) -> Settings:
    try:
        s = Settings.from_env(args.env)
    except ConfigError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        sys.exit(2)
    if getattr(args, "symbols", None):
        s.symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    if getattr(args, "provider", None):
        s.data_provider = args.provider
    if getattr(args, "broker", None):
        s.broker = args.broker
    s.validate()
    return s


def _provider(s: Settings, broker=None, days: int = 500) -> DataProvider:
    bars_fn = getattr(broker, "bars", None) if s.data_provider == "ctrader" else None
    return DataProvider(s.data_provider, s.alpaca_api_key, s.alpaca_secret_key, days=days, bars_fn=bars_fn)


def cmd_backtest(args) -> int:
    s = _settings(args)
    broker = build_broker(s) if s.data_provider == "ctrader" else None
    provider = _provider(s, broker, days=args.days)
    data = provider.bars_many(s.symbols)
    if not data:
        print("Sin datos; abortando", file=sys.stderr)
        return 1
    result = run_backtest(
        data,
        StrategyParams(s.fast_sma, s.slow_sma, s.rsi_period, s.rsi_max_entry),
        RiskParams(s.risk_per_trade, s.max_positions, s.max_position_pct, s.max_daily_loss_pct, s.stop_loss_pct, s.exposure_leverage),
        initial_cash=s.initial_cash,
    )
    m = result.metrics()
    print(f"Backtest {', '.join(data)} | {result.equity.index[0].date()} -> {result.equity.index[-1].date()}")
    for k, v in m.items():
        print(f"  {k:>18}: {v}")
    if args.report:
        os.makedirs(args.report, exist_ok=True)
        result.equity.to_csv(os.path.join(args.report, "equity.csv"), header=["equity"])
        with open(os.path.join(args.report, "trades.csv"), "w", encoding="utf-8") as fh:
            fh.write("symbol,entry_date,entry_price,qty,exit_date,exit_price,pnl,ret,reason\n")
            for t in result.trades:
                fh.write(f"{t.symbol},{t.entry_date.date()},{t.entry_price:.4f},{t.qty},{t.exit_date.date() if t.exit_date is not None else ''},"
                         f"{t.exit_price if t.exit_price is not None else ''},{t.pnl:.2f},{t.ret:.4f},{t.reason}\n")
        with open(os.path.join(args.report, "metrics.json"), "w", encoding="utf-8") as fh:
            json.dump(m, fh, indent=2)
        print(f"Informe guardado en {args.report}/")
    return 0


def cmd_run(args) -> int:
    from .bot import run_cycle

    s = _settings(args)
    broker = build_broker(s)
    provider = _provider(s, broker)
    print(f"Broker: {broker.name} | datos: {s.data_provider} | simbolos: {','.join(s.symbols)} | dry_run={args.dry_run}")
    while True:
        summary = run_cycle(s, broker, provider, dry_run=args.dry_run, force=args.force)
        print(json.dumps(summary, indent=2, default=str))
        if not args.loop:
            return 0
        print(f"[run] esperando {args.interval} s...")
        time.sleep(args.interval)


def cmd_preopen(args) -> int:
    from .preopen import preopen_check

    s = _settings(args)
    broker = build_broker(s)
    provider = _provider(s, broker)
    summary = preopen_check(s, broker, provider, dry_run=args.dry_run)
    print(json.dumps(summary, indent=2, default=str, ensure_ascii=False))
    return 0


def cmd_status(args) -> int:
    s = _settings(args)
    broker = build_broker(s)
    acct = broker.account()
    print(f"Broker: {broker.name}")
    print(f"  equity: {acct.equity:,.2f}   cash: {acct.cash:,.2f}")
    for sym, p in acct.positions.items():
        print(f"  {sym:>6}: {p.qty} @ {p.avg_price:.2f}")
    log = os.path.join(s.state_dir, "run_log.jsonl")
    if os.path.exists(log):
        with open(log, encoding="utf-8") as fh:
            lines = fh.readlines()
        print(f"  ciclos registrados: {len(lines)} (ultimo: {json.loads(lines[-1])['timestamp']})")
    return 0


def cmd_ctrader_check(args) -> int:
    """Verifica autorizacion, cuenta y simbolos de cTrader sin operar."""
    s = _settings(args)
    s.broker = "ctrader"
    s.validate()
    from .ctrader import CTraderSession
    from .ctrader_auth import load_access_token

    token = load_access_token(os.path.join(s.state_dir, "ctrader_tokens.json"), s.ctrader_client_id,
                              s.ctrader_client_secret, s.ctrader_access_token, s.ctrader_refresh_token)
    session = CTraderSession(s.ctrader_client_id, s.ctrader_client_secret, token, s.ctrader_account_login or None, demo=s.ctrader_demo)
    balance, digits, leverage = session.trader()
    print(f"Cuenta {s.ctrader_account_login} ({'demo' if s.ctrader_demo else 'live'}) id={session.account_id}")
    print(f"  balance: {balance:,.2f}   apalancamiento: 1:{leverage:.0f}   PnL abierto: {session.unrealized_pnl():,.2f}")
    names = session.all_symbol_names()
    print(f"  simbolos disponibles: {len(names)}")
    for key in ["XAU", "XAG", "XPT", "XPD", "XTI", "XBR", "BRENT", "WTI", "OIL", "NGAS", "US500", "SPX", "NAS", "US30", "COPPER"]:
        hits = [n for n in names if key in n.upper()]
        if hits:
            print(f"    {key:>6}: {', '.join(hits[:8])}")
    try:
        info = session.load_symbols(s.symbols)
        for name, i in info.items():
            print(f"  {name:>8}: id={i.symbol_id} digits={i.digits} lote={i.lot_size / 100:g} min={i.min_volume / 100:g} paso={i.step_volume / 100:g}")
    except ValueError as exc:
        print(f"  [aviso] {exc}")
    session.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="autotrader", description="Bot de trading automatico (paper trading)")
    p.add_argument("--env", default=".env", help="ruta al archivo .env")
    p.add_argument("--symbols", help="lista separada por comas, sobreescribe SYMBOLS")
    p.add_argument("--provider", choices=["yahoo", "alpaca", "synthetic", "ctrader"], help="sobreescribe DATA_PROVIDER")
    p.add_argument("--broker", choices=["sim", "alpaca", "ctrader"], help="sobreescribe BROKER")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("backtest", help="simula la estrategia sobre historico")
    b.add_argument("--days", type=int, default=500)
    b.add_argument("--report", help="directorio para guardar equity/trades/metrics")
    b.set_defaults(func=cmd_backtest)

    r = sub.add_parser("run", help="ejecuta un ciclo (o bucle) de operacion")
    r.add_argument("--loop", action="store_true", help="repetir indefinidamente")
    r.add_argument("--interval", type=int, default=3600, help="segundos entre ciclos en modo --loop")
    r.add_argument("--dry-run", action="store_true", help="decidir sin enviar ordenes")
    r.add_argument("--force", action="store_true", help="operar aunque el mercado este cerrado (en Alpaca las ordenes quedan en cola hasta la apertura)")
    r.set_defaults(func=cmd_run)

    po = sub.add_parser("preopen", help="valida las ordenes en cola con el precio de premercado")
    po.add_argument("--dry-run", action="store_true", help="solo informar, sin cancelar ni recolocar")
    po.set_defaults(func=cmd_preopen)

    st = sub.add_parser("status", help="muestra cuenta y posiciones")
    st.set_defaults(func=cmd_status)

    cc = sub.add_parser("ctrader-check", help="verifica autorizacion, cuenta y simbolos de cTrader")
    cc.set_defaults(func=cmd_ctrader_check)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
