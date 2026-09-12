"""Configuracion del bot cargada desde variables de entorno / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

PAPER_URL = "https://paper-api.alpaca.markets"


class ConfigError(ValueError):
    """Configuracion invalida o insegura."""


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


@dataclass
class Settings:
    broker: str = "sim"
    data_provider: str = "yahoo"
    symbols: list[str] = field(default_factory=lambda: ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"])
    fast_sma: int = 20
    slow_sma: int = 50
    rsi_period: int = 14
    rsi_max_entry: float = 70.0
    initial_cash: float = 100_000.0
    risk_per_trade: float = 0.02
    max_positions: int = 5
    max_position_pct: float = 0.25
    max_daily_loss_pct: float = 0.03
    stop_loss_pct: float = 0.05
    state_dir: str = "state"
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = PAPER_URL

    @classmethod
    def from_env(cls, dotenv_path: str | None = None) -> "Settings":
        load_dotenv(dotenv_path, override=False)
        symbols = [s.strip().upper() for s in _env("SYMBOLS", "SPY,QQQ,AAPL,MSFT,NVDA").split(",") if s.strip()]
        settings = cls(
            broker=_env("BROKER", "sim").lower(),
            data_provider=_env("DATA_PROVIDER", "yahoo").lower(),
            symbols=symbols,
            fast_sma=int(_env("FAST_SMA", "20")),
            slow_sma=int(_env("SLOW_SMA", "50")),
            rsi_period=int(_env("RSI_PERIOD", "14")),
            rsi_max_entry=float(_env("RSI_MAX_ENTRY", "70")),
            initial_cash=float(_env("INITIAL_CASH", "100000")),
            risk_per_trade=float(_env("RISK_PER_TRADE", "0.02")),
            max_positions=int(_env("MAX_POSITIONS", "5")),
            max_position_pct=float(_env("MAX_POSITION_PCT", "0.25")),
            max_daily_loss_pct=float(_env("MAX_DAILY_LOSS_PCT", "0.03")),
            stop_loss_pct=float(_env("STOP_LOSS_PCT", "0.05")),
            state_dir=_env("STATE_DIR", "state"),
            alpaca_api_key=_env("ALPACA_API_KEY", ""),
            alpaca_secret_key=_env("ALPACA_SECRET_KEY", ""),
            alpaca_base_url=_env("ALPACA_BASE_URL", PAPER_URL).rstrip("/"),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.broker not in {"sim", "alpaca"}:
            raise ConfigError(f"BROKER debe ser 'sim' o 'alpaca', no {self.broker!r}")
        if self.data_provider not in {"yahoo", "alpaca", "synthetic"}:
            raise ConfigError(f"DATA_PROVIDER invalido: {self.data_provider!r}")
        if self.fast_sma <= 0 or self.slow_sma <= 0 or self.fast_sma >= self.slow_sma:
            raise ConfigError("Se requiere 0 < FAST_SMA < SLOW_SMA")
        if not self.symbols:
            raise ConfigError("SYMBOLS no puede estar vacio")
        if not 0 < self.risk_per_trade <= 0.1:
            raise ConfigError("RISK_PER_TRADE debe estar en (0, 0.1]")
        if not 0 < self.max_position_pct <= 1:
            raise ConfigError("MAX_POSITION_PCT debe estar en (0, 1]")
        if not 0 < self.stop_loss_pct < 1:
            raise ConfigError("STOP_LOSS_PCT debe estar en (0, 1)")
        if self.broker == "alpaca" or self.data_provider == "alpaca":
            if not (self.alpaca_api_key and self.alpaca_secret_key):
                raise ConfigError("Faltan ALPACA_API_KEY / ALPACA_SECRET_KEY")
        if self.broker == "alpaca" and self.alpaca_base_url != PAPER_URL:
            # Bloqueo duro: este proyecto solo opera en cuentas paper.
            raise ConfigError(
                f"ALPACA_BASE_URL debe ser {PAPER_URL}. Este bot solo opera en paper trading."
            )
