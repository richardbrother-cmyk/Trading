# autotrader — bot de trading automático (paper trading)

Proyecto experimental para que un bot opere una cuenta **simulada** de forma automática:
descarga datos reales, calcula una señal, gestiona el riesgo y envía órdenes a un broker.

> **Solo paper trading.** El bot rechaza arrancar contra la URL de dinero real de Alpaca.
> Es un experimento educativo, no un consejo de inversión. Los resultados del backtest no
> garantizan rendimientos futuros.

## Qué hay dentro

| Módulo | Función |
|---|---|
| `autotrader/data.py` | Barras diarias desde **Yahoo Finance** (sin claves), **Alpaca** (con claves) o **sintéticas** (sin red). |
| `autotrader/strategy.py` | Cruce de medias móviles (SMA 20/50) con filtro RSI. |
| `autotrader/risk.py` | Tamaño de posición por riesgo, máximo de posiciones, stop loss y límite de pérdida diaria. |
| `autotrader/backtest.py` | Backtester multi-símbolo con ejecución en la apertura siguiente, slippage y stops intradía. |
| `autotrader/broker.py` | `SimulatedBroker` (cuenta local persistida en `state/`) y `AlpacaBroker` (solo paper). |
| `autotrader/bot.py` | Un ciclo: datos → decisión → orden → registro en `state/run_log.jsonl`. |
| `autotrader/cli.py` | Comandos `backtest`, `run`, `status`. |
| `.github/workflows/paper-trading.yml` | Ejecuta un ciclo cada hora en horario de mercado desde GitHub Actions. |

## Instalación

```bash
pip install -r requirements.txt
cp .env.example .env      # opcional: por defecto usa broker simulado + datos de Yahoo
pytest -q
```

## Uso

```bash
# 1) Backtest de 2 años sobre el universo por defecto (datos reales de Yahoo)
python -m autotrader.cli backtest --report reports

# 2) Un ciclo de operación con el broker simulado (--force ignora si el mercado está cerrado)
python -m autotrader.cli run --force

# 3) Ver decisiones sin enviar órdenes
python -m autotrader.cli run --dry-run

# 4) Bucle continuo, un ciclo por hora
python -m autotrader.cli run --loop --interval 3600

# 5) Estado de la cuenta
python -m autotrader.cli status
```

Parámetros como símbolos, medias, riesgo o broker se cambian en `.env` o con
`--symbols`, `--provider`, `--broker`.

## Conectar a una plataforma real (Alpaca, cuenta paper)

1. Crea una cuenta gratuita en <https://alpaca.markets> y entra en el panel **Paper Trading**.
2. Genera un par de claves API de la cuenta paper.
3. En `.env`:
   ```
   BROKER=alpaca
   ALPACA_API_KEY=...
   ALPACA_SECRET_KEY=...
   ALPACA_BASE_URL=https://paper-api.alpaca.markets
   ```
4. `python -m autotrader.cli run` enviará órdenes de mercado reales a tu cuenta paper
   (solo si el mercado está abierto).

Para que opere sola sin tener un ordenador encendido, añade `ALPACA_API_KEY` y
`ALPACA_SECRET_KEY` como *secrets* del repositorio en GitHub. El workflow
`paper-trading.yml` correrá un ciclo cada hora de 14:35 a 20:35 UTC de lunes a viernes
y subirá el registro como artefacto.

## Estrategia

- **Entrada**: SMA rápida > SMA lenta y RSI < 70.
- **Salida**: SMA rápida < SMA lenta, o precio ≤ precio de entrada × (1 − stop loss).
- **Tamaño**: `equity × riesgo_por_operación / (precio × stop_loss)`, acotado al 15 % del
  equity por posición y al efectivo disponible. Máximo 8 posiciones. El bot nunca usa
  margen: dimensiona sobre efectivo y equity, no sobre el *buying power* del broker.
- **Cortafuegos**: si el equity cae un 3 % respecto al inicio del día, no se abren más
  posiciones ese día.

## Universo por defecto

- **Índices y tecnología**: SPY, QQQ, AAPL, MSFT, NVDA.
- **Metales con respaldo físico**: GLD (oro), SLV (plata), PPLT (platino), PALL (paladio).
  Se eligen ETFs físicos y no de futuros (USO, UNG, DBC…) porque estos últimos sufren
  *contango* y se desvían del precio contado con el tiempo.

Alpaca no ofrece futuros ni contado de materias primas; la exposición se obtiene vía ETFs.

## Resultado de referencia (backtest 2024-09-12 → 2026-09-11)

| Universo | Retorno | Sharpe | Drawdown máx. | Operaciones | Profit factor |
|---|---|---|---|---|---|
| Solo acciones (5 pos., 25 %) | +3.5 % | 0.20 | −12.3 % | 45 | 1.13 |
| Solo metales físicos | +73.1 % | 1.12 | −24.9 % | 30 | 4.66 |
| **Combinado (8 pos., 15 %)** | **+31.9 %** | **0.93** | **−13.6 %** | 72 | 2.28 |

Dos años con un mercado alcista en metales favorecen mucho al segundo grupo; no hay que
leerlo como rendimiento esperado. La iteración (otros parámetros, otras señales, otros
universos) se hace con `backtest` antes de cambiar lo que opera en `run`.

## Estructura de estado

```
state/
  sim_account.json   # efectivo y posiciones del broker simulado
  sim_orders.jsonl   # órdenes ejecutadas en el simulador
  run_log.jsonl      # un registro por ciclo con decisiones, órdenes y motivos
  day_start.json     # equity al inicio del día (límite de pérdida diaria)
```
