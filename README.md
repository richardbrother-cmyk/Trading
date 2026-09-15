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
| `autotrader/ctrader.py` | `CTraderBroker` sobre la Open API de Spotware (solo cuentas demo), con barras diarias del propio broker. |
| `autotrader/ctrader_auth.py` | Tokens OAuth de cTrader: intercambio, renovación y carga. |
| `autotrader/bot.py` | Un ciclo: datos → decisión → orden → registro en `state/run_log.jsonl`. |
| `autotrader/cli.py` | Comandos `backtest`, `run`, `status`. |
| `scripts/dashboard.py` | Genera el panel HTML (`reports/dashboard.html`) con backtests, cuenta y órdenes. |
| `.github/workflows/paper-trading.yml` | Ejecuta un ciclo cada hora en horario de mercado desde GitHub Actions (Alpaca). |
| `.github/workflows/ctrader-demo.yml` | Igual, contra la cuenta demo de cTrader. |

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

## Eventos de alto impacto

`data/events.json` contiene el calendario de eventos que mueven el mercado (decisiones de la Fed,
IPC, nóminas). Se edita a mano y el bot lo lee en cada ciclo. Alrededor de cada evento, desde
`EVENT_HOURS_BEFORE` (3 h) antes hasta `EVENT_HOURS_AFTER` (1 h) después:

- no se abren posiciones nuevas;
- `EVENT_MODE=trail` (por defecto): a las posiciones que ganan al menos `EVENT_MIN_GAIN` (1 %) se
  les sube el stop hasta `EVENT_TRAIL_PCT` (1,5 %) bajo el precio actual, sin bajarlo nunca;
- `EVENT_MODE=close`: esas posiciones se cierran antes del evento y se reevalúan en el primer
  ciclo tras la ventana;
- `EVENT_MODE=off`: sin protección.

El panel muestra los eventos de las próximas tres semanas y marca la ventana activa.

## Validación previa a la apertura

Las señales se calculan con el cierre diario y las órdenes se ejecutan en la apertura siguiente.
Entre medias el mercado se mueve, así que antes de abrir (`preopen`) el bot revisa cada compra en
cola con el precio de premercado de Yahoo:

- la cancela si el precio cae más de `MAX_GAP_DOWN` (1,5 %) respecto al último cierre, si sube más
  de `MAX_GAP_UP` (3 %), o si la señal recalculada con el precio proyectado deja de ser de compra;
- si se mantiene pero el precio se ha movido más de un 1 %, la recoloca con tamaño y stop
  recalculados sobre el precio en vivo.

La misma regla de hueco se aplica a las compras nuevas durante la sesión: el ciclo horario no abre
posición en un activo que ese día caiga más del 1,5 %.

```bash
python -m autotrader.cli preopen --dry-run   # solo informa
python -m autotrader.cli preopen             # cancela o recoloca
```

## Estrategia

- **Entrada**: SMA rápida > SMA lenta y RSI < 70.
- **Salida**: SMA rápida < SMA lenta, o precio ≤ precio de entrada × (1 − stop loss).
  En Alpaca el stop viaja con la compra como orden vinculada (clase OTO), así que el broker lo
  ejecuta aunque el bot no esté mirando; antes de vender por señal, el bot cancela ese stop.
- **Tamaño**: `equity × riesgo_por_operación / (precio × stop_loss)`, acotado al 12 % del
  equity por posición y al efectivo disponible. Máximo 10 posiciones. El bot nunca usa
  margen: dimensiona sobre efectivo y equity, no sobre el *buying power* del broker.
- **Cortafuegos**: si el equity cae un 3 % respecto al inicio del día, no se abren más
  posiciones ese día.

## Universo por defecto

- **Índices y tecnología**: SPY, QQQ, AAPL, MSFT, NVDA.
- **Metales con respaldo físico**: GLD (oro), SLV (plata), PPLT (platino), PALL (paladio).
- **Energía y agrícolas vía futuros**: USO (petróleo WTI), WEAT (trigo), CORN (maíz), DBA (cesta
  agrícola). Estos ETFs sufren desgaste por el traspaso de contratos (*contango*); la estrategia
  rota rápido y lo mitiga, pero no sirven para mantener meses. Café, algodón y cacao no se pueden
  operar en Alpaca: sus ETN fueron liquidados.

Alpaca no ofrece futuros ni contado de materias primas; la exposición se obtiene vía ETFs.

## Resultado de referencia (backtest 2024-09-12 → 2026-09-11)

| Universo | Retorno | Sharpe | Drawdown máx. | Operaciones | Profit factor |
|---|---|---|---|---|---|
| Solo acciones (5 pos., 25 %) | +3.5 % | 0.20 | −12.3 % | 45 | 1.13 |
| Solo metales físicos | +73.1 % | 1.12 | −24.9 % | 30 | 4.66 |
| Combinado (8 pos., 15 %) | +31.9 % | 0.93 | −13.6 % | 72 | 2.28 |
| Agrícolas y energía (5 pos., 25 %) | +32.0 % | 0.97 | −16.7 % | 64 | 2.74 |
| **Universo completo (10 pos., 12 %)** | **+51.4 %** | **1.45** | **−12.4 %** | 91 | 3.57 |

Dos años con un mercado alcista en metales favorecen mucho al segundo grupo; no hay que
leerlo como rendimiento esperado. La iteración (otros parámetros, otras señales, otros
universos) se hace con `backtest` antes de cambiar lo que opera en `run`.

## cTrader (Fusion Markets y otros brokers de CFDs)

El mismo bot puede operar una cuenta **demo** de cTrader a través de la Open API de Spotware.
Sirve para brokers como Fusion Markets, que ofrecen oro, plata, petróleo e índices como CFD
con apalancamiento alto. El bot rechaza cuentas reales (`CTRADER_DEMO` debe ser `true`).

Pasos:

1. Crea un cTrader ID en id.ctrader.com y vincula tu cuenta demo del broker.
2. Registra una app en openapi.ctrader.com. Queda en estado *Submitted* hasta que Spotware la
   activa (24 a 48 h). Copia el Client ID y el Client Secret al `.env`.
3. Autoriza y guarda los tokens (el código de autorización caduca en 60 segundos):
   ```bash
   python scripts/ctrader_token.py --url            # abre el enlace y autoriza la cuenta demo
   python scripts/ctrader_token.py "<URL con code=...>"
   ```
4. Verifica conexión, saldo y símbolos disponibles, y luego opera:
   ```bash
   python -m autotrader.cli --broker ctrader --provider ctrader --symbols XAUUSD,XAGUSD ctrader-check
   python -m autotrader.cli --broker ctrader --provider ctrader --symbols XAUUSD,XAGUSD run
   ```

Los servidores de trading de cTrader usan el puerto 5035, no HTTPS, así que hace falta una red
sin restricciones: tu ordenador, un VPS o los runners de GitHub Actions. El workflow
`ctrader-demo.yml` corre un ciclo cada hora con los secrets `CTRADER_CLIENT_ID`,
`CTRADER_CLIENT_SECRET`, `CTRADER_ACCOUNT_LOGIN` y `CTRADER_ACCESS_TOKEN`, y se activa con la
variable de repositorio `CTRADER_ENABLED=true`.

El access token dura 30 días. cTrader rota el refresh token en cada renovación e invalida el
anterior, así que el workflow no renueva nada: cada mes se ejecuta `scripts/ctrader_token.py
--refresh` donde esté guardado `state/ctrader_tokens.json` y se actualiza el secret
`CTRADER_ACCESS_TOKEN` con el nuevo valor. Los tokens nunca se suben al repositorio.

Configuración activa en la demo de Fusion Markets (fijada en el propio workflow): universo
XAUUSD, XAGUSD, XPTUSD, XTIUSD, XBRUSD, XNGUSD, COFARA (café), COTTON (algodón), WHEAT, CORN,
SUGAR, US500 y NAS100; stop del 3 %; hasta 8 posiciones del 12 % del equity con exposición
hasta 5x; ciclo cada hora de domingo noche a viernes. El interruptor `SCHEDULE_ENABLED` del
workflow para los ciclos programados; los lanzamientos manuales siempre corren.

Diferencias con acciones:

- Los tamaños son fraccionarios y se redondean al paso mínimo del símbolo (p.ej. 0.01 lotes).
- El stop loss se envía al servidor con la orden, además de la vigilancia del bot.
- `EXPOSURE_LEVERAGE` permite que la exposición nominal supere el equity (p.ej. 3 = hasta 3x),
  pero el riesgo por operación sigue siendo `RISK_PER_TRADE` del equity: el apalancamiento
  amplía el tamaño, no la pérdida máxima aceptada por operación.

## Panel visual

`scripts/dashboard.py` genera `reports/dashboard.html`, una página autocontenida con:

- las curvas de capital de los tres universos en backtest, con métricas comparadas;
- el estado real de la cuenta paper de Alpaca: capital, efectivo, posiciones y órdenes en cola;
- las decisiones del último ciclo del bot, símbolo a símbolo;
- el resultado por activo y la lista completa de operaciones del backtest activo.

```bash
python scripts/dashboard.py            # con cuenta (requiere claves en .env)
python scripts/dashboard.py --no-live  # solo backtests
```

Hay una copia estática en `docs/index.html`, publicada con GitHub Pages en
<https://richardbrother-cmyk.github.io/Trading/>. El workflow de Alpaca la regenera en cada ciclo y la adjunta como
artefacto de la ejecución.

## Investigación intradía

`autotrader/intraday.py` es un backtester sobre barras de 15 minutos con dos estrategias de sesión
(ruptura del rango de apertura y cruce EMA9/EMA21), costes de spread y comisión, y tamaños que
respetan el lote mínimo de cada CFD (útil para ver qué es operable con cuentas pequeñas).

- `scripts/fetch_intraday.py` descarga barras M15 de cTrader a `data/intraday/` (solo desde una red
  que alcance el puerto 5035, p.ej. el workflow `intraday-research.yml`).
- `scripts/intraday_backtest.py` corre las estrategias sobre esos CSV y deja el informe en
  `docs/intraday_report.json`.

Resultado de la primera pasada (mayo a septiembre de 2026, seis símbolos): ninguna de las dos
estrategias tiene expectativa positiva en conjunto; las pocas configuraciones ganadoras se
concentran en petróleo solo largo, en un periodo de fuerte subida del crudo, y no se sostienen
como regla general. Ver la conversación de investigación antes de operar nada de esto.

## Estructura de estado

```
state/
  sim_account.json   # efectivo y posiciones del broker simulado
  sim_orders.jsonl   # órdenes ejecutadas en el simulador
  run_log.jsonl      # un registro por ciclo con decisiones, órdenes y motivos
  day_start.json     # equity al inicio del día (límite de pérdida diaria)
```
