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

`requirements.txt` fija versiones exactas (`==`): los workflows instalan las dependencias en cada ciclo y el código
envía órdenes a brokers, así que cada ejecución debe usar exactamente lo mismo que se probó. Para actualizar una
librería: cambiar el pin, correr `pytest` y lanzar un ciclo con `dry_run` antes de dejarlo en producción.

Los cuatro workflows programados (Alpaca, cTrader tendencia, swing y agresiva) llevan un grupo `concurrency` por
workflow: si un ciclo se retrasa y se solapa con el siguiente, el nuevo espera en cola y nunca se cancela el que
puede estar enviando órdenes, de modo que no hay dos ciclos del mismo bot operando a la vez.

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

Resultado (septiembre 2025 a septiembre 2026, seis símbolos, 144 configuraciones de ruptura del
rango de apertura): ninguna alcanza un profit factor de 1,2 con al menos 60 operaciones. La
configuración que destacaba a cuatro meses (petróleo solo largo) pierde en tres de los cuatro
trimestres del año. Conclusión: con estas estrategias simples no hay ventaja intradía
demostrable; no operar nada de esto con dinero real.

## Investigación swing (1 h y 4 h, posiciones de 1 a 3 días)

`autotrader/swing.py` reagrupa las barras M15 a 1 h y 4 h y prueba tres familias, largos y cortos:
retroceso en tendencia (EMA50/200 + RSI), ruptura de máximos de N barras y reversión en bandas de
Bollinger; stops por ATR, comisión, spread, swap diario y cierre forzoso a los 3 días.
`scripts/swing_backtest.py` corre el barrido y deja `docs/swing_report.json`.

Resultado (sep 2025 – sep 2026): retroceso y ruptura pierden en ambos marcos. Solo la reversión en
bandas de 4 h, solo largos, tiene expectativa positiva (68 operaciones, profit factor 1,8, unos
2,4 días por operación), pero se debilita a lo largo del año y el último trimestre es negativo.
Candidata a prueba en demo, no a dinero real.

Esa prueba es `autotrader/swingbot.py` (comando `swing-run`) y el workflow `ctrader-swing.yml`:
corre cada hora sobre una segunda cuenta demo de cTrader fondeada con 200 USD y apalancamiento
1:500. GitHub Actions retrasa u omite los cron con frecuencia, así que el workflow se lanza a menudo
y el bot solo entra si la última barra de 4 h cerrada lo hizo hace menos de 2 h
(`SWING_MAX_SIGNAL_AGE_HOURS`); una entrada más tardía ya no es la que probó el backtest. Arriesga
el 5 % del equity real de la cuenta por operación (`RISK_PER_TRADE`, decisión del usuario para
simular una operativa agresiva; el tope `SWING_MAX_RISK_PCT` de 7.5 % decide si se acepta un lote
mínimo que arriesgue más). No hay techo de equity, así que el tamaño crece o se reduce con la cuenta
(`EQUITY_CAP` permite fijar uno). Compra con stop a 2 ATR y objetivo en la media de las bandas enviados
con la orden, cierra a los 3 días, y etiqueta sus posiciones para no mezclarse con el bot
tendencial ni con operaciones manuales. Cada ciclo publica `docs/swing_state.json` y el panel
muestra la cuenta en la pestaña "Fusion · swing 200 USD". Oro y petróleo se descartan solos porque su lote mínimo
arriesga más del tope mientras la cuenta sea pequeña.

### Simulación de la cuenta agresiva a 3 años

`scripts/aggr_simulation.py` (resultado en `docs/aggr_simulation.json`, gráfica en la pestaña agresiva) recorre
los 3 años de barras M15 de `data/intraday/` (septiembre 2023 a septiembre 2026, descargados con el workflow
`intraday-research` y `days=1100`) con los parámetros que operan hoy: ruptura H4 de 20 barras, stop 0,75 ATR,
objetivo 6R, break even tras 2 R, 7 días, 6 % de riesgo, lote mínimo, máximo 3 posiciones y freno al 30 %.
Primero genera las 693 señales de la estrategia (acierto 29 %, 0,21 R de media, PF 1,31; 2025 fue un año plano
con −1 R en 267 señales) y después simula una cuenta de 500 USD que las recorre en orden, más 2.000 recorridos
Monte Carlo que conservan fechas y distancias al stop pero barajan los resultados.

| Escenario | Histórico (3 años) | Monte Carlo p10 / mediana / p90 | P(freno) | Caída máx. mediana |
|---|---|---|---|---|
| Con freno 30 %, riesgo 6 % (config. actual) | 462 USD (−8 %), freno a los 2 meses | 335 / 457 / 1.192 | 100 % | −33 % |
| Sin freno, riesgo 6 % | 178 USD (−64 %), caída −99 % | 78 / 5.967 / 367.845 | — | −92 % |
| Con freno, riesgo 2 % | 650 USD (+30 %) | 367 / 658 / — | 99 % | −31 % |
| Sin freno, riesgo 2 % | 1.522 USD (+204 %), caída −65 % | 726 / 3.217 / — | — | −49 % |

Lectura: con 6 % de riesgo y un acierto de un tercio, cinco o seis pérdidas seguidas (habituales) ya son un 30 %
de caída, así que el freno detiene la cuenta en todos los recorridos, de mediana en el segundo mes, y desde ahí
no opera hasta que se rearme. Sin freno la mediana es alta pero la dispersión es enorme y el recorrido histórico
real termina en pérdida tras una caída del 99 %. Con 2 % de riesgo la misma estrategia sobrevive al año plano.

### Variantes probadas de "151 Trading Strategies" (Alpaca)

`scripts/alpaca_variants.py` (resultados en `docs/alpaca_variants.json`, datos de 2 años cacheados en
`data/research/alpaca_daily_2y.csv`) prueba tres ideas del catálogo de Kakushadze y Serur (2018,
`docs/ssrn-3247865.pdf`) sobre el bot de tendencia con el motor de backtest y ganchos opcionales de
`run_backtest` (`entry_allowed`, `size_fn`, `drop_exit_pct`, `reentry_cooldown_days`):

| Variante | Retorno | Sharpe | Caída máx. | Ops. | PF |
|---|---|---|---|---|---|
| **Configuración actual** | 49,1 % | 1,43 | −12,0 % | 90 | 3,35 |
| Régimen: solo entra si SPY > SMA 200 | 40,0 % | 1,04 | −18,8 % | 66 | 3,29 |
| Régimen SPY > SMA 100 | 47,1 % | 1,36 | −12,8 % | 78 | 3,49 |
| Tamaño por volatilidad (0,12 % diario por posición) | 39,9 % | 1,27 | −13,4 % | 95 | 3,34 |
| Salida por caída diaria > 3 %, recompra al día siguiente | 55,8 % | 1,68 | −13,2 % | 217 | 1,94 |
| Salida por caída > 3 % con ganancia, sin reentrar 10 días | 42,4 % | 1,62 | −9,4 % | 134 | 2,13 |
| Comprar y mantener SPY | 33,6 % | 0,97 | −19,0 % | — | — |

Conclusiones: el filtro de régimen con SPY empeora porque el universo es multiactivo y bloquea oro, plata y
agrícolas justo cuando diversifican; el tamaño por volatilidad recorta a los grandes ganadores (NVDA) sin reducir la
caída; la salida por caída diaria solo mejora si se recompra al día siguiente, cosa que el bot horario no
reproduce (vendería y recompraría en la hora siguiente), y con una espera realista de 5 a 10 días baja la caída
máxima a cambio de menos retorno. Ninguna variante se activó.

### Cuenta agresiva (500 USD)

Tercera cuenta demo, workflow `ctrader-aggr.yml`, mismo motor que el bot swing con otro perfil
(`SWING_STRATEGY`, `SWING_STOP_ATR`, `SWING_TP_ATR`, `SWING_PURE_RR`, `SWING_MAX_HOLD_DAYS`,
`SWING_LABEL`, `SWING_MAX_POSITIONS`): ruptura de 4 h (cierre sobre el máximo de 20 barras y sobre la
EMA200), solo largos, stop a 0,75 ATR y objetivo fijo a 6 veces el stop enviados con la orden, salida a
los 7 días, máximo 3 posiciones abiertas, etiqueta `autotrader-aggr`. El riesgo por operación se fija
con la variable `AGGR_RISK_PER_TRADE` (3 % por defecto) y el freno por drawdown con
`AGGR_MAX_DRAWDOWN_PCT` (30 %). Estado en `docs/aggr_state.json` y pestaña propia en el panel.

**Stop a break even** (`SWING_BREAKEVEN_R`, variable `AGGR_BREAKEVEN_R`, 2 R por defecto; `SWING_BREAKEVEN_LOCK_R`,
variable `AGGR_BREAKEVEN_LOCK_R`, 0,1 R): en cada ciclo el bot pide el último precio de 1 minuto de cada posición
propia y, si ya lleva ganados N R (R = distancia entre la entrada y el stop original), sube el stop a la entrada más
0,1 R para cubrir spread y comisión, reenviando el take profit (la API de cTrader borra el objetivo si no se reenvía
al cambiar el stop). Solo actúa una vez por posición y nunca toca posiciones manuales. El mismo comportamiento está
en el backtest (`breakeven_r`, `breakeven_lock_r` en `SwingParams`; la salida se etiqueta "break even").

Coste de la regla en el histórico (`docs/aggr_breakeven.json`, 12 meses, seis símbolos, R por operación):

| Regla | Ops. | Llega al 6R | Sale en BE | R medio | R total | PF | Racha sin ganar | Caída máx. al 6 % |
|---|---|---|---|---|---|---|---|---|
| Sin break even | 170 | 19 % | 0 % | 0,47 | 79 | 1,58 | 16 | −71 % |
| BE tras 0,5 R | 212 | 9 % | 51 % | 0,22 | 47 | 1,56 | 7 | −55 % |
| BE tras 1 R | 193 | 11 % | 37 % | 0,29 | 55 | 1,57 | 7 | −60 % |
| BE tras 1,5 R | 184 | 12 % | 31 % | 0,26 | 48 | 1,46 | 9 | −68 % |
| **BE tras 2 R (activa)** | 177 | 15 % | 21 % | 0,34 | 60 | 1,53 | 9 | −73 % |
| BE tras 3 R | 173 | 16 % | 13 % | 0,36 | 62 | 1,50 | 18 | −74 % |

Con un objetivo de 6R el precio vuelve a la entrada muy a menudo antes de llegar: la regla recorta la ganancia media
por operación (de 0,47 R a 0,29 R con 1 R, a 0,34 R con 2 R) porque parte de las operaciones que habrían llegado al
objetivo salen en cero, pero acorta las rachas de pérdidas (de 16 a 7 o 9). Es una decisión de preferencia, no de
borde: protege capital a cambio de menos beneficio esperado. Se eligió 2 R como punto intermedio.

Por qué 3 % y no 10 %: con la única combinación que mostró borde en 12 meses (factor de beneficio 1,73,
148 operaciones, solo el 20 % llega al objetivo, rachas de 14 pérdidas), una simulación de una cuenta
de 500 USD con máximo 3 posiciones y 3.000 remuestreos del orden de las operaciones da: al 2 % de
riesgo, mediana final 2.146 USD y 2 % de probabilidad de caer a la mitad; al 3 %, 3.805 USD y 18 %;
al 5 %, 9.048 USD y 72 %; al 10 %, 20.615 USD de mediana pero 100 % de probabilidad de pasar por una
caída del 50 % y un peor decil de 470 USD. Es la misma estrategia; solo cambia si sobrevive.

### Walk-forward de la estrategia swing

`scripts/swing_walkforward.py` reparte los 12 meses de barras en ventanas rodantes (4 meses para elegir
parámetros, 2 para probarlos, avanzando de 2 en 2) y encadena los tramos de prueba como resultado
fuera de muestra. Deja `docs/swing_walkforward.json` y el panel lo muestra en la pestaña swing.
Resultado con los datos hasta el 15 de septiembre de 2026: reoptimizar los parámetros en cada tramo
pierde fuera de muestra (factor de beneficio 0,92 en 44 operaciones) y la combinación elegida cambia
de un tramo a otro; la configuración desplegada (stop 2 ATR, bandas 20/2, RSI < 30, 3 días) mantiene
un factor de beneficio de 1,34 en esos mismos tramos, pero fue elegida mirando todo el año, así que no
cuenta como prueba independiente. Lectura honesta: el borde es débil y sensible a los parámetros; la
demo de 8 semanas es la prueba que falta.

### Investigación: Kronos como filtro de confirmación

`scripts/kronos_filter.py` prueba el modelo fundacional de velas
[Kronos](https://github.com/shiyu-coder/Kronos) (Kronos-small, 24,7M de parámetros, en CPU) como
filtro de las señales del swing: en cada señal de compra pide la predicción de las siguientes barras
de 4 h con 400 barras de contexto y solo entra si la subida prevista supera un umbral. Las
predicciones se guardan en `data/research/kronos_predictions.json` y los informes en
`docs/kronos_report.json` (6 barras) y `docs/kronos_report_h3.json` (3 barras). Resultados sobre los
12 meses y 126 señales con predicción:

- El signo de la predicción no sirve: Kronos prevé subida en el 100 % de las señales, porque en la
  banda inferior el precio está por debajo de la media de su ventana y el modelo regresa a ella.
- A 6 barras, la magnitud prevista sí correlaciona con el rebote real (IC de Spearman 0,19). Entrar
  solo en el tercil superior de subida prevista deja 27 operaciones con factor de beneficio 2,12 y
  drawdown del 2,7 %, frente a 1,20 de media (p90 1,66) de un filtro aleatorio con la misma tasa de
  paso, y es positivo en los seis símbolos. Pero en la segunda mitad del año, donde la estrategia
  pierde, el filtro solo la deja en tablas.
- A 3 barras el efecto desaparece: IC 0,08, ningún umbral supera al azar y el tercil inferior rinde
  igual que el superior.

Lectura honesta: sugerente pero no robusto con 126 señales de un solo año; no se despliega. Merece
repetirse con más historia y con Kronos-base cuando la demo lleve unos meses.

### Vencimientos de opciones

El calendario genera solo el tercer viernes de cada mes (trimestral en marzo, junio, septiembre y
diciembre, cuando vencen también futuros de índices y el S&P aplica su rebalanceo) como evento con
modo propio "congelar": desde 2 h antes hasta 30 min después del cierre de Nueva York ningún bot abre
posiciones y ninguno toca los stops, porque con el precio anclado a los strikes subir stops es
regalar la posición al ruido. Si un dato macro coincide, manda su protección. El bot swing, además,
no entra con la señal de la barra de 4 h que contiene ese cierre. Se desactiva con `"opex": false`
en `data/events.json`.

### Freno global

Los tres bots comparten un freno (`autotrader/guard.py`): la variable `BOT_HALT` (`off`, `freeze` para
no abrir posiciones nuevas, `close` para además cerrar las del bot) y `MAX_DRAWDOWN_PCT`, que congela
las entradas si el equity cae ese porcentaje desde su máximo histórico (10 % en la cuenta swing, 15 %
en las otras). El máximo se lee del historial publicado del panel y se persiste en
`state/peak_equity.json`. En GitHub Actions ambos se controlan con variables del repositorio
(Settings → Secrets and variables → Actions → Variables) sin tocar el código; en la rutina de Alpaca,
con las mismas variables de entorno. El freno nunca toca posiciones abiertas a mano.

### Atribución: bot, manual y excluidas

`autotrader/attribution.py` clasifica cada operación cerrada y cada posición abierta. En cTrader el
origen sale de la etiqueta con la que se abrió la posición (el historial de órdenes la conserva aunque
la posición ya esté cerrada), así que las manuales quedan fuera sin estimar por tamaño. Las entradas
anteriores a la fecha de corte de `data/exclusions.json` (14 de septiembre de 2026, el fallo del filtro
de huecos) se marcan como excluidas. En Alpaca las operaciones cerradas se reconstruyen por FIFO sobre
las ejecuciones. Cada pestaña muestra una tarjeta "Bot ajustado" con el capital y el retorno que
tendría la cuenta contando solo las operaciones del bot con la lógica vigente, y las tablas de
operaciones marcan el origen de cada una.

### Ejecución frente a señal

La cuenta swing registra, en cada compra, la diferencia entre el cierre de la barra que dio la señal y
el precio real de entrada, junto con el retraso con el que llegó el ciclo. El panel acumula ese
deslizamiento en USD y puntos básicos: es el coste real del retraso de GitHub Actions más el spread.

## Estructura de estado

```
state/
  sim_account.json   # efectivo y posiciones del broker simulado
  sim_orders.jsonl   # órdenes ejecutadas en el simulador
  run_log.jsonl      # un registro por ciclo con decisiones, órdenes y motivos
  day_start.json     # equity al inicio del día (límite de pérdida diaria)
```
