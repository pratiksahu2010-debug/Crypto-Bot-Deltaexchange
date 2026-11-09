# Delta Exchange India Signal Bot v3.0

Signal-only crypto futures scanner for **Delta Exchange India**. It publishes LONG/SHORT alerts to Telegram and never places orders or executes trades.

## Exchange migration

- REST: `https://api.india.delta.exchange`
- Public WebSocket: `wss://public-socket.india.delta.exchange`
- Product discovery: `GET /v2/products` filtered to live `perpetual_futures`
- Tickers: `GET /v2/tickers?contract_types=perpetual_futures`
- Candles: `GET /v2/history/candles`
- Only public market-data endpoints are used. No Delta API key or secret is needed.
- The bot retains its historical internal strategy keys (for example `B-BTC_USDT`) so persisted signals, Telegram formatting, deduplication and strategy state do not change. The exchange adapter maps these keys to discovered Delta product symbols (for example `BTCUSD`). It does not blindly replace suffixes.
- Only live, operational perpetual futures are eligible. Spot, options, expired, upcoming and disrupted contracts are excluded.
- The existing static symbol allowlists remain the source of candidate membership. A symbol not listed as a live Delta perpetual is reported unavailable; it is not silently replaced by a different asset.
- Delta's published 24-hour `turnover_usd` is used for the existing liquidity floors. The existing environment variable names ending in `_USDT` are retained for deployment compatibility; thresholds are treated as approximate USD/USDT-equivalent liquidity floors, not an FX conversion.

## Strategies — preserved

### Normal Engine (5m entries, 15m context)

- EMA 9/21, UTC-session VWAP, RSI(14), ADX(14), ATR(14), completed 15m EMA 9/21 context.
- ADX hard floor: **25**.
- Relative volume floor: **1.4**.
- Existing rising-ADX / ADX >= 30 confirmation, directional DI alignment and fresh-entry triggers remain intact.
- Fresh EMA-cross, VWAP-reclaim and EMA-pullback entry paths remain intact.

### Momentum Explosion

- Existing Bollinger Band / Keltner squeeze, ATR expansion, relative volume, EMA alignment, MACD, RSI acceleration and ROC logic are retained.
- Pre-explosion qualification and scoring remain unchanged.
- The exchange migration does not rewrite the Momentum Explosion detector.

### Opening Range Breakout (ORB)

- Existing UTC reset and Sydney, Tokyo, London, New York and CME session logic remain.
- The 15-minute range is built from three completed 5-minute candles.
- Existing range validation, extension, RVOL, ADX and scoring gates remain unchanged.

### Shared signal and risk behavior

- Existing 10-point scoring, filters, cooldowns, completed-candle confirmation, pending retests, duplicate protection and ATR-based SL/TP1/TP2/TP3 remain.
- ATR-based stop-loss and target levels use gross structural reward/risk; fees, spread and slippage are not signal rejection gates.
- Existing signal history, SQLite local / PostgreSQL production persistence, transactional claims, Telegram outbox and delivery worker remain.
- No order placement, trade execution, account access or private WebSocket channels are included.

## Market data integrity

- REST is authoritative for completed candles.
- Delta native candle resolutions are used, including 5m and 15m; 15m candles are no longer synthesized from 5m bars.
- Candle timestamps are normalized to UTC milliseconds. Malformed, duplicate, misaligned, unfinished, stale and out-of-window candles are rejected.
- The public WebSocket uses Delta's native `candlestick_<resolution>` subscription protocol. It only feeds forming-candle telemetry and LTP cache; it does not replace REST closed-candle confirmation.
- Delta WebSocket timestamps are microseconds and are normalized before freshness checks.
- REST fallback remains available when WebSocket disconnects.

## Health and keepalive

- `GET /health`, `GET /ping`, `GET /status`, `GET /metrics`
- `/ping` and `HEAD /ping` are lightweight liveness responses (204, no body) and do not trigger scans, database work, exchange calls or Telegram delivery.
- Use an external cron ping every 10–14 minutes if required by the hosting platform.
- Render runs one web instance and one Uvicorn worker. Production scanning remains paused if durable PostgreSQL storage is unavailable.

## Configuration

Required Telegram variables:

```env
TELEGRAM_BOT_TOKEN=
TELEGRAM_ADMIN_IDS=
```

Exchange variables (public endpoints; normally leave defaults):

```env
DELTA_REST_BASE=https://api.india.delta.exchange
DELTA_WS_URL=wss://public-socket.india.delta.exchange
```

Database:

```env
# Local development
DATABASE_URL=sqlite+aiosqlite:///./delta_signal_bot.db

# Production
DATABASE_URL=postgresql+asyncpg://user:password@host:5432/delta_signals
```

Existing strategy thresholds and environment variable names are retained. In particular, keep `NORMAL_ADX_MIN=25`, `NORMAL_RVOL_MIN=1.4`, `NORMAL_MIN_24H_TURNOVER_USDT=1000000`, `MOMENTUM_MIN_24H_TURNOVER_USDT=500000`, `SIGNAL_ONLY=true` and `ORDERS_ENABLED=false` unless intentionally changing strategy policy.

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill Telegram credentials and configure DATABASE_URL
python main.py
```

## Important deployment note

The adapter discovers the live Delta product catalogue at startup. Review `/status` and startup logs for eligible markets, unavailable allowlist entries and liquidity exclusions before relying on alerts. A migration build should be tested with the live public market-data API in the target deployment environment before enabling production scans.
