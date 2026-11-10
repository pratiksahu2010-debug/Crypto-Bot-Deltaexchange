# Delta Exchange India Signal Bot v3.0

Signal-only crypto futures scanner for Delta Exchange India. It publishes LONG/SHORT alerts to Telegram. It never places orders or executes trades.

## Exchange and data

- REST: https://api.india.delta.exchange
- Public WebSocket: wss://public-socket.india.delta.exchange
- Product discovery uses the public live perpetual-futures catalogue.
- Only public market-data endpoints are used; no Delta API key or secret is required.
- The adapter maps canonical strategy symbols to live Delta product symbols and excludes spot, options, expired, upcoming and disrupted contracts.
- Static allowlists define candidate membership. Unavailable symbols are reported rather than silently replaced.
- Delta 24-hour USD turnover is used for liquidity floors. Existing environment names ending in USDT are retained for compatibility.

## Signal engines

### Normal — completed 5-minute candles only

Required indicator conditions:
- EMA 9/21 directional alignment.
- Price on the correct side of UTC-session VWAP.
- RSI(14) above 50 for LONG or below 50 for SHORT.
- ADX(14) at least 25, with directional +DI/-DI alignment.
- Valid positive ATR(14).
- A fresh EMA-cross, VWAP-reclaim or EMA-pullback trigger; default maximum trigger age is one candle.
- Anti-chase limit: price-to-EMA9 extension must be no more than NORMAL_MAX_EXTENSION_ATR (default 0.80 ATR).

Relative volume is optional and disabled by default. Set NORMAL_RVOL_GATE_ENABLED=true to enable NORMAL_RVOL_MIN. Normal scoring is diagnostic after the required indicator gates; SIGNAL_THRESHOLD is not a second Normal rejection gate.

Normal does not require a 15-minute context, rising ADX, ADX of 30, second-candle follow-through, candle colour, body/wick shape, or order-book confirmation. Normal signals are checked again for freshness and entry-zone validity immediately before Telegram queuing.

### Momentum Explosion

- Separate detector states for explosion, squeeze release, pre-explosion, continuation and volume spike.
- Each detector owns its indicator qualification (EMA, RSI, MACD/ROC, squeeze, volatility and volume as applicable).
- The shared engine does not add a second universal MACD, ADX/DI or RVOL veto after a detector qualifies.
- Shared safety checks require valid ATR and reject entries more than 2.0 ATR from EMA9.
- Scores below 5.5 are rejected; 5.5 to the signal threshold are watchlist-only; alerts require at least max(8.0, SIGNAL_THRESHOLD).

### Opening Range Breakout (ORB)

- Crypto ORB runs 24/7 at four documented session opens: Asia/Daily UTC 00:00, London 08:00 Europe/London, New York 09:30 America/New_York, and optional Late US 16:00 America/New_York.
- Each session builds its own 30-minute opening range from exactly six completed 5-minute candles. OR High/Low are the highest high and lowest low across those six candles.
- London and New York use IANA time zones, so summer/winter DST shifts are automatic. Asia remains fixed at 00:00 UTC.
- Asia trades until 04:30 UTC; London and New York trade for four hours after their range closes; optional Late US trades for three hours after its range closes.
- Every session has an independent opening-range calculation, session-anchored VWAP, and RVOL baseline calculated only from volume since that session's open. Overlapping London/New York windows remain independent.
- ORB alerts require a completed 5-minute close outside that session's own range, at least three directional indicator checks (EMA 9/21 or session VWAP must agree), and session RVOL at or above ORB_MIN_RVOL. No pre-breakout alerts are emitted.
- A symbol can produce at most one LONG and one SHORT ORB alert per session date; this session/direction lock is durable in the database.

## Shared signal, risk and delivery behavior

- Only the configured primary timeframe is fetched/subscribed for closed-candle signal qualification; production entries use 5m.
- No second-candle follow-through is required.
- ATR-based stop-loss and target levels are calculated for the alert. Fees, spread and slippage are not signal rejection gates.
- Persistent deduplication is global across engines for the same pair, direction and candle, including across process restarts.
- Telegram uses push-only mode and a durable per-recipient outbox with retry and expiry. Retries cannot extend a signal's original entry-validity window.
- Signal state, watchlists, pending retests and outcomes are persisted. Production scanning pauses if durable PostgreSQL is unavailable.
- Signal-only is hard-coded true and order execution is hard-disabled. No order-placement or private trading endpoints are part of the exchange adapter.

## Market-data integrity

- REST is authoritative for completed candles; WebSocket supplies forming-candle telemetry and fresh LTP cache only.
- Candle timestamps are normalized to UTC milliseconds. Invalid, duplicate, future-dated, unfinished and stale rows are diagnosed or excluded.
- REST fallback remains available during WebSocket reconnects.
- The scanner, readiness checks and quarantine recovery require only the primary timeframe. CANDLE_CONFIRM_TF is retained only for backward-compatible environment files and has no signal role.

## Health and keepalive

- GET /health reports readiness, including database, enabled engines, scan/retest/delivery workers and per-symbol primary-candle freshness.
- GET /status and GET /metrics expose operational diagnostics.
- GET /ping and HEAD /ping return an empty 204 response. They do not scan, access the exchange, query the database or send Telegram messages.
- Render uses one web instance and one Uvicorn worker. Configure DATABASE_URL to a reachable managed PostgreSQL database for production.

## Configuration

Required Telegram variables:
- TELEGRAM_BOT_TOKEN
- TELEGRAM_ADMIN_IDS

Public exchange defaults:
- DELTA_REST_BASE=https://api.india.delta.exchange
- DELTA_WS_URL=wss://public-socket.india.delta.exchange

Database:
- Local development: sqlite+aiosqlite:///./delta_signal_bot.db
- Production: postgresql+asyncpg://user:password@host:5432/delta_signals

Important defaults:
- CANDLE_PRIMARY_TF=5m
- NORMAL_ADX_MIN=25
- NORMAL_MAX_SIGNAL_AGE_SECONDS=60
- NORMAL_MAX_EXTENSION_ATR=0.80
- NORMAL_RVOL_GATE_ENABLED=false
- MAX_SIGNAL_AGE_SECONDS=120 (Momentum/ORB)
- MOMENTUM_SYMBOL_LIMIT=100
- NORMAL_SYMBOL_LIMIT=40
- SIGNAL_ONLY=true
- ORDERS_ENABLED=false

## Run locally

Install requirements, copy .env.example to .env, fill Telegram credentials, configure DATABASE_URL, then run python main.py.

## Deployment note

Review /status and startup logs for eligible markets, unavailable allowlist entries, liquidity exclusions, database readiness and scan freshness before relying on alerts. Validate live public market-data behavior in the target deployment environment.
