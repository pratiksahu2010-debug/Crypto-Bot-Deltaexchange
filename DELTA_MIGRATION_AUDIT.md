# Delta Exchange India migration audit

## Scope

This release changes the active exchange data provider from CoinDCX Futures to Delta Exchange India's public perpetual-futures market data. The strategy engine, indicators, signal lifecycle, risk model, Telegram delivery, persistence, health routes and deployment model are retained.

## Active exchange integration

| Purpose | Delta India API |
|---|---|
| Product discovery | `GET /v2/products?contract_types=perpetual_futures&states=live` |
| Ticker / 24h liquidity | `GET /v2/tickers?contract_types=perpetual_futures` |
| OHLCV | `GET /v2/history/candles` |
| L2 order book | `GET /v2/l2orderbook/{symbol}` |
| Public WebSocket | `wss://public-socket.india.delta.exchange` |
| Candle channels | `candlestick_5m`, `candlestick_15m` (based on configured timeframes) |

The REST base defaults to `https://api.india.delta.exchange`. All these data endpoints are public; the bot does not request account permissions and has no order-placement implementation.

## Symbol compatibility

The strategy and persistence layers continue using canonical keys such as `B-BTC_USDT`. The Delta adapter builds a runtime map between each canonical key and an actual live USD-quoted Delta perpetual product symbol (typically `BTCUSD`). It does not synthesize product symbols by changing suffixes.

The static allowlists remain fixed. Only products returned as `state=live`, `contract_type=perpetual_futures`, `trading_status=operational`, and USD-quoted are admitted. Unavailable or non-matching allowlist assets remain excluded with diagnostics; there is no dynamic substitute. Products whose symbol/underlying mapping is ambiguous are not allowed to silently collide.

## Data normalization and safety

- Delta REST success envelopes are unwrapped from `result`.
- Product pagination follows the `meta.after` cursor.
- Delta ticker `close` is normalized as the last traded price and `turnover_usd` is preferred for the existing liquidity filter.
- Delta REST timestamps in seconds and compact WebSocket timestamps in microseconds are normalized to UTC epoch milliseconds.
- Native 15-minute candles replace the old 5-minute-to-15-minute synthetic aggregation.
- Completed-candle gating, close grace, UTC alignment, deduplication, malformed OHLC rejection, stale-feed quarantine and bounded caches are retained.
- The Delta public WebSocket uses the current JSON subscribe protocol and `candlestick_<resolution>` channels. It is best-effort; REST remains authoritative for completed candles.
- WebSocket candle close is used as traded-price LTP; mark price is never substituted for LTP.
- Delta L2 book sizes are contracts. The adapter converts quote notional only when `contract_value` and `contract_unit_currency` permit a defensible conversion. Unknown depth is not fabricated.
- Signal-only enforcement remains unchanged: `SIGNAL_ONLY=true`, `ORDERS_ENABLED=false`; there are no private account channels or order APIs in the active adapter.

## Strategy preservation

No Normal, Momentum Explosion, ORB, scoring, ADX, RVOL, confirmation, cooldown, retest, ATR stop/target, Telegram outbox, database or signal-deduplication rules were intentionally changed. The Normal ADX floor remains 25. The existing `_USDT` environment-variable names for liquidity/depth thresholds are retained for compatibility; Delta's explicit USD turnover is used as the approximate equivalent liquidity unit. No FX conversion is applied.

## Validation performed in the available workspace

- `compileall` completed successfully across the project source.
- New Delta adapter tests: **8 passed**. Coverage includes product pagination and symbol mapping, ticker/turnover normalization, native 15m candles, candle window filtering, order-book contract-value conversion, compact WebSocket candle normalization and USD liquidity extraction.
- Focused strategy regression selection: **123 passed, 1 skipped**. This selection includes Normal entry/gates, Momentum Explosion, ORB, scoring, risk, retests and the new Delta adapter tests.
- The complete test suite could not be truthfully certified in this workspace: several project dependencies (including `python-telegram-bot` and `aiosqlite`) are not installed in the execution environment, and attempts to collect the full suite stop on those missing packages. The project `requirements.txt` includes them.
- No live Delta API request was made from this sandbox. Live product availability, exact account-region product coverage and current liquidity must be checked in the deployment environment before enabling production scanning.

## Deployment configuration

Set the existing Telegram and PostgreSQL variables as before. The exchange defaults are:

```env
DELTA_REST_BASE=https://api.india.delta.exchange
DELTA_WS_URL=wss://public-socket.india.delta.exchange
```

Use the `/status` endpoint and startup logs to review `eligible_symbols`, static allowlist diagnostics, liquidity exclusions and WebSocket state after deployment. Production scanning continues to require durable PostgreSQL storage.
