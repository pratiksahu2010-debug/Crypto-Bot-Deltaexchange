# CoinDCX Signal Bot — Upgrade 1–24

## Implemented

### Points 1–8: reliability and market data
1. Fixed undefined 15m confirmation reference; confirmation is explicitly directional.
2. REST candle loader normalizes timestamps to epoch milliseconds, validates OHLCV, deduplicates, sorts, and excludes unfinished bars.
3. Signal manager allows re-entry after terminal states and expires stale active records.
4. Candle buffers are rebuilt from authoritative REST snapshots; timestamps and order are normalized.
5. Added a 15-second per-pair/timeframe candle cache shared by both engines.
6. WebSocket is disabled by default; this version deliberately uses REST as the authoritative data feed rather than advertising an unstarted socket.
7. Existing dynamic symbol manager is retained; Normal/Momentum lists remain separately configured.
8. Symbol scan exceptions are now visible with stack traces and per-engine counters exposed in status/metrics.

### Points 9–20: entry and false-signal controls
9. Normal entries require EMA directional alignment and the existing EMA50 trend predicate where applicable.
10. ADX minimum is 20 for Normal; +DI/-DI must agree with trade direction and ADX must be rising unless ADX is at least 30.
11. Normal signals require 15m EMA9/21/50 alignment, daily VWAP location and EMA9 slope confirmation.
12. Anti-chasing rejects entries more than 1.25 ATR from EMA9.
13. Breakout predicates receive real resistance/support levels and use direction-specific close breaks, relative volume and candle body quality.
14. Pullback predicate is now actually required to classify a Normal signal; the engine no longer merely labels unqualified signals.
15. Momentum requires a substantive explosion, squeeze-release or continuation state, not a generic fallback label.
16. Reversal strategy remains separately classified and must pass the engine's directional context and confirmation gates.
17. Candle body is now body/range; close location and upper/lower wick ratios are calculated and used for directional candle validation.
18. VWAP is UTC-session anchored, rather than a rolling 100-bar average.
19. Directional structure scoring no longer rewards a break in the wrong direction; mandatory DI/context gates reduce correlated-score false positives.
20. Only confirmed completed-candle signals are sent; strategy predicates and mandatory confirmation gates execute before scoring.

### Points 21–24: risk and lifecycle
21. Stop loss uses recent swing structure with ATR bounds.
22. Targets are risk multiples (1.5R, 2.5R, 4R), avoiding the previous 0.67R TP1 design.
23. Entry zones are narrower ATR-based zones around the signal price; price is explicitly the completed candle close.
24. Active alerts expire; later completed candles are monitored for TP/SL. If SL and TP occur in the same candle, SL is assumed first. Terminal signals no longer block a future setup.

## Validation

- `python -m compileall -q .` — passed.
- `PYTHONPATH=. python -m pytest -q tests/test_regressions.py tests/test_scoring.py` — 8 passed.
- Full `tests/test_health.py` collection could not be completed in the audit container because `python-telegram-bot` is not installed and package installation is unavailable in this environment. Render installs it from `requirements.txt`.

## Important limitations

- This is not a claim of perfect entries or proven profitability. Win rate, expectancy, false-positive rate and missed-move rate still require historical replay and forward paper-signal data.
- REST-only scanning is intentional for this release; WebSocket is disabled until a verified feed integration is implemented.
- Signal outcome state is tracked internally; this patch does not add new Telegram TP/SL follow-up notifications.
- No order execution has been introduced.
