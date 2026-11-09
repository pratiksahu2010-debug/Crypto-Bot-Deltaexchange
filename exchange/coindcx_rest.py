"""CoinDCX REST client — public market data only. No order placement."""
import asyncio
from typing import Any, Dict, List, Optional
import httpx
import time
from utils.time_utils import parse_interval

from config import settings
from utils.logger import logger
from utils.retry import async_retry


def _should_retry_coindcx_request(exc: Exception) -> bool:
    """Retry transport failures, rate limits and server errors, not permanent 4xx."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status in (408, 425, 429) or status >= 500
    return True


class CoinDCXRest:
    def __init__(self):
        self.base = settings.coindcx_rest_base
        self.public = settings.coindcx_public_base
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore = asyncio.Semaphore(5)  # rate-limit friendly
        self._candle_cache = {}
        self._ticker_snapshot = None
        self._ticker_snapshot_at = 0.0
        self._ticker_lock = asyncio.Lock()
        self._orderbook_cache: Dict[str, tuple] = {}
        self._ws_prices: Dict[str, tuple] = {}
        self._ws_last_update_ms: Optional[int] = None
        self._last_ws_price_prune_mono = 0.0
        self.MAX_WS_PRICE_CACHE_ENTRIES = 2000
        self.WS_PRICE_CACHE_PRUNE_INTERVAL_SECONDS = 30.0
        # pair -> timeframe -> consecutive stale API observations
        self._stale_market_timeframes: Dict[str, Dict[str, int]] = {}
        self.request_count = 0
        self.request_failures = 0
        self.candle_cache_hits = 0
        self.last_successful_request_at = None

    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(15.0, connect=10.0),
                headers={"User-Agent": "CoinDCX-SignalBot/2.0"},
            )

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @async_retry(max_attempts=3, delay=1.0, retry_if=_should_retry_coindcx_request)
    async def _get(self, url: str, params: Optional[Dict] = None) -> Any:
        async with self._semaphore:
            assert self._client is not None
            self.request_count += 1
            try:
                resp = await self._client.get(url, params=params)
                resp.raise_for_status()
                payload = resp.json()
                self.last_successful_request_at = time.time()
                return payload
            except Exception:
                self.request_failures += 1
                raise

    async def get_markets(self) -> List[str]:
        """Return Futures instrument names only."""
        return [str(row.get("pair") or row.get("symbol") or "")
                for row in await self.get_market_details() if isinstance(row, dict)]

    async def get_market_details(self) -> List[Dict]:
        """Discover active USDT Futures instruments; never query Spot markets."""
        data = await self._get(
            f"{self.base}/exchange/v1/derivatives/futures/data/active_instruments",
            params={"margin_currency_short_name[]": "USDT"},
        )
        rows = data if isinstance(data, list) else (
            data.get("data", []) if isinstance(data, dict) else []
        )
        normalized = []
        for row in rows:
            # CoinDCX's documented active_instruments response is a JSON list
            # of instrument-name strings (for example, "B-BTC_USDT"), not
            # instrument metadata objects. Accept both the documented shape
            # and object/wrapped variants for forward compatibility.
            if isinstance(row, str):
                raw_pair = row.strip().upper()
                metadata = {}
            elif isinstance(row, dict):
                raw_pair = str(
                    row.get("pair") or row.get("symbol") or row.get("instrument") or ""
                ).strip().upper()
                metadata = row
            else:
                continue
            if not raw_pair:
                continue
            pair = raw_pair
            if not pair.startswith(("B-", "I-", "HB-", "KC-")):
                pair = "B-" + pair.replace("/", "_")
            if not pair.endswith("_USDT"):
                continue
            quote = str(
                metadata.get("margin_currency_short_name")
                or metadata.get("quote_currency_short_name")
                or "USDT"
            ).upper()
            if quote != "USDT":
                continue
            target = str(metadata.get("target_currency_short_name") or "").upper()
            if not target:
                target = pair.split("-", 1)[-1].split("_", 1)[0]
            normalized.append({
                **metadata,
                "pair": pair,
                "coindcx_name": pair,
                "base_currency_short_name": "USDT",
                "target_currency_short_name": target,
                "status": metadata.get("status", "active"),
            })
        # The endpoint should not return duplicates, but deduplicate defensively
        # while preserving exchange ordering.
        unique = {}
        for item in normalized:
            unique.setdefault(item["pair"], item)
        return list(unique.values())

    @staticmethod
    def _timestamp_ms(value: Any) -> Optional[int]:
        """Normalize Unix seconds or milliseconds without accepting invalid epochs."""
        try:
            stamp = float(value)
            if not __import__("math").isfinite(stamp) or stamp <= 0:
                return None
            # Current Unix seconds are ~1e9; milliseconds are ~1e12.
            stamp_ms = int(stamp * 1000) if stamp < 10**11 else int(stamp)
            # Reject implausible epochs instead of silently making stale data fresh.
            if stamp_ms < 946684800000 or stamp_ms > 4102444800000:
                return None
            return stamp_ms
        except (TypeError, ValueError, OverflowError):
            return None

    # Bound future exchange/host clock skew; never accept arbitrary future quotes.
    MAX_FUTURES_CLOCK_SKEW_SECONDS = 15.0
    MAX_REST_TICKER_AGE_SECONDS = 10.0
    MAX_WS_TICKER_AGE_SECONDS = 5.0

    @classmethod
    def _ticker_age_seconds(cls, timestamp: Any, now_ms: int) -> Optional[float]:
        stamp_ms = cls._timestamp_ms(timestamp)
        return (now_ms - stamp_ms) / 1000.0 if stamp_ms is not None else None

    @classmethod
    def _is_fresh_ticker_age(cls, age: Optional[float], max_age: float) -> bool:
        return age is not None and -cls.MAX_FUTURES_CLOCK_SKEW_SECONDS <= age <= max_age

    @staticmethod
    def _normalize_ticker_row(market: Any, quote: Any, fallback_ts: Any = None) -> Optional[Dict]:
        """Normalize Futures ticker rows and retain the timestamp's provenance."""
        if not isinstance(quote, dict):
            return None
        market_name = market or quote.get("market") or quote.get("pair") or quote.get("symbol") or quote.get("mkt")
        if not market_name:
            return None
        row_ts = quote.get("ts") or quote.get("timestamp")
        timestamp = row_ts if row_ts is not None else fallback_ts
        return {
            "market": str(market_name),
            "pair": quote.get("pair") or market_name,
            "symbol": quote.get("symbol") or quote.get("mkt") or market_name,
            "last_price": quote.get("ls") or quote.get("last_price") or quote.get("lastPrice"),
            "mark_price": quote.get("mp") or quote.get("mark_price"),
            "timestamp": timestamp,
            "timestamp_source": "row" if row_ts is not None else ("root" if fallback_ts is not None else "missing"),
            "quote_volume": quote.get("quote_volume") or quote.get("quoteVolume") or 0,
            "base_volume": quote.get("v") or quote.get("bv") or quote.get("base_volume") or quote.get("volume") or 0,
        }

    MAX_ORDERBOOK_AGE_SECONDS = 5.0
    ORDERBOOK_CACHE_TTL_SECONDS = 1.0

    @staticmethod
    def _orderbook_levels(levels: Any) -> List[tuple]:
        """Normalize CoinDCX orderbook levels from mapping or list payloads."""
        if isinstance(levels, dict):
            items = list(levels.items())
        elif isinstance(levels, list):
            items = []
            for level in levels:
                if isinstance(level, dict):
                    items.append((level.get("price", level.get("p")),
                                  level.get("quantity", level.get("q"))))
                elif isinstance(level, (list, tuple)) and len(level) >= 2:
                    items.append((level[0], level[1]))
        else:
            return []
        normalized = []
        for raw_price, raw_quantity in items:
            try:
                price, quantity = float(raw_price), float(raw_quantity)
            except (TypeError, ValueError, OverflowError):
                continue
            if price > 0 and quantity > 0 and __import__("math").isfinite(price) and __import__("math").isfinite(quantity):
                normalized.append((price, quantity))
        return normalized

    def _prune_orderbook_cache(self, now_mono: Optional[float] = None) -> None:
        """Discard expired snapshots and bound cache growth across unique markets."""
        now = time.monotonic() if now_mono is None else now_mono
        cutoff = now - self.ORDERBOOK_CACHE_TTL_SECONDS
        self._orderbook_cache = {
            key: value for key, value in self._orderbook_cache.items()
            if value[0] >= cutoff
        }
        if len(self._orderbook_cache) > 1000:
            for old_key in sorted(
                self._orderbook_cache,
                key=lambda key: self._orderbook_cache[key][0],
            )[:len(self._orderbook_cache) - 1000]:
                self._orderbook_cache.pop(old_key, None)

    async def get_futures_orderbook(self, pair: str, depth: int = 10) -> Dict[str, Any]:
        """Return a fresh, normalized public Futures orderbook snapshot.

        The endpoint is queried only for a signal candidate, not for every
        symbol in the scan. Stale, crossed, empty, or malformed books fail closed.
        """
        market = str(pair or "").strip().upper()
        if not market.startswith("B-") or not market.endswith("_USDT"):
            raise ValueError("orderbook requires a USDT Futures instrument pair")
        if depth not in (10, 20, 50):
            raise ValueError("unsupported Futures orderbook depth")
        cache_key = f"{market}:{depth}"
        now_mono = time.monotonic()
        self._prune_orderbook_cache(now_mono)
        cached = self._orderbook_cache.get(cache_key)
        if cached and now_mono - cached[0] <= self.ORDERBOOK_CACHE_TTL_SECONDS:
            cached_snapshot = dict(cached[1])
            cached_age = self._ticker_age_seconds(
                cached_snapshot.get("timestamp"), int(time.time() * 1000)
            )
            if self._is_fresh_ticker_age(cached_age, self.MAX_ORDERBOOK_AGE_SECONDS):
                cached_snapshot["age_seconds"] = cached_age
                return cached_snapshot
            self._orderbook_cache.pop(cache_key, None)

        payload = await self._get(
            f"{self.public}/market_data/v3/orderbook/{market}-futures/{depth}"
        )
        if not isinstance(payload, dict):
            raise ValueError("invalid Futures orderbook payload")
        timestamp = payload.get("ts") or payload.get("timestamp")
        now_ms = int(time.time() * 1000)
        age = self._ticker_age_seconds(timestamp, now_ms)
        if not self._is_fresh_ticker_age(age, self.MAX_ORDERBOOK_AGE_SECONDS):
            raise ValueError("stale or missing Futures orderbook timestamp")

        bids = self._orderbook_levels(payload.get("bids"))
        asks = self._orderbook_levels(payload.get("asks"))
        if not bids or not asks:
            raise ValueError("Futures orderbook has no valid two-sided liquidity")
        best_bid = max(price for price, _ in bids)
        best_ask = min(price for price, _ in asks)
        if best_bid <= 0 or best_ask <= best_bid:
            raise ValueError("Futures orderbook is crossed or locked")
        midpoint = (best_bid + best_ask) / 2.0
        spread_bps = ((best_ask - best_bid) / midpoint) * 10_000.0
        if not __import__("math").isfinite(spread_bps) or spread_bps < 0:
            raise ValueError("invalid Futures orderbook spread")

        # Aggregate quote-notional across the nearest five price levels on
        # each side. CoinDCX Futures level quantities are base-asset amounts.
        bid_levels = sorted(bids, key=lambda level: level[0], reverse=True)[:5]
        ask_levels = sorted(asks, key=lambda level: level[0])[:5]
        bid_depth_usdt = sum(price * quantity for price, quantity in bid_levels)
        ask_depth_usdt = sum(price * quantity for price, quantity in ask_levels)
        min_side_depth_usdt = min(bid_depth_usdt, ask_depth_usdt)
        if not all(__import__("math").isfinite(value) and value >= 0 for value in (
            bid_depth_usdt, ask_depth_usdt, min_side_depth_usdt
        )):
            raise ValueError("invalid Futures orderbook depth")
        result = {
            "pair": market, "best_bid": best_bid, "best_ask": best_ask,
            "mid_price": midpoint, "spread_bps": spread_bps,
            "bid_depth_usdt": bid_depth_usdt, "ask_depth_usdt": ask_depth_usdt,
            "min_side_depth_usdt": min_side_depth_usdt, "depth_levels": 5,
            "timestamp": self._timestamp_ms(timestamp), "age_seconds": age,
            "depth": depth,
        }
        self._orderbook_cache[cache_key] = (time.monotonic(), result)
        self._prune_orderbook_cache()
        return dict(result)

    async def get_ticker(self) -> List[Dict]:
        """Return normalized Futures current-price rows. Spot data is never used."""
        now = time.monotonic()
        if self._ticker_snapshot is not None and now - self._ticker_snapshot_at <= 1.0:
            return self._ticker_snapshot
        async with self._ticker_lock:
            now = time.monotonic()
            if self._ticker_snapshot is not None and now - self._ticker_snapshot_at <= 1.0:
                return self._ticker_snapshot
            data = await self._get(f"{self.public}/market_data/v3/current_prices/futures/rt")
            fallback_ts = None
            payload = data
            # Accept direct and nested API envelopes such as
            # {ts, prices}, {data: {ts, prices}}, or {data: {ts, market: quote}}.
            for _ in range(3):
                if not isinstance(payload, dict):
                    break
                candidate_ts = payload.get("ts") or payload.get("timestamp")
                if candidate_ts is not None:
                    fallback_ts = candidate_ts
                if isinstance(payload.get("prices"), (dict, list)):
                    payload = payload["prices"]
                    break
                if isinstance(payload.get("data"), (dict, list)):
                    payload = payload["data"]
                    continue
                break
            rows: List[Dict] = []
            if isinstance(payload, dict):
                for market, quote in payload.items():
                    normalized = self._normalize_ticker_row(market, quote, fallback_ts)
                    if normalized:
                        rows.append(normalized)
            elif isinstance(payload, list):
                for quote in payload:
                    normalized = self._normalize_ticker_row(None, quote, fallback_ts)
                    if normalized:
                        rows.append(normalized)

            now_ms = int(time.time() * 1000)
            valid_ltp = 0
            fresh_ages = []
            stale_rows = 0
            for row in rows:
                try:
                    price = float(row.get("last_price") or 0)
                    age = self._ticker_age_seconds(row.get("timestamp"), now_ms)
                    if price > 0 and __import__("math").isfinite(price):
                        valid_ltp += 1
                        if self._is_fresh_ticker_age(age, self.MAX_REST_TICKER_AGE_SECONDS):
                            fresh_ages.append(age)
                        else:
                            stale_rows += 1
                except (TypeError, ValueError, OverflowError):
                    stale_rows += 1
            logger.info(
                f"FUTURES_TICKER_SNAPSHOT rows={len(rows)} valid_ltp={valid_ltp} "
                f"fresh_ltp={len(fresh_ages)} stale_ltp={stale_rows} "
                f"oldest_fresh_age_s={max(fresh_ages):.2f}"
                if fresh_ages else
                f"FUTURES_TICKER_SNAPSHOT rows={len(rows)} valid_ltp={valid_ltp} fresh_ltp=0 stale_ltp={stale_rows}"
            )
            self._ticker_snapshot = rows
            self._ticker_snapshot_at = time.monotonic()
            return rows

    @staticmethod
    def _market_key(value: Any) -> str:
        import re
        text = str(value or "").upper().strip()
        text = re.sub(r"^(B-|I-)", "", text)
        return re.sub(r"[^A-Z0-9]", "", text)

    def get_stale_market_timeframes(self) -> Dict[str, Dict[str, int]]:
        """Return a copy of stale candle-feed observations for universe quarantine."""
        return {pair: dict(timeframes) for pair, timeframes in self._stale_market_timeframes.items()}

    def _prune_ws_price_cache(self, now_ms: Optional[int] = None, force: bool = False) -> None:
        """Expire stale Futures quotes and cap retained markets to a fixed bound."""
        now_mono = time.monotonic()
        if (
            not force
            and now_mono - self._last_ws_price_prune_mono
            < self.WS_PRICE_CACHE_PRUNE_INTERVAL_SECONDS
            and len(self._ws_prices) <= self.MAX_WS_PRICE_CACHE_ENTRIES
        ):
            return
        current_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
        cutoff_ms = current_ms - int(self.MAX_WS_TICKER_AGE_SECONDS * 1000)
        self._ws_prices = {
            key: value for key, value in self._ws_prices.items()
            if isinstance(value, tuple) and len(value) >= 2 and value[1] >= cutoff_ms
        }
        if len(self._ws_prices) > self.MAX_WS_PRICE_CACHE_ENTRIES:
            newest = sorted(
                self._ws_prices.items(),
                key=lambda item: item[1][1],
                reverse=True,
            )[:self.MAX_WS_PRICE_CACHE_ENTRIES]
            self._ws_prices = dict(newest)
        self._last_ws_price_prune_mono = now_mono

    def set_ws_prices(self, prices: Dict[str, Any], timestamp_ms: Optional[int] = None) -> int:
        """Accept Futures LTP updates; nested market payloads use ls, never mark price."""
        now_ms = int(time.time() * 1000)
        accepted = 0
        root_ts = self._timestamp_ms(timestamp_ms) if timestamp_ms is not None else now_ms
        if root_ts is None:
            return 0
        root_age = (now_ms - root_ts) / 1000.0
        if not self._is_fresh_ticker_age(root_age, self.MAX_WS_TICKER_AGE_SECONDS):
            logger.warning(f"COINDCX_WS_ROOT_TIMESTAMP_REJECTED age_seconds={root_age:.3f} max_future_skew_s={self.MAX_FUTURES_CLOCK_SKEW_SECONDS:g}")
            return 0
        accepted = 0
        for market, raw in prices.items():
            try:
                if isinstance(raw, dict):
                    value = raw.get("ls") or raw.get("last_price") or raw.get("lastPrice")
                    raw_event_ts = raw.get("ts") or raw.get("timestamp") or timestamp_ms
                else:
                    value, raw_event_ts = raw, timestamp_ms
                event_ts = self._timestamp_ms(raw_event_ts)
                price = float(value)
                if (
                    event_ts is not None and price > 0
                    and __import__("math").isfinite(price)
                    and self._is_fresh_ticker_age((now_ms - event_ts) / 1000.0, self.MAX_WS_TICKER_AGE_SECONDS)
                ):
                    market_key = self._market_key(market)
                    cached = self._ws_prices.get(market_key)
                    # WebSocket delivery can be reordered. Never let an older
                    # exchange event roll the live-price cache backwards.
                    if cached is not None and event_ts < cached[1]:
                        continue
                    self._ws_prices[market_key] = (price, event_ts)
                    accepted += 1
            except (TypeError, ValueError, OverflowError):
                continue
        self._prune_ws_price_cache(now_ms)
        if accepted:
            self._ws_last_update_ms = now_ms
        return accepted

    async def get_last_price(self, pair: str) -> Optional[float]:
        """Return fresh Futures LTP only. Missing/stale Futures data fails closed."""
        now_ms = int(time.time() * 1000)
        self._prune_ws_price_cache(now_ms)
        wanted = self._market_key(pair)
        cached = self._ws_prices.get(wanted)
        now_ms = int(time.time() * 1000)
        if cached and self._is_fresh_ticker_age((now_ms - cached[1]) / 1000.0, self.MAX_WS_TICKER_AGE_SECONDS):
            return cached[0]
        logger.debug(f"FUTURES_LTP_REST_FALLBACK pair={pair} reason=ws_cache_missing_or_stale")
        rows = await self.get_ticker()
        matched = 0
        stale = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            aliases = (row.get("market"), row.get("pair"), row.get("symbol"))
            if not any(self._market_key(alias) == wanted for alias in aliases if alias):
                continue
            matched += 1
            try:
                age = self._ticker_age_seconds(row.get("timestamp"), now_ms)
                if not self._is_fresh_ticker_age(age, self.MAX_REST_TICKER_AGE_SECONDS):
                    stale += 1
                    logger.warning(
                        f"FUTURES_LTP_ROW_REJECTED pair={pair} matched_market={row.get('market')} "
                        f"timestamp_source={row.get('timestamp_source', 'unknown')} "
                        f"timestamp={row.get('timestamp')} age_seconds={age}"
                    )
                    continue
                price = float(row.get("last_price") or 0)
                if price > 0 and __import__("math").isfinite(price):
                    return price
            except (TypeError, ValueError, OverflowError):
                stale += 1
                continue
        logger.warning(
            f"FUTURES_LTP_UNAVAILABLE pair={pair} ticker_rows={len(rows)} "
            f"matched={matched} stale={stale}"
        )
        return None

    def _record_candle_feed_observation(self, pair: str, interval: str, available: bool) -> None:
        """Track unavailable as well as stale candle feeds for symbol quarantine."""
        if available:
            self._stale_market_timeframes.get(pair, {}).pop(interval, None)
            return
        counts = self._stale_market_timeframes.setdefault(pair, {})
        counts[interval] = counts.get(interval, 0) + 1

    def _prune_candle_cache(self) -> None:
        """Keep candle cache bounded even on early malformed-response returns."""
        if len(self._candle_cache) <= 1000:
            return
        cutoff = time.monotonic() - 15
        self._candle_cache = {
            key: value for key, value in self._candle_cache.items()
            if value[0] >= cutoff
        }
        if len(self._candle_cache) > 1000:
            for old_key in list(self._candle_cache)[:len(self._candle_cache) - 1000]:
                self._candle_cache.pop(old_key, None)

    async def get_candles(
        self, pair: str, interval: str = "5m", limit: int = 200,
        start_time: Optional[int] = None, end_time: Optional[int] = None,
    ) -> List[Dict]:
        """Fetch completed Futures candles; synthesize 15m from three Futures 5m bars."""
        # Only default live-scan windows contribute to symbol feed health.
        # Explicit historical/backfill requests must not quarantine a market.
        track_feed_health = start_time is None and end_time is None
        # Validate the request before constructing cache keys or making exchange
        # calls. In particular, limit=0 makes Python's [-0:] return every candle.
        if isinstance(limit, bool) or (isinstance(limit, float) and not limit.is_integer()):
            raise ValueError("Candle limit must be an integer between 1 and 1000")
        try:
            limit = int(limit)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Candle limit must be an integer between 1 and 1000") from exc
        if limit < 1 or limit > 1000:
            raise ValueError("Candle limit must be between 1 and 1000")
        if not isinstance(pair, str) or not pair.strip():
            raise ValueError("Futures pair must be a non-empty string")
        pair = pair.strip()
        if start_time is not None:
            try:
                start_time = int(start_time)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("start_time must be a positive Unix timestamp in milliseconds") from exc
            if start_time <= 0:
                raise ValueError("start_time must be a positive Unix timestamp in milliseconds")
        if end_time is not None:
            try:
                end_time = int(end_time)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("end_time must be a positive Unix timestamp in milliseconds") from exc
            if end_time <= 0:
                raise ValueError("end_time must be a positive Unix timestamp in milliseconds")
        if start_time is not None and end_time is not None and start_time >= end_time:
            raise ValueError("start_time must be earlier than end_time")
        if interval == "15m":
            end_ms = int(end_time or time.time() * 1000)
            start_ms = int(start_time or (end_ms - max(limit + 5, 100) * 15 * 60 * 1000))
            five = await self.get_candles(
                pair, "5m", limit=min(max(limit * 3 + 12, 100), 1000),
                start_time=start_ms, end_time=end_ms,
            )
            buckets: Dict[int, List[Dict]] = {}
            width = 15 * 60 * 1000
            for candle in five:
                bucket = int(candle["time"]) // width * width
                buckets.setdefault(bucket, []).append(candle)
            result = []
            for bucket, bars in sorted(buckets.items()):
                bars.sort(key=lambda x: x["time"])
                # A synthesized candle is valid only when its entire 15m
                # interval is contained in the caller's requested window.
                if bucket < start_ms or bucket + width > end_ms:
                    continue
                if len(bars) != 3 or [x["time"] for x in bars] != [bucket, bucket + 300000, bucket + 600000]:
                    continue
                result.append({
                    "time": bucket, "open": bars[0]["open"],
                    "high": max(x["high"] for x in bars),
                    "low": min(x["low"] for x in bars),
                    "close": bars[-1]["close"],
                    "volume": sum(x["volume"] for x in bars),
                })
            result = result[-limit:]
            if track_feed_health:
                self._record_candle_feed_observation(pair, interval, bool(result))
            return result

        resolution_map = {"1m": "1", "5m": "5", "1h": "60", "1d": "1D"}
        resolution = resolution_map.get(interval)
        if resolution is None:
            raise ValueError(f"Unsupported CoinDCX Futures candle interval: {interval}")
        now_ms = int(time.time() * 1000)
        end_ms = int(end_time or now_ms)
        duration_ms = int(parse_interval(interval).total_seconds() * 1000)
        grace_ms = max(0, int(settings.candle_close_grace_seconds)) * 1000
        # A cached default-window response is valid only within the same
        # completed-candle bucket. Otherwise a pre-close response could hide
        # the candle that becomes eligible immediately after the close grace.
        cache_boundary = (
            (now_ms - grace_ms) // duration_ms if end_time is None else None
        )
        cache_key = (pair, interval, limit, start_time, end_time, cache_boundary)
        cached = self._candle_cache.get(cache_key)
        if cached and time.monotonic() - cached[0] < 5:
            self.candle_cache_hits += 1
            return [dict(x) for x in cached[1]]
        start_ms = int(start_time or (end_ms - min(limit, 1000) * duration_ms))
        params = {
            "pair": pair, "from": int(start_ms / 1000),
            "to": int(end_ms / 1000), "resolution": resolution, "pcode": "f",
        }
        data = await self._get(f"{self.public}/market_data/candlesticks", params=params)
        payload = data
        for _ in range(3):
            if not isinstance(payload, dict):
                break
            if isinstance(payload.get("candles"), list):
                payload = payload["candles"]
                break
            if isinstance(payload.get("data"), (dict, list)):
                payload = payload["data"]
                continue
            break
        data = payload
        if not isinstance(data, list):
            if track_feed_health:
                self._record_candle_feed_observation(pair, interval, False)
            # Cache malformed/empty-envelope responses briefly as well. Without
            # this, each engine can repeat the same failed REST request during
            # one scan cycle and amplify exchange load for an unhealthy market.
            self._candle_cache[cache_key] = (time.monotonic(), [])
            self._prune_candle_cache()
            return []
        normalized = []
        for row in data:
            if not isinstance(row, dict):
                continue
            try:
                raw_ts = row.get("time", row.get("t", row.get("timestamp")))
                ts = int(float(raw_ts))
                if ts < 10**12:
                    ts *= 1000
                candle = {
                    "time": ts, "open": float(row.get("open", row.get("o", 0))),
                    "high": float(row.get("high", row.get("h", 0))),
                    "low": float(row.get("low", row.get("l", 0))),
                    "close": float(row.get("close", row.get("c", 0))),
                    "volume": float(row.get("volume", row.get("v", 0))),
                }
                vals = [candle[k] for k in ("open", "high", "low", "close", "volume")]
                if not all(__import__("math").isfinite(v) for v in vals):
                    continue
                if min(vals[:4]) <= 0 or candle["volume"] < 0:
                    continue
                if candle["high"] < max(candle["open"], candle["close"], candle["low"]) or candle["low"] > min(candle["open"], candle["close"], candle["high"]):
                    continue
                # The exchange may return candles just outside the requested
                # window (for example, inclusive end boundaries). Never let
                # those rows leak into historical warmup or backfill results.
                if ts < start_ms or ts >= end_ms:
                    continue
                if ts % duration_ms == 0 and ts + duration_ms + max(0, int(settings.candle_close_grace_seconds))*1000 <= now_ms:
                    normalized.append(candle)
            except (TypeError, ValueError, OverflowError):
                continue
        dedup = {c["time"]: c for c in normalized}
        candles = [dedup[k] for k in sorted(dedup)][-limit:]
        if candles:
            latest_close = candles[-1]["time"] + duration_ms
            age = (now_ms - latest_close) / 1000
            stale_limit = max(float(settings.max_signal_age_seconds), duration_ms * 2 / 1000)
            if age > stale_limit:
                if track_feed_health:
                    self._record_candle_feed_observation(pair, interval, False)
                logger.warning(f"FUTURES_CANDLES_STALE {pair} {interval} age={age:.1f}s; rejected")
                candles = []
            elif track_feed_health:
                self._record_candle_feed_observation(pair, interval, True)
        elif track_feed_health:
            # Empty, malformed, or entirely unfinished responses are unavailable
            # live-feed observations too; otherwise these symbols evade quarantine.
            self._record_candle_feed_observation(pair, interval, False)
        self._candle_cache[cache_key] = (time.monotonic(), [dict(x) for x in candles])
        self._prune_candle_cache()
        return candles

