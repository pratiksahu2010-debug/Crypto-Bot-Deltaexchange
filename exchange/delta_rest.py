"""Delta Exchange India REST market-data adapter.

The strategy layer keeps its historical canonical keys (B-BTC_USDT) so signal
persistence, deduplication and strategy behavior remain unchanged. This adapter
maps those keys to live Delta product symbols (BTCUSD, ETHUSD, ...).
Public market-data only: this module contains no order/trade endpoints.
"""
import asyncio
import math
import time
from typing import Any, Dict, List, Optional

import httpx

from config import settings
from utils.logger import logger
from utils.retry import async_retry


def _should_retry_delta_request(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status in (408, 425, 429) or status >= 500
    return isinstance(exc, (httpx.TimeoutException, httpx.NetworkError))


class DeltaExchangeRest:
    MAX_WS_PRICE_CACHE_ENTRIES = 2000
    WS_PRICE_CACHE_PRUNE_INTERVAL_SECONDS = 30.0
    MAX_FUTURES_CLOCK_SKEW_SECONDS = 15.0
    MAX_REST_TICKER_AGE_SECONDS = 15.0
    MAX_WS_TICKER_AGE_SECONDS = 5.0
    MAX_ORDERBOOK_AGE_SECONDS = 5.0
    ORDERBOOK_CACHE_TTL_SECONDS = 1.0

    def __init__(self):
        self.base = settings.delta_rest_base.rstrip("/")
        self.public = self.base
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore = asyncio.Semaphore(5)
        self._candle_cache = {}
        self._ticker_snapshot = None
        self._ticker_snapshot_at = 0.0
        self._ticker_lock = asyncio.Lock()
        self._orderbook_cache: Dict[str, tuple] = {}
        self._ws_prices: Dict[str, tuple] = {}
        self._ws_last_update_ms: Optional[int] = None
        self._last_ws_price_prune_mono = 0.0
        self._stale_market_timeframes: Dict[str, Dict[str, int]] = {}
        # Last REST observation for each required symbol/timeframe; this is
        # the authoritative feed-freshness source used by readiness checks.
        self._candle_feed_status: Dict[str, Dict[str, dict]] = {}
        self._canonical_to_delta: Dict[str, str] = {}
        self._delta_to_canonical: Dict[str, str] = {}
        self._product_metadata: Dict[str, dict] = {}
        self.request_count = 0
        self.request_failures = 0
        self.candle_cache_hits = 0
        self.last_successful_request_at = None

    async def start(self):
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(15.0, connect=10.0),
                headers={"Accept": "application/json", "User-Agent": "Delta-India-SignalBot/3.0"},
            )

    async def stop(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    @async_retry(max_attempts=3, delay=1.0, retry_if=_should_retry_delta_request)
    async def _get(self, url: str, params: Optional[Dict] = None) -> Any:
        async with self._semaphore:
            if self._client is None:
                await self.start()
            self.request_count += 1
            try:
                response = await self._client.get(url, params=params)
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict) and payload.get("success") is False:
                    raise ValueError(f"Delta API error: {payload.get('error')}")
                self.last_successful_request_at = time.time()
                return payload
            except Exception:
                self.request_failures += 1
                raise

    @staticmethod
    def _result(payload):
        if isinstance(payload, dict) and "result" in payload:
            return payload.get("result")
        return payload

    @staticmethod
    def _timestamp_ms(value: Any) -> Optional[int]:
        try:
            stamp = float(value)
            if not math.isfinite(stamp) or stamp <= 0:
                return None
            # Delta REST commonly uses seconds; newer WS compact channels use
            # microseconds. Also accept milliseconds for test/compat payloads.
            if stamp >= 10**14:
                stamp_ms = int(stamp / 1000)
            elif stamp >= 10**11:
                stamp_ms = int(stamp)
            else:
                stamp_ms = int(stamp * 1000)
            if stamp_ms < 946684800000 or stamp_ms > 4102444800000:
                return None
            return stamp_ms
        except (TypeError, ValueError, OverflowError):
            return None

    @classmethod
    def _ticker_age_seconds(cls, timestamp: Any, now_ms: int):
        stamp_ms = cls._timestamp_ms(timestamp)
        return (now_ms - stamp_ms) / 1000.0 if stamp_ms is not None else None

    @classmethod
    def _is_fresh_ticker_age(cls, age: Optional[float], max_age: float):
        return age is not None and -cls.MAX_FUTURES_CLOCK_SKEW_SECONDS <= age <= max_age

    @staticmethod
    def _asset_symbol(product: dict) -> str:
        # Prefer the product symbol + quoting asset: the public Product schema
        # has historically shown inconsistent example values for underlying_asset.
        symbol = str(product.get("symbol") or "").upper().strip()
        quote = product.get("quoting_asset")
        quote_symbol = str(quote.get("symbol") or "").upper() if isinstance(quote, dict) else ""
        if symbol and quote_symbol and symbol.endswith(quote_symbol):
            return symbol[:-len(quote_symbol)]
        if symbol.endswith("USD"):
            return symbol[:-3]
        if symbol.endswith("USDT"):
            return symbol[:-4]
        underlying = product.get("underlying_asset")
        if isinstance(underlying, dict):
            return str(underlying.get("symbol") or "").upper()
        return str(product.get("underlying_asset_symbol") or "").upper()

    @staticmethod
    def _canonical_for_asset(asset: str) -> str:
        return f"B-{asset.upper()}_USDT" if asset else ""

    def delta_symbol_for_pair(self, pair: str) -> str:
        key = str(pair or "").upper().strip()
        if key in self._canonical_to_delta:
            return self._canonical_to_delta[key]
        # Never synthesize an exchange symbol by suffix replacement. Only a
        # discovered live product can be requested.
        raise KeyError(f"No live Delta product mapping for strategy symbol {key}")

    def canonical_pair_for_symbol(self, symbol: str) -> str:
        return self._delta_to_canonical.get(str(symbol or "").upper().strip(), "")

    async def get_market_details(self) -> List[Dict]:
        """Fetch all live USD-quoted perpetual products with cursor pagination."""
        products = []
        after = None
        seen_cursors = set()
        for _ in range(20):
            params = {"contract_types": "perpetual_futures", "states": "live", "page_size": 100}
            if after:
                params["after"] = after
            payload = await self._get(f"{self.base}/v2/products", params=params)
            rows = self._result(payload)
            if not isinstance(rows, list):
                raise ValueError("Delta products response did not contain a result list")
            products.extend(row for row in rows if isinstance(row, dict))
            meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
            next_after = meta.get("after") if isinstance(meta, dict) else None
            if not next_after or not rows:
                break
            if next_after in seen_cursors:
                raise ValueError("Delta product pagination repeated a cursor; refusing partial universe")
            seen_cursors.add(next_after)
            after = next_after
        else:
            if after:
                raise ValueError("Delta product catalogue exceeded the 20-page safety limit")

        normalized = []
        self._canonical_to_delta.clear()
        self._delta_to_canonical.clear()
        self._product_metadata.clear()
        for product in products:
            symbol = str(product.get("symbol") or "").upper().strip()
            state = str(product.get("state") or "").lower()
            contract_type = str(product.get("contract_type") or "").lower()
            trading_status = str(product.get("trading_status") or "operational").lower()
            asset = self._asset_symbol(product)
            quote = product.get("quoting_asset")
            quote_symbol = str(quote.get("symbol") or "").upper() if isinstance(quote, dict) else ""
            if not quote_symbol and symbol.endswith("USD"):
                quote_symbol = "USD"
            # The scanner's liquidity/risk thresholds are denominated in USD
            # equivalents. Do not mix USD- and USDT/INR-quoted price series.
            if quote_symbol != "USD":
                continue
            if not symbol or not asset or state != "live" or contract_type != "perpetual_futures":
                continue
            if trading_status != "operational":
                continue
            # Exclude non-crypto underlyings and stablecoin bases using the
            # same filters already enforced by SymbolManager.
            pair = self._canonical_for_asset(asset)
            if not pair:
                continue
            self._canonical_to_delta[pair] = symbol
            self._delta_to_canonical[symbol] = pair
            self._product_metadata[pair] = product
            normalized.append({
                **product,
                "pair": pair,
                "symbol": symbol,
                "delta_symbol": symbol,
                "base_currency_short_name": "USD",
                "target_currency_short_name": asset,
                "quote_currency": quote_symbol,
                "status": "active",
                "contract_value": product.get("contract_value"),
                "contract_unit_currency": product.get("contract_unit_currency"),
            })
        # Product symbols and canonical aliases must be one-to-one; fail closed
        # rather than silently letting two contracts map to one strategy key.
        pairs = [row["pair"] for row in normalized]
        if len(pairs) != len(set(pairs)):
            duplicates = sorted({pair for pair in pairs if pairs.count(pair) > 1})
            raise ValueError(f"Multiple live Delta perpetual contracts map to same underlying: {duplicates}")
        logger.info(f"DELTA_PRODUCTS live_perpetuals={len(normalized)}")
        return normalized

    async def get_markets(self) -> List[str]:
        return [row["pair"] for row in await self.get_market_details()]

    @staticmethod
    def _normalize_ticker(row: dict, pair: str) -> Optional[dict]:
        if not isinstance(row, dict):
            return None
        symbol = str(row.get("symbol") or "").upper()
        last = row.get("close")
        if last is None:
            last = row.get("last_price")
        timestamp = row.get("timestamp") or row.get("ts")
        turnover_usd = row.get("turnover_usd")
        if turnover_usd is None and str(row.get("turnover_symbol") or "").upper() in ("USD", "USDT"):
            turnover_usd = row.get("turnover")
        quotes = row.get("quotes") if isinstance(row.get("quotes"), dict) else {}
        return {
            "market": pair, "pair": pair, "symbol": symbol,
            "delta_symbol": symbol,
            "last_price": last, "mark_price": row.get("mark_price"),
            "timestamp": timestamp,
            "timestamp_source": "row" if timestamp is not None else "missing",
            "quote_volume": turnover_usd or 0,
            "turnover_usd": turnover_usd or 0,
            "base_volume": row.get("volume") or 0,
            "best_bid": quotes.get("best_bid"),
            "best_ask": quotes.get("best_ask"),
            "contract_type": row.get("contract_type"),
        }

    async def get_ticker(self) -> List[Dict]:
        now = time.monotonic()
        if self._ticker_snapshot is not None and now - self._ticker_snapshot_at <= 1.0:
            return self._ticker_snapshot
        async with self._ticker_lock:
            now = time.monotonic()
            if self._ticker_snapshot is not None and now - self._ticker_snapshot_at <= 1.0:
                return self._ticker_snapshot
            payload = await self._get(
                f"{self.base}/v2/tickers",
                params={"contract_types": "perpetual_futures"},
            )
            rows = self._result(payload)
            if not isinstance(rows, list):
                raise ValueError("Delta ticker response did not contain a result list")
            normalized = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                symbol = str(row.get("symbol") or "").upper()
                pair = self._delta_to_canonical.get(symbol)
                if not pair:
                    # If products have not yet been loaded, derive a canonical
                    # asset key only from a documented product-symbol pattern.
                    # The symbol still must be a simple ASSETUSD perpetual.
                    if symbol.endswith("USD") and symbol not in ("MARK:BTCUSD",):
                        pair = self._canonical_for_asset(symbol[:-3])
                if pair:
                    item = self._normalize_ticker(row, pair)
                    if item:
                        normalized.append(item)
            self._ticker_snapshot = normalized
            self._ticker_snapshot_at = time.monotonic()
            return normalized

    def _prune_ws_price_cache(self, now_ms=None, force=False):
        now_mono = time.monotonic()
        if not force and now_mono - self._last_ws_price_prune_mono < self.WS_PRICE_CACHE_PRUNE_INTERVAL_SECONDS and len(self._ws_prices) <= self.MAX_WS_PRICE_CACHE_ENTRIES:
            return
        current_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
        cutoff = current_ms - int(self.MAX_WS_TICKER_AGE_SECONDS * 1000)
        self._ws_prices = {k: v for k, v in self._ws_prices.items() if isinstance(v, tuple) and len(v) >= 2 and v[1] >= cutoff}
        if len(self._ws_prices) > self.MAX_WS_PRICE_CACHE_ENTRIES:
            self._ws_prices = dict(sorted(self._ws_prices.items(), key=lambda x: x[1][1], reverse=True)[:self.MAX_WS_PRICE_CACHE_ENTRIES])
        self._last_ws_price_prune_mono = now_mono

    def set_ws_prices(self, prices: Dict[str, Any], timestamp_ms=None) -> int:
        now_ms = int(time.time() * 1000)
        accepted = 0
        for raw_symbol, raw in (prices or {}).items():
            try:
                symbol = str(raw_symbol or "").upper().strip()
                pair = self._delta_to_canonical.get(symbol)
                if not pair:
                    continue
                if isinstance(raw, dict):
                    value = raw.get("close") or raw.get("last_price") or raw.get("price") or raw.get("p")
                    event_ts = raw.get("timestamp") or raw.get("ts") or timestamp_ms
                else:
                    value, event_ts = raw, timestamp_ms
                stamp = self._timestamp_ms(event_ts)
                price = float(value)
                if stamp is None or not math.isfinite(price) or price <= 0:
                    continue
                age = (now_ms - stamp) / 1000
                if not self._is_fresh_ticker_age(age, self.MAX_WS_TICKER_AGE_SECONDS):
                    continue
                cached = self._ws_prices.get(pair)
                if cached and stamp < cached[1]:
                    continue
                self._ws_prices[pair] = (price, stamp)
                accepted += 1
            except (TypeError, ValueError, OverflowError):
                continue
        self._prune_ws_price_cache(now_ms)
        if accepted:
            self._ws_last_update_ms = now_ms
        return accepted

    async def get_last_price(self, pair: str) -> Optional[float]:
        now_ms = int(time.time() * 1000)
        self._prune_ws_price_cache(now_ms)
        cached = self._ws_prices.get(str(pair).upper())
        if cached and self._is_fresh_ticker_age((now_ms - cached[1]) / 1000, self.MAX_WS_TICKER_AGE_SECONDS):
            return cached[0]
        for row in await self.get_ticker():
            if row.get("pair") != str(pair).upper():
                continue
            age = self._ticker_age_seconds(row.get("timestamp"), now_ms)
            if not self._is_fresh_ticker_age(age, self.MAX_REST_TICKER_AGE_SECONDS):
                continue
            try:
                value = float(row.get("last_price") or 0)
                if math.isfinite(value) and value > 0:
                    return value
            except (TypeError, ValueError, OverflowError):
                pass
        return None

    def get_stale_market_timeframes(self):
        return {pair: dict(values) for pair, values in self._stale_market_timeframes.items()}

    def get_candle_feed_status(self):
        """Return a defensive snapshot of per-symbol REST candle freshness."""
        return {
            pair: {interval: dict(status) for interval, status in values.items()}
            for pair, values in self._candle_feed_status.items()
        }

    def _record_candle_feed_observation(
        self, pair, interval, available, candles=None, error=None
    ):
        now_ms = int(time.time() * 1000)
        previous = self._candle_feed_status.setdefault(pair, {}).get(interval, {})
        if available and candles:
            latest = candles[-1]
            interval_ms = int({
                "1m": 60_000, "3m": 180_000, "5m": 300_000,
                "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000,
                "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
                "12h": 43_200_000, "1d": 86_400_000, "1w": 604_800_000,
            }[interval])
            candle_open_ms = int(latest["time"])
            self._candle_feed_status[pair][interval] = {
                "status": "fresh",
                "observed_at_ms": now_ms,
                "last_completed_open_ms": candle_open_ms,
                "last_completed_close_ms": candle_open_ms + interval_ms,
                "candle_age_seconds": round(
                    max(0.0, (now_ms - candle_open_ms - interval_ms) / 1000.0), 1
                ),
                "error": None,
            }
            self._stale_market_timeframes.get(pair, {}).pop(interval, None)
        else:
            values = self._stale_market_timeframes.setdefault(pair, {})
            values[interval] = values.get(interval, 0) + 1
            self._candle_feed_status[pair][interval] = {
                **previous,
                "status": "stale",
                "observed_at_ms": now_ms,
                "error": str(error or "no_fresh_completed_candles"),
            }

    def _prune_candle_cache(self):
        if len(self._candle_cache) > 1000:
            cutoff = time.monotonic() - 15
            self._candle_cache = {k: v for k, v in self._candle_cache.items() if v[0] >= cutoff}
            if len(self._candle_cache) > 1000:
                for key in list(self._candle_cache)[:len(self._candle_cache)-1000]:
                    self._candle_cache.pop(key, None)

    async def get_candles(self, pair: str, interval="5m", limit=200, start_time=None, end_time=None):
        track = start_time is None and end_time is None
        if isinstance(limit, bool) or (isinstance(limit, float) and not limit.is_integer()):
            raise ValueError("Candle limit must be an integer between 1 and 1000")
        try:
            limit = int(limit)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Candle limit must be an integer between 1 and 1000") from exc
        if not 1 <= limit <= 1000:
            raise ValueError("Candle limit must be between 1 and 1000")
        if interval not in ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"):
            raise ValueError(f"Unsupported Delta candle interval: {interval}")
        symbol = self.delta_symbol_for_pair(pair)
        now_ms = int(time.time() * 1000)
        duration_ms = {
            "1m":60000, "3m":180000, "5m":300000, "15m":900000,
            "30m":1800000, "1h":3600000, "2h":7200000, "4h":14400000,
            "6h":21600000, "12h":43200000, "1d":86400000, "1w":604800000,
        }[interval]
        grace_ms = max(0, int(settings.candle_close_grace_seconds)) * 1000
        if start_time is not None:
            start_ms = int(start_time)
        else:
            start_ms = now_ms - (limit + 5) * duration_ms
        if end_time is not None:
            end_ms = int(end_time)
        else:
            end_ms = now_ms
        if start_ms <= 0 or end_ms <= start_ms:
            raise ValueError("Invalid candle time window")
        boundary = (now_ms - grace_ms) // duration_ms if end_time is None else None
        cache_key = (pair, interval, limit, start_ms if start_time is not None else None, end_ms if end_time is not None else None, boundary)
        cached = self._candle_cache.get(cache_key)
        if cached and time.monotonic() - cached[0] < 5:
            self.candle_cache_hits += 1
            return [dict(c) for c in cached[1]]

        params = {
            "resolution": interval, "symbol": symbol,
            "start": int(start_ms / 1000), "end": int(end_ms / 1000),
        }
        try:
            payload = await self._get(f"{self.base}/v2/history/candles", params=params)
        except Exception as exc:
            if track:
                self._record_candle_feed_observation(
                    pair, interval, False,
                    error=f"rest_request_failed:{type(exc).__name__}",
                )
            raise
        rows = self._result(payload)
        if not isinstance(rows, list):
            if track:
                self._record_candle_feed_observation(
                    pair, interval, False, error="invalid_candle_response"
                )
            self._candle_cache[cache_key] = (time.monotonic(), [])
            return []

        normalized = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                stamp = self._timestamp_ms(row.get("time"))
                if stamp is None:
                    continue
                candle = {
                    "time": stamp,
                    "open": float(row.get("open")), "high": float(row.get("high")),
                    "low": float(row.get("low")), "close": float(row.get("close")),
                    "volume": float(row.get("volume") or 0),
                }
                vals = [candle[k] for k in ("open","high","low","close","volume")]
                if not all(math.isfinite(x) for x in vals) or min(vals[:4]) <= 0 or vals[4] < 0:
                    continue
                if candle["high"] < max(candle["open"], candle["close"], candle["low"]) or candle["low"] > min(candle["open"], candle["close"], candle["high"]):
                    continue
                if stamp < start_ms or stamp >= end_ms:
                    continue
                if stamp % duration_ms == 0 and stamp + duration_ms + grace_ms <= now_ms:
                    normalized.append(candle)
            except (TypeError, ValueError, OverflowError):
                continue
        dedup = {c["time"]: c for c in normalized}
        candles = [dedup[k] for k in sorted(dedup)][-limit:]
        if candles:
            age = (now_ms - (candles[-1]["time"] + duration_ms)) / 1000
            # The 5m entry feed must be fresh enough for every enabled engine.
            # Normal has the stricter age budget; Momentum may use its own wider
            # budget only when Normal is disabled. The 15m context may naturally
            # be up to one full context interval old at a 5m close, plus that
            # same entry freshness allowance.
            enabled_age_limits = []
            if settings.enable_normal_engine:
                enabled_age_limits.append(float(settings.normal_max_signal_age_seconds))
            if settings.enable_momentum_engine:
                enabled_age_limits.append(float(settings.max_signal_age_seconds))
            entry_age_limit = min(enabled_age_limits) if enabled_age_limits else float(settings.max_signal_age_seconds)
            stale_limit = (
                entry_age_limit
                if interval == settings.candle_primary_tf
                else duration_ms / 1000.0 + entry_age_limit
            )
            if age > stale_limit:
                if track:
                    self._record_candle_feed_observation(
                        pair, interval, False, error="stale_completed_candles"
                    )
                logger.warning(f"DELTA_CANDLES_STALE {pair} {interval} age={age:.1f}s rejected")
                candles = []
            elif track:
                self._record_candle_feed_observation(
                    pair, interval, True, candles=candles
                )
        elif track:
            self._record_candle_feed_observation(
                pair, interval, False, error="no_fresh_completed_candles"
            )
        self._candle_cache[cache_key] = (time.monotonic(), [dict(c) for c in candles])
        self._prune_candle_cache()
        return candles

    @staticmethod
    def _levels(rows):
        result = []
        for row in rows or []:
            try:
                if isinstance(row, dict):
                    price, size = row.get("price"), row.get("size")
                else:
                    price, size = row[0], row[1]
                price, size = float(price), float(size)
                if price > 0 and size > 0 and math.isfinite(price) and math.isfinite(size):
                    result.append((price, size))
            except (TypeError, ValueError, IndexError, OverflowError):
                continue
        return result

    async def get_futures_orderbook(self, pair: str, depth: int = 10):
        if depth not in (10, 20, 50):
            raise ValueError("unsupported Futures orderbook depth")
        symbol = self.delta_symbol_for_pair(pair)
        now_mono = time.monotonic()
        cutoff = now_mono - self.ORDERBOOK_CACHE_TTL_SECONDS
        self._orderbook_cache = {
            key: value for key, value in self._orderbook_cache.items()
            if value[0] >= cutoff
        }
        cached = self._orderbook_cache.get(f"{pair}:{depth}")
        if cached and now_mono - cached[0] <= self.ORDERBOOK_CACHE_TTL_SECONDS:
            snapshot = dict(cached[1])
            age = self._ticker_age_seconds(snapshot.get("timestamp"), int(time.time()*1000))
            if self._is_fresh_ticker_age(age, self.MAX_ORDERBOOK_AGE_SECONDS):
                snapshot["age_seconds"] = age
                return snapshot
        payload = await self._get(f"{self.base}/v2/l2orderbook/{symbol}", params={"depth": depth})
        result = self._result(payload)
        if not isinstance(result, dict):
            raise ValueError("invalid Delta orderbook response")
        timestamp = result.get("last_updated_at") or result.get("timestamp")
        now_ms = int(time.time()*1000)
        age = self._ticker_age_seconds(timestamp, now_ms)
        if not self._is_fresh_ticker_age(age, self.MAX_ORDERBOOK_AGE_SECONDS):
            raise ValueError("stale or missing Delta orderbook timestamp")
        bids = self._levels(result.get("buy"))
        asks = self._levels(result.get("sell"))
        if not bids or not asks:
            raise ValueError("Delta orderbook has no two-sided liquidity")
        best_bid = max(x[0] for x in bids)
        best_ask = min(x[0] for x in asks)
        if best_ask <= best_bid:
            raise ValueError("Delta orderbook is crossed or locked")
        midpoint = (best_bid + best_ask) / 2
        spread_bps = (best_ask - best_bid) / midpoint * 10000
        product = self._product_metadata.get(pair, {})
        try:
            contract_value = float(product.get("contract_value") or 0)
        except (TypeError, ValueError):
            contract_value = 0
        unit_currency = str(product.get("contract_unit_currency") or "").upper()
        asset = self._asset_symbol(product)
        quote = product.get("quoting_asset")
        quote_currency = str(quote.get("symbol") or "").upper() if isinstance(quote, dict) else "USD"
        # Delta L2 size is contracts, not base-asset quantity. Convert using
        # the product's own contract unit; never assume every contract is 1 coin.
        if contract_value <= 0:
            depth_usd = None
        elif unit_currency == asset:
            bid_depth = sum(p * q * contract_value for p, q in sorted(bids, reverse=True)[:5])
            ask_depth = sum(p * q * contract_value for p, q in sorted(asks)[:5])
            depth_usd = min(bid_depth, ask_depth)
        elif unit_currency == quote_currency:
            bid_depth = sum(q * contract_value for _, q in sorted(bids, reverse=True)[:5])
            ask_depth = sum(q * contract_value for _, q in sorted(asks)[:5])
            depth_usd = min(bid_depth, ask_depth)
        else:
            depth_usd = None
        snapshot = {
            "pair": pair, "best_bid": best_bid, "best_ask": best_ask,
            "mid_price": midpoint, "spread_bps": spread_bps,
            "bid_depth_usdt": depth_usd, "ask_depth_usdt": depth_usd,
            "min_side_depth_usdt": depth_usd, "depth_levels": 5,
            "timestamp": self._timestamp_ms(timestamp), "age_seconds": age,
            "depth": depth, "delta_symbol": symbol,
        }
        self._orderbook_cache[f"{pair}:{depth}"] = (time.monotonic(), snapshot)
        return dict(snapshot)
