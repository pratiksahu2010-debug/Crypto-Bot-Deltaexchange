"""Optional CoinDCX Futures Socket.IO market-data stream.

REST remains authoritative for completed-candle decisions. This client only
feeds live, forming candles into CandleBuilder and safely degrades on disconnect.
"""
import asyncio
import math
import re
import time
from typing import Any, Dict, Iterable, Optional

import socketio

from config import settings
from utils.logger import logger
from utils.time_utils import parse_interval


class CoinDCXFuturesStream:
    def __init__(self, candle_builder, rest_client=None):
        self.candles = candle_builder
        self.rest = rest_client
        self.client = socketio.AsyncClient(
            reconnection=True,
            reconnection_attempts=0,
            reconnection_delay=1,
            reconnection_delay_max=30,
            logger=False,
            engineio_logger=False,
        )
        # Desired subscriptions survive transport disconnects. The subscribed
        # set is connection-local and is cleared whenever Socket.IO disconnects.
        self._channels = set()
        self._subscribed_channels = set()
        self._subscription_lock = asyncio.Lock()
        self._running = False
        self._connected = False
        self._last_event_ms: Optional[int] = None
        self._reconnect_count = 0
        self.on_candle_callback = None
        self._register_handlers()

    def set_candle_callback(self, callback) -> None:
        self.on_candle_callback = callback

    def _register_handlers(self) -> None:
        @self.client.event
        async def connect():
            self._connected = True
            await self._subscribe_channels()
            logger.info(
                f"COINDCX_WS_CONNECTED price_cache_pairs={len(self.rest._ws_prices) if self.rest else 0} "
                f"desired_channels={len(self._channels)} "
                f"subscribed_channels={len(self._subscribed_channels)}"
            )

        @self.client.event
        async def disconnect():
            self._connected = False
            if self._running:
                self._reconnect_count += 1
            # The server drops all channel memberships with the transport.
            # Keep desired channels so the next connect event can restore them.
            self._subscribed_channels.clear()
            logger.warning("COINDCX_WS_DISCONNECTED subscriptions_cleared=true REST_fallback=true")

        @self.client.on("candlestick")
        async def on_candlestick(message):
            self._handle_candlestick(message)

        @self.client.on("currentPrices@futures#update")
        async def on_prices(message):
            self._handle_prices(message)

    async def _subscribe_channels(self) -> int:
        """Join every desired channel once on the current Socket.IO connection."""
        if not self._running or not self.client.connected:
            return 0
        joined = 0
        failed = []
        async with self._subscription_lock:
            pending = sorted(self._channels - self._subscribed_channels)
            for channel in pending:
                try:
                    await self.client.emit("join", {"channelName": channel})
                except Exception as exc:
                    failed.append(channel)
                    logger.warning(f"COINDCX_WS_SUBSCRIBE_FAILED channel={channel} error={exc}")
                    continue
                self._subscribed_channels.add(channel)
                joined += 1
        logger.info(
            f"COINDCX_WS_SUBSCRIPTION_STATE desired={len(self._channels)} "
            f"active={len(self._subscribed_channels)} joined_now={joined} failed={len(failed)}"
        )
        return joined

    def _handle_prices(self, message: Any) -> None:
        """Consume the documented currentPrices@futures#update event safely."""
        if self.rest is None:
            return
        payload = message.get("data", message) if isinstance(message, dict) else message
        if isinstance(payload, dict):
            prices = payload.get("prices") or payload.get("data")
            event_ts = payload.get("ts") or payload.get("timestamp")
            if isinstance(prices, dict):
                accepted = self.rest.set_ws_prices(prices, event_ts)
                self._last_event_ms = int(__import__("time").time() * 1000) if accepted else self._last_event_ms
                logger.info(
                    f"COINDCX_WS_PRICE_EVENT markets={len(prices)} "
                    f"accepted_ltp_updates={accepted} cache_pairs={len(self.rest._ws_prices)} "
                    f"event_ts={'present' if event_ts else 'missing'}"
                )
                return
            if isinstance(prices, list):
                prices = prices
            else:
                return
        else:
            prices = payload if isinstance(payload, list) else []
            event_ts = None
        normalized = {}
        for row in prices:
            if not isinstance(row, dict):
                continue
            market = row.get("market") or row.get("pair") or row.get("symbol") or row.get("mkt")
            if market:
                normalized[str(market)] = row
        if normalized:
            accepted = self.rest.set_ws_prices(normalized, event_ts)
            self._last_event_ms = int(__import__("time").time() * 1000) if accepted else self._last_event_ms
            logger.info(
                f"COINDCX_WS_PRICE_EVENT markets={len(normalized)} "
                f"accepted_ltp_updates={accepted} cache_pairs={len(self.rest._ws_prices)}"
            )

    def _handle_candlestick(self, message: Any) -> None:
        if not isinstance(message, dict):
            return
        payload = message.get("data", message)
        outer_channel = message.get("channel") or message.get("channelName")
        rows = payload if isinstance(payload, list) else [payload]
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                raw_pair = row.get("pair") or row.get("market") or row.get("symbol")
                if not raw_pair:
                    channel = str(row.get("channel") or outer_channel or "")
                    match = re.search(r"((?:B|I|HB|KC)-[A-Z0-9]+_[A-Z0-9]+)", channel.upper())
                    raw_pair = match.group(1) if match else ""
                pair = str(raw_pair or "").strip().upper()
                if not pair or not pair.startswith(("B-", "I-", "HB-", "KC-")):
                    continue
                raw_time = row.get("open_time", row.get("t", row.get("time")))
                ts = int(float(raw_time))
                if ts < 10**12:
                    ts *= 1000
                candle = {
                    "time": ts,
                    "open": float(row.get("open", row.get("o"))),
                    "high": float(row.get("high", row.get("h"))),
                    "low": float(row.get("low", row.get("l"))),
                    "close": float(row.get("close", row.get("c"))),
                    "volume": float(row.get("volume", row.get("v", 0)) or 0),
                }
                duration = str(row.get("duration") or row.get("i") or "").strip()
                # Uppercase M denotes a calendar month on exchange feeds, not
                # the lowercase m minute unit supported by parse_interval.
                # Never normalize a monthly candle into a one-minute candle.
                if duration.endswith("M"):
                    continue
                interval = duration if duration else None
                if interval not in ("1m", "5m", "15m", "30m", "1h", "4h", "8h", "1d", "3d", "1w"):
                    channel = str(row.get("channel") or outer_channel or "")
                    # Keep the unit case-sensitive: M (month) is not m (minute).
                    match = re.search(r"_(\d+[mhdw])-futures", channel)
                    interval = match.group(1) if match else None
                if interval:
                    interval = interval.strip().lower()
                    # WebSocket candles are forming-candle context only. Reject
                    # malformed, misaligned, stale, or implausibly future events
                    # before they can mutate the shared buffer or watchlist.
                    interval_ms = int(parse_interval(interval).total_seconds() * 1000)
                    now_ms = int(time.time() * 1000)
                    if interval_ms <= 0 or ts % interval_ms != 0:
                        continue
                    # This stream is forming-candle context only. Once the
                    # interval has closed, REST owns the final candle; a delayed
                    # websocket update must not overwrite it or create stale watchlist context.
                    if (
                        ts > now_ms + 6_000
                        or ts + interval_ms <= now_ms
                        or now_ms - ts > interval_ms + 30_000
                    ):
                        logger.debug(
                            f"WS_CANDLE_TIMESTAMP_REJECTED pair={pair} tf={interval} "
                            f"open_ms={ts} now_ms={now_ms}"
                        )
                        continue
                    prices = (candle["open"], candle["high"], candle["low"], candle["close"])
                    if (
                        not all(math.isfinite(value) and value > 0 for value in prices)
                        or not math.isfinite(candle["volume"])
                        or candle["volume"] < 0
                        or candle["high"] < max(candle["open"], candle["close"], candle["low"])
                        or candle["low"] > min(candle["open"], candle["close"], candle["high"])
                    ):
                        continue
                    # CandleBuffer rejects older timestamps, but the callback
                    # must be fenced too: otherwise a stale event can still
                    # update intrabar watchlists after the buffer rejects it.
                    get_buffer = getattr(self.candles, "get_buffer", None)
                    if callable(get_buffer):
                        buffer = get_buffer(pair, interval)
                        buffered = getattr(buffer, "candles", None)
                        if buffered and ts < int(buffered[-1].get("time", 0)):
                            logger.debug(
                                f"WS_CANDLE_OUT_OF_ORDER_REJECTED pair={pair} tf={interval} "
                                f"event_ms={ts} latest_ms={buffered[-1].get('time')}"
                            )
                            continue
                    self.candles.on_candle(pair, interval, candle)
                    if self.on_candle_callback is not None:
                        try:
                            self.on_candle_callback(pair, interval, candle)
                        except Exception as exc:
                            logger.debug(f"WS_CANDLE_CALLBACK_SKIPPED {pair}: {exc}")
                    # Track local receive time for feed liveness, not an exchange
                    # event timestamp whose clock may be skewed or malformed.
                    self._last_event_ms = now_ms
            except (TypeError, ValueError, OverflowError):
                continue

    async def start(self, pairs: Iterable[str]) -> bool:
        if not settings.enable_websocket:
            logger.info("COINDCX_WS_DISABLED ENABLE_WEBSOCKET=false REST_fallback=true")
            return False
        self._running = True
        # Build the complete desired set before connecting. The connect handler
        # performs the joins on both the initial connection and every reconnect.
        self._channels = {"currentPrices@futures@rt"}
        # Subscribe to the exact timeframes consumed by the configured engines.
        # A fixed 5m/15m subscription silently starves valid custom configurations.
        intervals = tuple(dict.fromkeys((settings.candle_primary_tf, settings.candle_confirm_tf)))
        for pair in sorted(set(pairs)):
            for interval in intervals:
                self._channels.add(f"{pair}_{interval}-futures")
        try:
            await self.client.connect(
                settings.coindcx_ws_url,
                transports=["websocket"],
            )
            # Some test doubles and alternate transports do not dispatch the
            # connect callback; this idempotent call also verifies initial joins.
            await self._subscribe_channels()
            logger.info(
                f"COINDCX_WS_SUBSCRIBE_SENT desired={len(self._channels)} "
                f"active={len(self._subscribed_channels)}"
            )
            return True
        except Exception as exc:
            logger.warning(f"COINDCX_WS_START_FAILED REST_fallback=true error={exc}")
            await self.stop()
            return False

    async def stop(self) -> None:
        self._running = False
        if self.client.connected:
            for channel in tuple(self._subscribed_channels):
                try:
                    await self.client.emit("leave", {"channelName": channel})
                except Exception:
                    pass
            await self.client.disconnect()
        self._channels.clear()
        self._subscribed_channels.clear()
        self._connected = False

    @property
    def status(self) -> str:
        return "connected" if self._connected else ("reconnecting" if self._running else "stopped")

    @property
    def reconnect_count(self) -> int:
        return self._reconnect_count

    @property
    def price_feed_age_seconds(self) -> Optional[float]:
        if self.rest is None or self.rest._ws_last_update_ms is None:
            return None
        return max(0.0, (__import__("time").time() * 1000 - self.rest._ws_last_update_ms) / 1000.0)
