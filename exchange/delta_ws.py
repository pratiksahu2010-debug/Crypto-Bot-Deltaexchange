"""Delta Exchange India public WebSocket feed.

Uses the current public-socket endpoint and native candlestick_<resolution>
channels. REST remains authoritative for closed candles; WS is used only for
forming-candle telemetry and fresh last-traded-price cache updates.
"""
import asyncio
import json
import math
import time
from typing import Iterable, Optional

import websockets

from config import settings
from utils.logger import logger
from utils.time_utils import parse_interval


class DeltaFuturesStream:
    def __init__(self, candle_builder, rest_client=None):
        self.candles = candle_builder
        self.rest = rest_client
        self._connection = None
        self._runner: Optional[asyncio.Task] = None
        self._running = False
        self._connected = False
        self._connected_event = asyncio.Event()
        self._channels = {}
        self._reconnect_count = 0
        self._reconnect_attempts = 0
        self._last_event_ms = None
        self._last_heartbeat_ms = None
        self._subscription_error = None
        self._subscribed_channels = set()
        # Last accepted forming-candle update per canonical symbol and timeframe.
        self._last_symbol_update_ms = {}
        self.on_candle_callback = None

    def set_candle_callback(self, callback):
        self.on_candle_callback = callback

    @property
    def connected(self):
        return self._connected

    @property
    def status(self):
        return "connected" if self._connected else ("reconnecting" if self._running else "stopped")

    @property
    def reconnect_count(self):
        return self._reconnect_count

    @property
    def symbol_feed_age_seconds(self):
        now_ms = int(time.time() * 1000)
        return {
            pair: {tf: round(max(0.0, (now_ms - stamp) / 1000.0), 1)
                   for tf, stamp in values.items()}
            for pair, values in self._last_symbol_update_ms.items()
        }

    @staticmethod
    def reconnect_delay(attempt: int) -> float:
        """Exponential reconnect backoff, capped to keep recovery bounded."""
        return float(min(30, 2 ** max(0, min(int(attempt) - 1, 5))))

    @property
    def price_feed_age_seconds(self):
        # Preserve the status field's meaning: age of accepted LTP updates,
        # not heartbeat/candlestick transport activity.
        stamp = getattr(self.rest, "_ws_last_update_ms", None) if self.rest else None
        if stamp is None:
            return None
        return max(0.0, (time.time() * 1000 - stamp) / 1000.0)

    async def start(self, pairs: Iterable[str]) -> bool:
        if not settings.enable_websocket:
            logger.info("DELTA_WS_DISABLED ENABLE_WEBSOCKET=false REST_fallback=true")
            return False
        if self.rest is None:
            logger.warning("DELTA_WS_NOT_STARTED reason=missing_rest_adapter")
            return False
        symbols = []
        for pair in sorted(set(pairs)):
            try:
                symbols.append(self.rest.delta_symbol_for_pair(pair))
            except KeyError:
                logger.warning(f"DELTA_WS_SYMBOL_SKIPPED pair={pair} reason=no_live_product_mapping")
        if not symbols:
            logger.warning("DELTA_WS_NOT_STARTED reason=no_mapped_products")
            return False
        intervals = tuple(dict.fromkeys((settings.candle_primary_tf, settings.candle_confirm_tf)))
        self._channels = {
            f"candlestick_{interval}": symbols for interval in intervals
        }
        self._running = True
        self._connected_event.clear()
        self._runner = asyncio.create_task(self._run(), name="delta-public-websocket")
        try:
            await asyncio.wait_for(self._connected_event.wait(), timeout=5.0)
            return True
        except asyncio.TimeoutError:
            # Keep the reconnect loop alive, but report REST fallback to startup
            # so live intrabar mode is not incorrectly marked as active.
            logger.warning("DELTA_WS_CONNECT_TIMEOUT REST_fallback=true reconnect_loop=active")
            return False

    async def _run(self):
        first_attempt = True
        while self._running:
            try:
                async with websockets.connect(
                    settings.delta_ws_url,
                    open_timeout=15,
                    close_timeout=5,
                    ping_interval=20,
                    ping_timeout=10,
                    max_size=2**20,
                ) as connection:
                    self._connection = connection
                    self._connected = False
                    self._connected_event.clear()
                    self._last_heartbeat_ms = None
                    self._subscription_error = None
                    self._subscribed_channels = set()
                    connection_started_ms = int(time.time() * 1000)
                    connection_counted = False
                    await connection.send(json.dumps({"type": "enable_heartbeat"}))
                    channels = [
                        {"name": name, "symbols": symbols}
                        for name, symbols in self._channels.items()
                    ]
                    await connection.send(json.dumps({
                        "type": "subscribe",
                        "payload": {"channels": channels},
                    }))
                    # The server must acknowledge each requested channel/symbol
                    # before this connection is considered subscribed/ready.
                    logger.info(
                        f"DELTA_WS_SUBSCRIBE_SENT symbols={len(next(iter(self._channels.values()), []))} "
                        f"channels={','.join(self._channels)}"
                    )
                    while self._running:
                        heartbeat_reference_ms = (
                            self._last_heartbeat_ms
                            if self._last_heartbeat_ms is not None
                            else connection_started_ms
                        )
                        heartbeat_age_seconds = (
                            int(time.time() * 1000) - heartbeat_reference_ms
                        ) / 1000.0
                        remaining_heartbeat_seconds = max(
                            0.0, 35.0 - heartbeat_age_seconds
                        )
                        try:
                            raw = await asyncio.wait_for(
                                connection.recv(), timeout=remaining_heartbeat_seconds
                            )
                        except asyncio.TimeoutError as exc:
                            raise ConnectionError("heartbeat_timeout_35s") from exc
                        try:
                            message = json.loads(raw)
                        except (TypeError, ValueError):
                            continue
                        self._handle_message(message)
                        if self._subscription_error:
                            raise ConnectionError(self._subscription_error)
                        if self._connected and not connection_counted:
                            if not first_attempt:
                                self._reconnect_count += 1
                            first_attempt = False
                            connection_counted = True
                            logger.info(
                                f"DELTA_WS_CONNECTED symbols={len(next(iter(self._channels.values()), []))} "
                                f"channels={','.join(self._channels)}"
                            )
                        heartbeat_ms = self._last_heartbeat_ms
                        heartbeat_age_ms = (
                            int(time.time() * 1000) - heartbeat_ms
                            if heartbeat_ms is not None
                            else int(time.time() * 1000) - connection_started_ms
                        )
                        if heartbeat_age_ms > 35_000:
                            raise ConnectionError("heartbeat_timeout_35s")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._running:
                    logger.warning(f"DELTA_WS_DISCONNECTED REST_fallback=true error={exc}")
            finally:
                self._connected = False
                self._connected_event.clear()
                self._connection = None
            if self._running:
                self._reconnect_attempts += 1
                delay = self.reconnect_delay(self._reconnect_attempts)
                logger.warning(
                    "DELTA_WS_RECONNECT_SCHEDULED attempt=%s delay_seconds=%.0f",
                    self._reconnect_attempts, delay,
                )
                await asyncio.sleep(delay)

    def _handle_message(self, message):
        if not isinstance(message, dict):
            return
        kind = str(message.get("type") or "")
        if kind == "heartbeat":
            self._last_event_ms = int(time.time() * 1000)
            self._last_heartbeat_ms = self._last_event_ms
            return
        if kind == "subscriptions":
            acknowledged = {}
            errors = []
            for channel in message.get("channels", []) or []:
                if not isinstance(channel, dict):
                    continue
                name = str(channel.get("name") or "")
                if channel.get("error"):
                    errors.append(f"{name}:{channel.get('error')}")
                    continue
                acknowledged[name] = set(str(x).upper() for x in channel.get("symbols", []) or [])
            missing = []
            for name, requested in self._channels.items():
                received = acknowledged.get(name, set())
                absent = set(str(x).upper() for x in requested) - received
                if absent:
                    missing.append(f"{name}:{','.join(sorted(absent))}")
            if errors or missing:
                self._subscription_error = (
                    "subscription_ack_incomplete errors="
                    + ",".join(errors) + " missing=" + ",".join(missing)
                )
                logger.error("DELTA_WS_SUBSCRIPTION_REJECTED %s", self._subscription_error)
                return
            self._subscribed_channels = set(acknowledged)
            self._connected = True
            self._connected_event.set()
            return
        if kind.startswith("candlestick_"):
            self._handle_candlestick(message)

    def _handle_candlestick(self, message):
        try:
            raw_symbol = str(message.get("sy") or message.get("symbol") or "").upper()
            pair = self.rest.canonical_pair_for_symbol(raw_symbol) if self.rest else ""
            if not pair:
                return
            interval = str(message.get("res") or message.get("type", "").removeprefix("candlestick_"))
            if interval not in ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"):
                return
            interval_ms = int(parse_interval(interval).total_seconds() * 1000)
            raw_ts = message.get("ts")
            stamp = self.rest._timestamp_ms(raw_ts)
            if stamp is None:
                return
            # Delta's compact feed timestamp is microseconds. It may represent
            # the candle update time, so floor it to the resolution boundary.
            candle_open_ms = stamp // interval_ms * interval_ms
            now_ms = int(time.time() * 1000)
            if (
                candle_open_ms > now_ms + 6_000
                or candle_open_ms + interval_ms <= now_ms
                or now_ms - candle_open_ms > interval_ms + 30_000
            ):
                return
            candle = {
                "time": candle_open_ms,
                "open": float(message.get("o")),
                "high": float(message.get("h")),
                "low": float(message.get("l")),
                "close": float(message.get("c")),
                "volume": float(message.get("v") or 0),
            }
            values = [candle[k] for k in ("open", "high", "low", "close", "volume")]
            if (
                not all(math.isfinite(v) for v in values)
                or min(values[:4]) <= 0 or values[4] < 0
                or candle["high"] < max(candle["open"], candle["close"], candle["low"])
                or candle["low"] > min(candle["open"], candle["close"], candle["high"])
            ):
                return
            get_buffer = getattr(self.candles, "get_buffer", None)
            if callable(get_buffer):
                buffer = get_buffer(pair, interval)
                buffered = getattr(buffer, "candles", None)
                if buffered and candle_open_ms < int(buffered[-1].get("time", 0)):
                    return
            self.candles.on_candle(pair, interval, candle)
            self._last_symbol_update_ms.setdefault(pair, {})[interval] = now_ms
            # A successful market update resets the consecutive reconnect backoff.
            self._reconnect_attempts = 0
            # A traded-price candle close is LTP. Do not substitute mark_price.
            self.rest.set_ws_prices(
                {raw_symbol: {"last_price": candle["close"], "timestamp": raw_ts}},
                raw_ts,
            )
            if self.on_candle_callback is not None:
                try:
                    self.on_candle_callback(pair, interval, candle)
                except Exception as exc:
                    logger.debug(f"DELTA_WS_CANDLE_CALLBACK_SKIPPED {pair}: {exc}")
            self._last_event_ms = now_ms
        except (TypeError, ValueError, OverflowError):
            return

    async def stop(self):
        self._running = False
        runner = self._runner
        if runner and not runner.done():
            runner.cancel()
            try:
                await runner
            except asyncio.CancelledError:
                pass
        self._runner = None
        self._connected = False
        self._connected_event.clear()
        self._connection = None
        self._channels.clear()
        logger.info("DELTA_WS_STOPPED")
