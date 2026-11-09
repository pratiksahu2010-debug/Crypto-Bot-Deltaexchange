"""CoinDCX WebSocket client (Socket.IO based). Best-effort on Free tier."""
import asyncio
from typing import Callable, Dict, List, Optional, Set
from utils.logger import logger
from config import settings

try:
    import socketio
    HAS_SIO = True
except ImportError:
    HAS_SIO = False
    logger.warning("python-socketio not installed — WebSocket disabled")


class CoinDCXWebSocket:
    """
    Public market data via Socket.IO.
    Official endpoint: https://stream.coindcx.com
    Only socket.io-client ~2.x protocol is fully supported by CoinDCX.
    On Render Free the connection drops when the service sleeps.
    """

    def __init__(self):
        self.url = settings.coindcx_ws_url
        self.sio = None
        self.connected = False
        self.subscribed: Set[str] = set()
        self._on_candle: Optional[Callable] = None
        self._reconnect_count = 0
        self._running = False

    async def start(self, on_candle: Optional[Callable] = None) -> None:
        if not HAS_SIO or not settings.enable_websocket:
            logger.info("WebSocket disabled or socketio missing")
            return
        self._on_candle = on_candle
        self._running = True
        self.sio = socketio.AsyncClient(reconnection=True, reconnection_attempts=10, reconnection_delay=2)
        self._register_handlers()
        try:
            await self.sio.connect(self.url, transports=["websocket"], wait_timeout=15)
            self.connected = True
            logger.info("WebSocket connected")
        except Exception as e:
            logger.error(f"WebSocket connect failed: {e}")
            self.connected = False

    def _register_handlers(self) -> None:
        assert self.sio

        @self.sio.event
        async def connect():
            self.connected = True
            logger.info("WS event: connected")
            # re-subscribe
            for ch in list(self.subscribed):
                await self.sio.emit("join", {"channelName": ch})

        @self.sio.event
        async def disconnect():
            self.connected = False
            self._reconnect_count += 1
            logger.warning("WS event: disconnected")

        @self.sio.on("candlestick")
        async def on_candlestick(data):
            if self._on_candle and data:
                try:
                    await self._on_candle(data)
                except Exception as e:
                    logger.debug(f"candle callback error: {e}")

    async def subscribe_candles(self, pair: str, interval: str = "5m") -> None:
        if not self.sio or not self.connected:
            return
        channel = f"candlestick@{pair}_{interval}"
        if channel in self.subscribed:
            return
        try:
            await self.sio.emit("join", {"channelName": channel})
            self.subscribed.add(channel)
            logger.debug(f"Subscribed {channel}")
        except Exception as e:
            logger.warning(f"Subscribe failed {channel}: {e}")

    async def unsubscribe(self, pair: str, interval: str = "5m") -> None:
        if not self.sio:
            return
        channel = f"candlestick@{pair}_{interval}"
        try:
            await self.sio.emit("leave", {"channelName": channel})
            self.subscribed.discard(channel)
        except Exception:
            pass

    async def stop(self) -> None:
        self._running = False
        if self.sio and self.connected:
            try:
                await self.sio.disconnect()
            except Exception:
                pass
        self.connected = False
        logger.info("WebSocket stopped")

    @property
    def reconnect_count(self) -> int:
        return self._reconnect_count
