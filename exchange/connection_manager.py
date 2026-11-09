"""Connection health and reconnect orchestration."""
from typing import Optional
from utils.logger import logger
from exchange.delta_ws import DeltaFuturesStream
from exchange.delta_rest import DeltaExchangeRest


class ConnectionManager:
    def __init__(self, rest: DeltaExchangeRest, ws: Optional[DeltaFuturesStream] = None):
        self.rest = rest
        self.ws = ws
        self._healthy = True

    async def check_rest(self) -> bool:
        try:
            data = await self.rest.get_ticker()
            self._healthy = bool(data)
            return self._healthy
        except Exception as e:
            logger.warning(f"REST health check failed: {e}")
            self._healthy = False
            return False

    @property
    def ws_connected(self) -> bool:
        return bool(self.ws and self.ws.connected)

    @property
    def status(self) -> dict:
        return {
            "rest_healthy": self._healthy,
            "ws_connected": self.ws_connected,
            "ws_reconnects": self.ws.reconnect_count if self.ws else 0,
        }
