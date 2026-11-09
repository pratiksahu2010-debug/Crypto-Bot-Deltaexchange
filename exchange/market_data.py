"""High-level market data facade."""
from typing import Dict, List, Optional
from exchange.delta_rest import DeltaExchangeRest
from exchange.candle_builder import CandleBuilder
from utils.logger import logger


class MarketData:
    def __init__(self, rest: DeltaExchangeRest, builder: CandleBuilder):
        self.rest = rest
        self.builder = builder

    async def get_ohlcv(self, pair: str, interval: str = "5m", limit: int = 100) -> List[dict]:
        return await self.rest.get_candles(pair, interval, limit=limit)

    async def ensure_buffer(self, pair: str, interval: str, min_candles: int = 50) -> bool:
        buf = self.builder.get_buffer(pair, interval)
        if len(buf) >= min_candles:
            return True
        data = await self.rest.get_candles(pair, interval, limit=max(min_candles + 20, 100))
        if data:
            self.builder.seed(pair, interval, data)
            return len(self.builder.get_buffer(pair, interval)) >= min_candles
        return False

    def last_price(self, pair: str, interval: str = "5m") -> Optional[float]:
        return self.builder.get_buffer(pair, interval).last_close()
