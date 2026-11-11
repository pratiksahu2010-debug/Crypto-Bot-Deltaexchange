"""Candle buffer management — bounded rolling windows."""
from collections import deque
from typing import Dict, Deque, List, Optional
from math import isfinite
import numpy as np

from config import settings
from utils.logger import logger


class CandleBuffer:
    def __init__(self, max_len: int = 300):
        self.max_len = max_len
        self.candles: Deque[dict] = deque(maxlen=max_len)

    def update(self, candle: dict) -> bool:
        """Add or update latest candle. Returns True if a new closed candle was added."""
        required = ("time", "open", "high", "low", "close", "volume")
        if any(k not in candle or candle[k] is None for k in required):
            return False
        try:
            normalized = {
                "time": int(candle["time"]),
                "open": float(candle["open"]),
                "high": float(candle["high"]),
                "low": float(candle["low"]),
                "close": float(candle["close"]),
                "volume": float(candle["volume"]),
            }
        except (TypeError, ValueError, OverflowError):
            return False
        prices = (normalized["open"], normalized["high"], normalized["low"], normalized["close"])
        if (
            normalized["time"] < 0
            or not all(isfinite(value) for value in (*prices, normalized["volume"]))
            or any(value <= 0 for value in prices)
            or normalized["volume"] < 0
            or normalized["high"] < max(normalized["open"], normalized["close"], normalized["low"])
            or normalized["low"] > min(normalized["open"], normalized["close"], normalized["high"])
        ):
            return False
        candle = normalized
        if not self.candles:
            self.candles.append(dict(candle))
            return True
        last = self.candles[-1]
        if last.get("time") == candle.get("time"):
            self.candles[-1] = dict(candle)
            return False
        if candle["time"] > (last.get("time") or 0):
            self.candles.append(dict(candle))
            return True
        return False

    def to_arrays(self) -> Dict[str, np.ndarray]:
        if not self.candles:
            return {}
        return {
            "open": np.array([c["open"] for c in self.candles], dtype=np.float64),
            "high": np.array([c["high"] for c in self.candles], dtype=np.float64),
            "low": np.array([c["low"] for c in self.candles], dtype=np.float64),
            "close": np.array([c["close"] for c in self.candles], dtype=np.float64),
            "volume": np.array([c["volume"] for c in self.candles], dtype=np.float64),
            "time": np.array([c["time"] for c in self.candles], dtype=np.int64),
        }

    def last_close(self) -> Optional[float]:
        return self.candles[-1]["close"] if self.candles else None

    def __len__(self) -> int:
        return len(self.candles)


class CandleBuilder:
    def __init__(self):
        self.buffers: Dict[str, Dict[str, CandleBuffer]] = {}  # pair -> tf -> buffer

    def get_buffer(self, pair: str, tf: str) -> CandleBuffer:
        if pair not in self.buffers:
            self.buffers[pair] = {}
        if tf not in self.buffers[pair]:
            self.buffers[pair][tf] = CandleBuffer(settings.max_candles_buffer)
        return self.buffers[pair][tf]

    def seed(self, pair: str, tf: str, candles: List[dict]) -> None:
        buf = self.get_buffer(pair, tf)
        # Build and validate the replacement off to the side. A malformed or
        # empty REST response must never erase a previously usable live buffer.
        staged = CandleBuffer(buf.max_len)
        seen = {}
        for c in candles:
            try:
                seen[int(c["time"])] = dict(c)
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
        for ts in sorted(seen):
            staged.update(seen[ts])
        if not staged.candles:
            logger.warning(
                f"CANDLE_SEED_REJECTED pair={pair} tf={tf} "
                f"received={len(candles)} reason=no_valid_candles"
            )
            return
        # Merge the REST history with newer WebSocket updates. REST repairs
        # historical gaps, while a forming candle already received from WS
        # must not be discarded by a concurrent warm-up/backfill request.
        merged = {int(c["time"]): dict(c) for c in staged.candles}
        rest_latest_ts = max(merged)
        for existing in buf.candles:
            ts = int(existing["time"])
            if ts > rest_latest_ts:
                merged[ts] = dict(existing)
        replacement = CandleBuffer(buf.max_len)
        for ts in sorted(merged):
            replacement.update(merged[ts])
        buf.candles.clear()
        buf.candles.extend(replacement.candles)
        logger.debug(
            f"Seeded {pair} {tf} with {len(buf.candles)} valid candles "
            f"from {len(candles)} rows"
        )

    def on_candle(self, pair: str, tf: str, candle: dict) -> bool:
        return self.get_buffer(pair, tf).update(candle)

    def get_arrays(self, pair: str, tf: str) -> Dict[str, np.ndarray]:
        return self.get_buffer(pair, tf).to_arrays()

    def get_completed_arrays(
        self, pair: str, tf: str, now_ms: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """Return only candles whose timeframe has fully elapsed.

        Delta's WebSocket candle channel publishes the forming candle. The
        latest buffer row is therefore not automatically safe for indicators.
        """
        from utils.time_utils import parse_interval
        import time

        interval_ms = int(parse_interval(tf).total_seconds() * 1000)
        grace_ms = max(0, int(settings.candle_close_grace_seconds)) * 1000
        current_ms = int(now_ms if now_ms is not None else time.time() * 1000)
        arrays = self.get_arrays(pair, tf)
        if not arrays or not len(arrays.get("time", [])):
            return {}
        mask = arrays["time"] + interval_ms + grace_ms <= current_ms
        if not np.any(mask):
            return {}
        return {key: values[mask] for key, values in arrays.items()}

    def clear_pair(self, pair: str) -> None:
        self.buffers.pop(pair, None)