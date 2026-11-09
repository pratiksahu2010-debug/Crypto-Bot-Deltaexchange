"""UTC-session or rolling volume-weighted average price."""
import numpy as np
from typing import Optional


def vwap(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    volume: np.ndarray, period: Optional[int] = 100,
) -> Optional[float]:
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    n = len(close)
    if (
        any(x.ndim != 1 for x in (high, low, close, volume))
        or n == 0 or not (len(high) == len(low) == len(volume) == n)
        or not all(np.all(np.isfinite(x)) for x in (high, low, close, volume))
        or np.any(high <= 0) or np.any(low <= 0) or np.any(close <= 0)
        or np.any(high < low) or np.any(close > high) or np.any(close < low)
        or np.any(volume < 0)
    ):
        return None
    if period is not None:
        try:
            p = int(period)
        except (TypeError, ValueError, OverflowError):
            return None
        if p != period or p < 1:
            return None
        high, low, close, volume = high[-p:], low[-p:], close[-p:], volume[-p:]
    total = float(np.sum(volume))
    if not np.isfinite(total) or total <= 0:
        return None
    typical = (high + low + close) / 3.0
    result = float(np.dot(typical, volume) / total)
    return result if np.isfinite(result) else None
