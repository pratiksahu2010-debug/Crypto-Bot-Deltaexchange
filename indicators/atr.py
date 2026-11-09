"""Average True Range using Wilder smoothing and validated OHLC input."""
import numpy as np
from typing import Optional


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Optional[float]:
    try:
        high = np.asarray(high, dtype=np.float64)
        low = np.asarray(low, dtype=np.float64)
        close = np.asarray(close, dtype=np.float64)
        p = int(period)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        high.ndim != 1 or low.ndim != 1 or close.ndim != 1
        or p != period or p < 1 or len(close) < p + 1
        or not (len(high) == len(low) == len(close))
        or not all(np.all(np.isfinite(x)) for x in (high, low, close))
        or np.any(high <= 0) or np.any(low <= 0) or np.any(close <= 0)
        or np.any(high < low) or np.any(close > high) or np.any(close < low)
    ):
        return None
    tr = np.empty(len(close), dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    value = float(np.mean(tr[1:p + 1]))
    for i in range(p + 1, len(tr)):
        value = (value * (p - 1) + tr[i]) / p
    return float(value) if np.isfinite(value) else None
