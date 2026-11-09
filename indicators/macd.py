"""MACD indicator."""
import numpy as np
from typing import Optional, Tuple
from .ema import ema_series


def macd(
    closes: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Optional[Tuple[float, float, float]]:
    """Return (macd_line, signal_line, histogram) or None."""
    try:
        values = np.asarray(closes, dtype=np.float64)
        fast_period, slow_period, signal_period = int(fast), int(slow), int(signal)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        values.ndim != 1 or fast_period != fast or slow_period != slow or signal_period != signal
        or fast_period < 1 or slow_period <= fast_period or signal_period < 1
        or len(values) < slow_period + signal_period - 1
        or not np.all(np.isfinite(values)) or np.any(values <= 0)
    ):
        return None
    fast_ema = ema_series(values, fast_period)
    slow_ema = ema_series(values, slow_period)
    if fast_ema is None or slow_ema is None:
        return None
    macd_line = fast_ema - slow_ema
    signal_line = ema_series(macd_line[slow_period - 1:], signal_period)
    if signal_line is None:
        return None
    macd_value = float(macd_line[-1])
    signal_value = float(signal_line[-1])
    histogram = macd_value - signal_value
    if not all(np.isfinite(x) for x in (macd_value, signal_value, histogram)):
        return None
    return macd_value, signal_value, histogram
