"""Keltner Channels."""
import math
import numpy as np
from typing import Optional, Tuple
from .ema import ema
from .atr import atr


def keltner_channels(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 20,
    atr_mult: float = 1.5,
) -> Optional[Tuple[float, float, float]]:
    """Return (upper, middle, lower) or None for invalid/incomplete input."""
    try:
        multiplier = float(atr_mult)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(multiplier) or multiplier <= 0:
        return None
    mid = ema(close, period)
    atr_value = atr(high, low, close, period)
    if mid is None or atr_value is None:
        return None
    upper = mid + multiplier * atr_value
    lower = mid - multiplier * atr_value
    if not all(math.isfinite(x) for x in (upper, mid, lower)):
        return None
    return float(upper), float(mid), float(lower)
