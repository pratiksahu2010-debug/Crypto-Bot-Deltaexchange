"""Bollinger Bands."""
import numpy as np
from typing import Optional, Tuple


def bollinger_bands(
    closes: np.ndarray,
    period: int = 20,
    std_dev: float = 2.0,
) -> Optional[Tuple[float, float, float]]:
    """Return (upper, middle, lower) or None for invalid/incomplete input."""
    try:
        values = np.asarray(closes, dtype=np.float64)
        p = int(period)
        deviation = float(std_dev)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        values.ndim != 1 or p != period or p < 2 or len(values) < p
        or not np.all(np.isfinite(values)) or np.any(values <= 0)
        or not np.isfinite(deviation) or deviation <= 0
    ):
        return None
    window = values[-p:]
    mid = float(np.mean(window))
    std = float(np.std(window, ddof=0))
    return mid + deviation * std, mid, mid - deviation * std


def bandwidth(upper: float, lower: float, mid: float) -> float:
    if not all(np.isfinite(x) for x in (upper, lower, mid)) or mid <= 0 or upper < lower:
        return 0.0
    return (upper - lower) / mid
