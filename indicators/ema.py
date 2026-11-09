"""Exponential Moving Average with finite-input and warm-up validation."""
import numpy as np
from typing import Optional


def ema(data: np.ndarray, period: int) -> Optional[float]:
    """Return the latest EMA value or None when input is not usable."""
    series = ema_series(data, period)
    return float(series[-1]) if series is not None else None


def ema_series(data: np.ndarray, period: int) -> Optional[np.ndarray]:
    try:
        values = np.asarray(data, dtype=np.float64)
        p = int(period)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        values.ndim != 1 or p != period or p < 1 or len(values) < p
        or not np.all(np.isfinite(values))
    ):
        return None
    alpha = 2.0 / (p + 1)
    out = np.empty(len(values), dtype=np.float64)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    # SMA seed avoids the first-value bias during EMA warm-up.
    out[p - 1] = np.mean(values[:p])
    for i in range(p, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out
