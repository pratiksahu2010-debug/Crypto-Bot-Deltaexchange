"""Relative Strength Index using Wilder smoothing."""
import numpy as np
from typing import Optional


def rsi(closes: np.ndarray, period: int = 14) -> Optional[float]:
    try:
        values = np.asarray(closes, dtype=np.float64)
        p = int(period)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        values.ndim != 1 or p != period or p < 1 or len(values) < p + 1
        or not np.all(np.isfinite(values)) or np.any(values <= 0)
    ):
        return None
    deltas = np.diff(values)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains[:p]))
    avg_loss = float(np.mean(losses[:p]))
    for i in range(p, len(gains)):
        avg_gain = (avg_gain * (p - 1) + gains[i]) / p
        avg_loss = (avg_loss * (p - 1) + losses[i]) / p
    if avg_gain == 0 and avg_loss == 0:
        return 50.0
    if avg_loss == 0:
        return 100.0
    if avg_gain == 0:
        return 0.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))
