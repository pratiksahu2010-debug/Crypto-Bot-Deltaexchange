"""Confirmed swing high / low support and resistance."""
import numpy as np
from typing import List, Tuple


def find_swing_levels(
    high: np.ndarray,
    low: np.ndarray,
    lookback: int = 5,
    max_levels: int = 3,
) -> Tuple[List[float], List[float]]:
    """Return confirmed swing levels, nearest to the latest candle midpoint first."""
    try:
        highs = np.asarray(high, dtype=np.float64)
        lows = np.asarray(low, dtype=np.float64)
        left_right = int(lookback)
        limit = int(max_levels)
    except (TypeError, ValueError, OverflowError):
        return [], []
    if (
        highs.ndim != 1 or lows.ndim != 1 or len(highs) != len(lows)
        or left_right != lookback or limit != max_levels
        or left_right < 1 or limit < 1
        or len(highs) < left_right * 2 + 1
        or not np.all(np.isfinite(highs)) or not np.all(np.isfinite(lows))
        or np.any(highs <= 0) or np.any(lows <= 0) or np.any(highs < lows)
    ):
        return [], []
    resistances = []
    supports = []
    for i in range(left_right, len(highs) - left_right):
        high_window = highs[i - left_right:i + left_right + 1]
        low_window = lows[i - left_right:i + left_right + 1]
        if highs[i] == np.max(high_window):
            resistances.append(float(highs[i]))
        if lows[i] == np.min(low_window):
            supports.append(float(lows[i]))
    anchor = float((highs[-1] + lows[-1]) / 2.0)
    resistances = sorted(set(resistances), key=lambda value: abs(value - anchor))[:limit]
    supports = sorted(set(supports), key=lambda value: abs(value - anchor))[:limit]
    return resistances, supports
