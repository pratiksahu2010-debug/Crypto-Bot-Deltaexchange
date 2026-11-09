"""Relative volume versus preceding completed bars (excludes current bar from baseline)."""
import numpy as np
from typing import Optional


def relative_volume(volumes: np.ndarray, period: int = 20) -> Optional[float]:
    """Return latest volume / mean of prior bars, or None for invalid data."""
    try:
        values = np.asarray(volumes, dtype=np.float64)
        if isinstance(period, (bool, np.bool_)):
            return None
        p = int(period)
    except (TypeError, ValueError, OverflowError):
        return None

    if (
        values.ndim != 1 or p != period or p < 1
        or len(values) < p + 1
        or not np.all(np.isfinite(values))
        or np.any(values < 0)
    ):
        return None

    baseline = values[-p - 1:-1]
    average = float(np.mean(baseline))
    if not np.isfinite(average) or average <= 0:
        return None
    result = float(values[-1] / average)
    return result if np.isfinite(result) else None
