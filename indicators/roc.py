"""Rate of Change."""
import numpy as np
from typing import Optional


def roc(closes: np.ndarray, period: int = 10) -> Optional[float]:
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
    previous = float(values[-p - 1])
    return float((values[-1] - previous) / previous * 100.0)
