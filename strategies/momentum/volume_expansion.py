"""Relative-volume expansion with indicator direction; candle shape is ignored."""
import math


def evaluate(indicators):
    try:
        rvol = float(indicators.get("rvol"))
    except (TypeError, ValueError, OverflowError):
        return False
    if not math.isfinite(rvol) or rvol < 1.25:
        return False
    direction = str(indicators.get("direction", "")).upper()
    if direction not in ("LONG", "SHORT"):
        return False
    return bool(
        indicators.get("macd_direction_ok") is True
        or indicators.get("roc_direction_ok") is True
        or indicators.get("ema_aligned") is True
    )
