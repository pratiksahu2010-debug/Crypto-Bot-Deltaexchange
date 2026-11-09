"""Bollinger/Keltner squeeze release and pre-release state."""
import math


def _number(indicators, key):
    try:
        value = float(indicators.get(key))
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _flag(indicators, key):
    value = indicators.get(key)
    return value is True or (type(value).__name__ == "bool_" and bool(value))


def evaluate(indicators):
    rvol = _number(indicators, "rvol")
    return bool(
        rvol is not None and rvol >= 1.10
        and _flag(indicators, "squeeze_release")
        and _flag(indicators, "bw_expand")
        and (_flag(indicators, "roc_direction_ok") or _flag(indicators, "macd_direction_ok"))
    )


def pre_explosion(indicators):
    rvol = _number(indicators, "rvol")
    expansion = _number(indicators, "atr_expansion")
    return bool(
        rvol is not None and rvol >= 1.10
        and expansion is not None and expansion >= 1.0
        and _flag(indicators, "compression_active")
    )
