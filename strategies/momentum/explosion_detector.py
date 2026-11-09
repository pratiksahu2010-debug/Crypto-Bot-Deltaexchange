"""Indicator-led Momentum Explosion and pre-ignition detection.

A squeeze identifies stored volatility; directional EMA/RSI/MACD/ROC state
selects the side; RVOL and ATR expansion measure participation and release.
Candle colour, body ratio, wick ratio and order-book/risk fields are not used.
"""
import math


def _num(data, key):
    try:
        value = float(data.get(key))
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _flag(data, key):
    value = data.get(key)
    return value is True or (type(value).__name__ == "bool_" and bool(value))


def evaluate(indicators: dict) -> bool:
    """True for a directional squeeze release / volatility ignition."""
    if not isinstance(indicators, dict):
        return False
    direction = str(indicators.get("direction", "")).upper()
    if direction not in ("LONG", "SHORT"):
        return False
    rvol = _num(indicators, "rvol")
    expansion = _num(indicators, "atr_expansion")
    rsi = _num(indicators, "rsi")
    if rvol is None or expansion is None or rsi is None:
        return False
    if rvol < 1.15 or expansion < 1.03:
        return False
    rsi_direction = rsi >= 52.0 if direction == "LONG" else rsi <= 48.0
    if not rsi_direction:
        return False
    directional_momentum = (
        _flag(indicators, "macd_direction_ok")
        or _flag(indicators, "macd_accel")
        or _flag(indicators, "roc_direction_ok")
    )
    trend_alignment = _flag(indicators, "ema_aligned")
    release = _flag(indicators, "squeeze_release") or _flag(indicators, "bw_expand")
    return bool(
        directional_momentum and
        release and
        (trend_alignment or _flag(indicators, "roc_direction_ok"))
    )
