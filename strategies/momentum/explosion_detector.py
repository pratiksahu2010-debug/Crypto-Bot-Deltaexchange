"""Strict indicator-led Momentum Explosion qualification.

A valid alert requires directional agreement across momentum, participation,
trend and volatility. A single acceleration flag or a generic bandwidth uptick
cannot create an alert by itself.
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


def _directional_di(data, direction):
    plus_di, minus_di = _num(data, "plus_di"), _num(data, "minus_di")
    if plus_di is None or minus_di is None:
        return False
    return plus_di > minus_di if direction == "LONG" else minus_di > plus_di


def evaluate(indicators: dict) -> bool:
    """True only for a confirmed, directional volatility release."""
    if not isinstance(indicators, dict):
        return False
    direction = str(indicators.get("direction", "")).upper()
    if direction not in ("LONG", "SHORT"):
        return False

    rvol = _num(indicators, "rvol")
    expansion = _num(indicators, "atr_expansion")
    rsi = _num(indicators, "rsi")
    adx = _num(indicators, "adx")
    if None in (rvol, expansion, rsi, adx):
        return False
    if rvol < 1.25 or expansion < 1.08 or adx < 18.0:
        return False
    if not _directional_di(indicators, direction):
        return False

    rsi_ok = rsi >= 55.0 if direction == "LONG" else rsi <= 45.0
    # Both momentum horizons must agree; acceleration alone is not direction.
    directional_momentum = (
        _flag(indicators, "macd_direction_ok")
        and _flag(indicators, "roc_direction_ok")
    )
    structure_or_trend = (
        _flag(indicators, "ema_aligned")
        or _flag(indicators, "structure_break")
    )
    confirmed_release = (
        _flag(indicators, "squeeze_release")
        and _flag(indicators, "bw_expand")
    )
    return bool(
        rsi_ok and directional_momentum
        and structure_or_trend and confirmed_release
    )
