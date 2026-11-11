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
    """A squeeze release is alertable only with multi-indicator direction."""
    direction = str(indicators.get("direction", "")).upper()
    if direction not in ("LONG", "SHORT"):
        return False
    rvol = _number(indicators, "rvol")
    expansion = _number(indicators, "atr_expansion")
    adx = _number(indicators, "adx")
    rsi = _number(indicators, "rsi")
    plus_di = _number(indicators, "plus_di")
    minus_di = _number(indicators, "minus_di")
    if None in (rvol, expansion, adx, rsi, plus_di, minus_di):
        return False
    if rvol < 1.25 or expansion < 1.05 or adx < 18.0:
        return False
    di_ok = plus_di > minus_di if direction == "LONG" else minus_di > plus_di
    rsi_ok = rsi >= 55.0 if direction == "LONG" else rsi <= 45.0
    return bool(
        di_ok and rsi_ok
        and _flag(indicators, "squeeze_release")
        and _flag(indicators, "bw_expand")
        and _flag(indicators, "roc_direction_ok")
        and _flag(indicators, "macd_direction_ok")
        and (_flag(indicators, "ema_aligned") or _flag(indicators, "structure_break"))
    )


def pre_explosion(indicators):
    """Compression is watchlist state only; alert qualification is separate."""
    rvol = _number(indicators, "rvol")
    expansion = _number(indicators, "atr_expansion")
    return bool(
        rvol is not None and rvol >= 1.10
        and expansion is not None and expansion >= 1.0
        and _flag(indicators, "compression_active")
    )
