"""Strict directional Momentum continuation confirmation."""
import math


def _number(indicators, key):
    try:
        value = float(indicators.get(key))
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def evaluate(indicators):
    rvol = _number(indicators, "rvol")
    expansion = _number(indicators, "atr_expansion")
    rsi = _number(indicators, "rsi")
    adx = _number(indicators, "adx")
    plus_di = _number(indicators, "plus_di")
    minus_di = _number(indicators, "minus_di")
    direction = str(indicators.get("direction", "")).upper()
    if direction not in ("LONG", "SHORT") or None in (
        rvol, expansion, rsi, adx, plus_di, minus_di
    ):
        return False
    if rvol < 1.25 or expansion < 1.08 or adx < 20.0:
        return False
    directional = (
        plus_di > minus_di if direction == "LONG" else minus_di > plus_di
    )
    rsi_ok = rsi >= 55.0 if direction == "LONG" else rsi <= 45.0
    return bool(
        directional and rsi_ok
        and indicators.get("macd_direction_ok") is True
        and indicators.get("roc_direction_ok") is True
        and indicators.get("ema_aligned") is True
    )
