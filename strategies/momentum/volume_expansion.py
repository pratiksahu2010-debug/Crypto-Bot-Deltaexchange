"""High-participation expansion with independent directional confirmation."""
import math


def evaluate(indicators):
    try:
        rvol = float(indicators.get("rvol"))
        expansion = float(indicators.get("atr_expansion"))
        adx = float(indicators.get("adx"))
        rsi = float(indicators.get("rsi"))
        plus_di = float(indicators.get("plus_di"))
        minus_di = float(indicators.get("minus_di"))
    except (TypeError, ValueError, OverflowError):
        return False
    if not all(math.isfinite(x) for x in (rvol, expansion, adx, rsi, plus_di, minus_di)):
        return False
    if rvol < 1.50 or expansion < 1.08 or adx < 17.0:
        return False
    direction = str(indicators.get("direction", "")).upper()
    if direction not in ("LONG", "SHORT"):
        return False
    di_ok = plus_di > minus_di if direction == "LONG" else minus_di > plus_di
    rsi_ok = rsi >= 55.0 if direction == "LONG" else rsi <= 45.0
    return bool(
        di_ok and rsi_ok
        and indicators.get("macd_direction_ok") is True
        and indicators.get("roc_direction_ok") is True
        and indicators.get("ema_aligned") is True
    )
