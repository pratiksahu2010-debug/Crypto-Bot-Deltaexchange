"""ATR expansion is a volatility state, not a directional signal by itself."""


def evaluate(indicators):
    import math
    try:
        ratio = float(indicators.get("atr_expansion"))
        adx = float(indicators.get("adx"))
        plus_di = float(indicators.get("plus_di"))
        minus_di = float(indicators.get("minus_di"))
    except (TypeError, ValueError, OverflowError):
        return False
    if not all(math.isfinite(x) for x in (ratio, adx, plus_di, minus_di)):
        return False
    if ratio < 1.08 or adx < 20.0:
        return False
    direction = str(indicators.get("direction", "")).upper()
    if direction not in ("LONG", "SHORT"):
        return False
    di_ok = plus_di > minus_di if direction == "LONG" else minus_di > plus_di
    return bool(
        di_ok
        and indicators.get("roc_direction_ok") is True
        and indicators.get("macd_direction_ok") is True
        and indicators.get("ema_aligned") is True
    )
