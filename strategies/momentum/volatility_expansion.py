"""ATR expansion is a volatility state, not a directional signal by itself."""


def evaluate(indicators):
    import math
    try:
        ratio = float(indicators.get("atr_expansion"))
    except (TypeError, ValueError, OverflowError):
        return False
    if not math.isfinite(ratio) or ratio < 1.08:
        return False
    direction = str(indicators.get("direction", "")).upper()
    if direction not in ("LONG", "SHORT"):
        return False
    roc_ok = indicators.get("roc_direction_ok") is True
    macd_ok = indicators.get("macd_direction_ok") is True
    ema_ok = indicators.get("ema_aligned") is True
    return bool((roc_ok or macd_ok) and (ema_ok or roc_ok))
