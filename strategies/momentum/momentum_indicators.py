"""Directional momentum confirmation using MACD, ROC and EMA alignment."""
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
    expansion = _number(indicators, "atr_expansion")
    rsi = _number(indicators, "rsi")
    direction = str(indicators.get("direction", "")).upper()
    if direction not in ("LONG", "SHORT") or None in (rvol, expansion, rsi):
        return False
    if rvol < 1.10 or expansion < 1.02:
        return False
    rsi_ok = rsi >= 50.0 if direction == "LONG" else rsi <= 50.0
    return bool(
        rsi_ok and
        (_flag(indicators, "macd_direction_ok") or _flag(indicators, "roc_direction_ok")) and
        (_flag(indicators, "ema_aligned") or _flag(indicators, "macd_accel"))
    )
