"""Momentum regime dispatcher: pre-ignition, release and continuation."""
import math
from typing import Dict, Any
from . import explosion_detector, squeeze_detector, volatility_expansion, volume_expansion, momentum_indicators


def _number(indicators: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(indicators.get(key, default))
    except (TypeError, ValueError, OverflowError):
        return default
    return value if math.isfinite(value) else default


def _flag(indicators: Dict[str, Any], key: str) -> bool:
    value = indicators.get(key)
    return value is True or (type(value).__name__ == "bool_" and bool(value))


def diagnose_momentum_state(indicators: Dict[str, Any]) -> Dict[str, bool]:
    return {
        "EXPLOSION": bool(explosion_detector.evaluate(indicators)),
        "SQUEEZE_RELEASE": bool(squeeze_detector.evaluate(indicators)),
        "PRE_EXPLOSION": bool(squeeze_detector.pre_explosion(indicators)),
        "CONTINUATION": bool(
            volatility_expansion.evaluate(indicators)
            and momentum_indicators.evaluate(indicators)
        ),
        "VOLUME_SPIKE": bool(volume_expansion.evaluate(indicators)),
    }


def detect_momentum_state(indicators: Dict[str, Any]) -> str:
    checks = diagnose_momentum_state(indicators)
    for state in ("EXPLOSION", "SQUEEZE_RELEASE", "CONTINUATION", "VOLUME_SPIKE", "PRE_EXPLOSION"):
        if checks[state]:
            return state
    return "MOMENTUM_SETUP"


def is_pre_explosion_signal(indicators: Dict[str, Any]) -> bool:
    """Early alert while volatility remains compressed, before a full range burst."""
    direction = str(indicators.get("direction", "")).upper()
    if direction not in ("LONG", "SHORT"):
        return False
    rvol = _number(indicators, "rvol", -1.0)
    expansion = _number(indicators, "atr_expansion", -1.0)
    rsi = _number(indicators, "rsi", -1.0)
    if rvol < 1.10 or expansion < 1.0:
        return False
    rsi_ok = rsi >= 52.0 if direction == "LONG" else 0.0 <= rsi <= 48.0
    if not rsi_ok or not _flag(indicators, "compression_active"):
        return False
    # Two independent directional clues: at least one trend/price-momentum
    # state plus one acceleration state. No candle geometry is used.
    direction_ok = (
        _flag(indicators, "ema_aligned")
        or _flag(indicators, "roc_direction_ok")
        or _flag(indicators, "macd_direction_ok")
    )
    acceleration_ok = (
        _flag(indicators, "macd_accel")
        or _flag(indicators, "rsi_accel")
        or (_flag(indicators, "roc_direction_ok") and _flag(indicators, "ema_aligned"))
    )
    return bool(direction_ok and acceleration_ok)
