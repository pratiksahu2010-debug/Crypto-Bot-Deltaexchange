"""Momentum regime dispatcher: strict pre-ignition, release and continuation."""
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
    """Stage 1: surface early ignition before every lagging indicator aligns."""
    direction = str(indicators.get("direction", "")).upper()
    if direction not in ("LONG", "SHORT"):
        return False
    rvol = _number(indicators, "rvol", -1.0)
    expansion = _number(indicators, "atr_expansion", -1.0)
    adx = _number(indicators, "adx", -1.0)
    if rvol < 1.10 or expansion < 1.0:
        return False
    adx_rising = _flag(indicators, "adx_rising")
    if adx < 17.0:
        return False
    if not _flag(indicators, "compression_active"):
        return False

    plus_di = _number(indicators, "plus_di", -1.0)
    minus_di = _number(indicators, "minus_di", -1.0)
    di_available = plus_di >= 0 and minus_di >= 0
    di_ok = (
        (plus_di > minus_di if direction == "LONG" else minus_di > plus_di)
        if di_available else False
    )

    # Two of three directional signals are enough to keep an early candidate
    # alive; one acceleration source must also be present.
    directional_votes = sum((
        _flag(indicators, "ema_aligned"),
        _flag(indicators, "roc_direction_ok"),
        _flag(indicators, "macd_direction_ok"),
    ))
    acceleration = (
        _flag(indicators, "macd_accel")
        or _flag(indicators, "rsi_accel")
        or _flag(indicators, "bw_expand")
    )
    rsi = _number(indicators, "rsi", -1.0)
    rsi_ok = rsi >= 52.0 if direction == "LONG" else rsi <= 48.0

    # If DI is temporarily opposed, require rising ADX and all three
    # directional votes. Otherwise DI remains useful confirmation, not a veto.
    di_conflict_ok = di_ok or (adx_rising and directional_votes == 3)
    return bool(
        directional_votes >= 2 and acceleration
        and (rsi_ok or directional_votes == 3)
        and di_conflict_ok
    )
