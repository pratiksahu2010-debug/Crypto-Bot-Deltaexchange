"""Normal Engine strategy labels built from the Normal indicator set.

The core engine owns mandatory RSI/ATR/15m gates. This module classifies a
candidate using EMA9/21, VWAP and ADX without requiring MACD or candle shape.
"""
from typing import Any, Dict
import math


def _number(indicators, key):
    try:
        value = float(indicators.get(key))
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def diagnose_strategy(indicators: Dict[str, Any], direction: str) -> Dict[str, bool]:
    names = ("BREAKOUT", "TREND_CONTINUATION", "EMA_PULLBACK", "REVERSAL")
    if direction not in ("LONG", "SHORT") or not isinstance(indicators, dict):
        return {name: False for name in names}
    close = _number(indicators, "close")
    e9, e21 = _number(indicators, "ema9"), _number(indicators, "ema21")
    vwap = _number(indicators, "vwap")
    adx = _number(indicators, "adx")
    atr = _number(indicators, "atr")
    sign = 1.0 if direction == "LONG" else -1.0
    aligned = e9 is not None and e21 is not None and sign * (e9-e21) > 0
    vwap_side = close is not None and vwap is not None and sign * (close-vwap) > 0
    trend = bool(aligned and vwap_side and adx is not None and adx >= 20.0)
    breakout_level = _number(indicators, "orb_breakout_level")
    breakout = bool(
        trend and breakout_level is not None and close is not None
        and sign * (close-breakout_level) > 0
    )
    pullback = bool(
        trend and atr is not None and atr > 0 and close is not None and e9 is not None
        and abs(close-e9) <= 0.35*atr
    )
    return {
        "BREAKOUT": breakout,
        "TREND_CONTINUATION": trend,
        "EMA_PULLBACK": pullback,
        # Normal is trend-following; reversal setups are not part of this engine.
        "REVERSAL": False,
    }


def detect_strategy(indicators: Dict[str, Any], direction: str) -> str:
    checks = diagnose_strategy(indicators, direction)
    for name in ("BREAKOUT", "TREND_CONTINUATION", "EMA_PULLBACK"):
        if checks[name]:
            return name
    return "UNCLASSIFIED"
