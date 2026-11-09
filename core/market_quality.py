"""Fail-closed market microstructure checks for signal candidates."""
import math
from typing import Any, Dict, Optional, Tuple


def evaluate_orderbook_spread(
    snapshot: Any, max_spread_bps: float
) -> Tuple[bool, Optional[float], str]:
    """Accept only a finite, non-negative spread within the configured limit."""
    try:
        spread = float(snapshot["spread_bps"])
        maximum = float(max_spread_bps)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False, None, "orderbook_unavailable"
    if not math.isfinite(spread) or spread < 0:
        return False, None, "invalid_orderbook_spread"
    if not math.isfinite(maximum) or maximum <= 0:
        return False, spread, "invalid_spread_limit"
    if spread > maximum:
        return False, spread, "spread_too_wide"
    return True, spread, "ok"



def evaluate_orderbook_depth(
    snapshot: Any, min_side_depth_usdt: float
) -> Tuple[bool, Optional[float], str]:
    """Optionally require minimum quote-notional on both sides of the top five levels.

    A zero threshold disables filtering while still allowing depth metrics to
    be attached to signals for observation and later calibration.
    """
    try:
        minimum = float(min_side_depth_usdt)
    except (TypeError, ValueError, OverflowError):
        return False, None, "invalid_depth_limit"
    if not math.isfinite(minimum) or minimum < 0:
        return False, None, "invalid_depth_limit"
    try:
        bid_depth = float(snapshot["bid_depth_usdt"])
        ask_depth = float(snapshot["ask_depth_usdt"])
        depth = min(bid_depth, ask_depth)
    except (KeyError, TypeError, ValueError, OverflowError):
        if minimum == 0:
            return True, None, "disabled"
        return False, None, "orderbook_depth_unavailable"
    if not all(math.isfinite(value) and value >= 0 for value in (bid_depth, ask_depth, depth)):
        return False, None, "invalid_orderbook_depth"
    if minimum == 0:
        return True, depth, "disabled"
    if depth < minimum:
        return False, depth, "insufficient_depth"
    return True, depth, "ok"
