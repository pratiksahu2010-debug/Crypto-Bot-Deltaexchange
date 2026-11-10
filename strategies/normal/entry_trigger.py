"""Indicator-only Normal Engine triggers and UTC-session VWAP.

The Normal strategy is deliberately limited to EMA(9/21), session VWAP,
RSI(14), ADX(14), and ATR(14). Candle colour,
body size, wick shape, volume, order-book data and risk outputs are not entry
indicators for this engine.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

FRESH_CROSS = "FRESH_CROSS"
VWAP_RECLAIM = "VWAP_RECLAIM"
PULLBACK_RESUME = "PULLBACK_RESUME"


def session_vwap_series(high, low, close, volume, times_ms):
    """Calculate cumulative VWAP for the latest UTC day only."""
    arrays = [np.asarray(x, dtype=np.float64) for x in (high, low, close, volume)]
    times = np.asarray(times_ms, dtype=np.int64)
    h, l, c, v = arrays
    n = len(c)
    out = np.full(n, np.nan, dtype=np.float64)
    if (any(x.ndim != 1 for x in arrays) or times.ndim != 1 or n == 0
            or any(len(x) != n for x in (*arrays, times))
            or np.any(np.diff(times) < 0)):
        return out
    active_day = int(times[-1]) // 86_400_000
    pv = total = 0.0
    for i in range(n):
        if int(times[i]) // 86_400_000 != active_day:
            continue
        values = (h[i], l[i], c[i], v[i])
        if (not all(np.isfinite(x) for x in values) or h[i] <= 0 or l[i] <= 0
                or c[i] <= 0 or h[i] < l[i] or c[i] < l[i] or c[i] > h[i] or v[i] < 0):
            continue
        if v[i] > 0:
            pv += ((h[i] + l[i] + c[i]) / 3.0) * v[i]
            total += v[i]
        if total > 0:
            out[i] = pv / total
    return out


def _valid_arrays(*values):
    arrays = [np.asarray(x, dtype=np.float64) for x in values]
    return arrays if arrays and all(x.ndim == 1 and len(x) == len(arrays[0]) for x in arrays) else None


def detect_fresh_trigger(
    open_, high, low, close, ema9_s, ema21_s, vwap_s, atr_value, direction, *,
    max_trigger_age=1, pullback_lookback=4, pullback_touch_atr=0.25,
) -> Optional[Tuple[str, int]]:
    """Detect a fresh EMA cross, VWAP reclaim, or EMA pullback resumption.

    Only completed indicator values and price levels are used. Candle colour,
    body ratio and wick shape are intentionally excluded.
    """
    if direction not in ("LONG", "SHORT"):
        return None
    arrays = _valid_arrays(close, ema9_s, ema21_s, vwap_s)
    if arrays is None:
        return None
    c, e9, e21, vw = arrays
    n = len(c)
    try:
        atr_v = float(atr_value)
    except (TypeError, ValueError, OverflowError):
        return None
    if n < 30 or not math.isfinite(atr_v) or atr_v <= 0:
        return None
    if not all(np.isfinite(x[-3:]).all() for x in (c, e9, e21)):
        return None
    sign = 1.0 if direction == "LONG" else -1.0
    age_limit = max(0, int(max_trigger_age))

    aligned = sign * (e9 - e21) > 0
    run = 0
    for flag in aligned[::-1]:
        if not flag or run >= age_limit + 1:
            break
        run += 1
    fresh_cross = (FRESH_CROSS, run - 1) if (run and n > run and not aligned[-1-run]) else None

    if np.isfinite(vw[-1]):
        side = np.isfinite(vw) & (sign * (c - vw) > 0)
        run = 0
        for flag in side[::-1]:
            if not flag or run >= age_limit + 1:
                break
            run += 1
        if run and n > run and not side[-1-run] and aligned[-1]:
            return VWAP_RECLAIM, run - 1

    if fresh_cross is not None:
        return fresh_cross

    # Indicator-only pullback: recent close approached EMA9 while EMA9/21
    # stayed aligned, then latest close recovered to the trend side of EMA9.
    lookback = max(1, int(pullback_lookback))
    start = max(0, n - 1 - lookback)
    touch_limit = max(0.35, float(pullback_touch_atr)) * atr_v
    recent_touch = np.any(
        (np.abs(c[start:-1] - e9[start:-1]) <= touch_limit)
        & (sign * (e9[start:-1] - e21[start:-1]) > 0)
    )
    prior_near_ema = abs(c[-2] - e9[-2]) <= touch_limit
    resumed = sign * (c[-1] - c[-2]) > 0 and sign * (c[-1] - e9[-1]) > 0
    if recent_touch and aligned[-1] and prior_near_ema and resumed:
        return PULLBACK_RESUME, 0
    return None


def normal_entry_filters(indicators: Dict[str, object], direction: str, **_ignored) -> Dict[str, bool]:
    """Only ATR validity is supplementary; direction comes from the core indicators."""
    try:
        value = float(indicators.get("atr"))
    except (TypeError, ValueError, OverflowError):
        value = float("nan")
    return {"atr_valid": bool(math.isfinite(value) and value > 0)}


def resolve_symbol_pullback_touch_atr(pair: str, default_value: float, overrides: Dict[str, object]) -> float:
    """Resolve a safe per-symbol pullback proximity override."""
    import re

    def normalize(value):
        return re.sub(r"[^A-Z0-9]", "", re.sub(r"^[A-Z]-", "", str(value or "").upper().strip()))

    try:
        fallback = float(default_value)
    except (TypeError, ValueError, OverflowError):
        fallback = 0.25
    if not math.isfinite(fallback) or not 0.05 <= fallback <= 1.0:
        fallback = 0.25
    if not isinstance(overrides, dict):
        return fallback
    target = normalize(pair)
    for key, raw in overrides.items():
        if normalize(key) == target:
            try:
                candidate = float(raw)
            except (TypeError, ValueError, OverflowError):
                return fallback
            return candidate if math.isfinite(candidate) and 0.05 <= candidate <= 1.0 else fallback
    return fallback
