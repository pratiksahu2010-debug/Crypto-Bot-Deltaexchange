"""Candle-time alignment and freshness guards for multi-timeframe signals."""
from typing import Optional, Sequence


def candle_age_seconds(now_ms: int, candle_open_ms: int, interval_ms: int) -> float:
    """Age since a candle closed, not since it opened."""
    if interval_ms <= 0:
        raise ValueError("interval_ms must be positive")
    return (int(now_ms) - (int(candle_open_ms) + int(interval_ms))) / 1000.0


def latest_asof_candle_index(
    open_times_ms: Sequence[int], interval_ms: int, decision_close_ms: int,
) -> Optional[int]:
    """Latest fully closed context candle no later than the decision candle close.

    Reject context older than one full context interval. This keeps a delayed or
    gapped 15m series from silently validating an unrelated 5m entry candle.
    """
    if interval_ms <= 0:
        raise ValueError("interval_ms must be positive")
    eligible = [
        i for i, opened in enumerate(open_times_ms)
        if int(opened) + interval_ms <= int(decision_close_ms)
    ]
    if not eligible:
        return None
    idx = eligible[-1]
    context_close_ms = int(open_times_ms[idx]) + interval_ms
    if int(decision_close_ms) - context_close_ms > interval_ms:
        return None
    return idx


def seconds_until_next_candle_boundary(now_ms: int, interval_ms: int, grace_ms: int = 0) -> float:
    """Seconds until the next usable candle-close boundary."""
    if interval_ms <= 0 or grace_ms < 0:
        raise ValueError("interval_ms must be positive and grace_ms non-negative")
    now, interval, grace = int(now_ms), int(interval_ms), int(grace_ms)
    boundary = (now // interval) * interval
    ready_at = boundary + grace
    target = ready_at if now <= ready_at else boundary + interval + grace
    return max(0.0, (target - now) / 1000.0)
