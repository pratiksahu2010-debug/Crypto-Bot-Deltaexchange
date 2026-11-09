"""Time utilities — always UTC."""
from datetime import datetime, timezone, timedelta
import re
from typing import Optional, Union


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso(dt: Optional[datetime] = None) -> str:
    d = dt or utc_now()
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.isoformat().replace("+00:00", "Z")


def ms_to_dt(ms: Union[int, float]) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def dt_to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def parse_interval(interval: str) -> timedelta:
    """Parse a positive candle interval such as 5m, 15m, 1h, or 1d."""
    if not isinstance(interval, str):
        raise ValueError(f"Invalid interval: {interval!r}")
    match = re.fullmatch(r"(\d+)([mhdw])", interval.strip().lower())
    if not match:
        raise ValueError(f"Unsupported interval: {interval}")
    value = int(match.group(1))
    unit = match.group(2)
    if value <= 0:
        raise ValueError(f"Interval must be positive: {interval}")
    if unit == "m":
        return timedelta(minutes=value)
    if unit == "h":
        return timedelta(hours=value)
    if unit == "d":
        return timedelta(days=value)
    return timedelta(weeks=value)


def floor_to_interval(dt: datetime, interval: str) -> datetime:
    delta = parse_interval(interval)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    ts = int(dt.timestamp())
    step = int(delta.total_seconds())
    floored = ts - (ts % step)
    return datetime.fromtimestamp(floored, tz=timezone.utc)