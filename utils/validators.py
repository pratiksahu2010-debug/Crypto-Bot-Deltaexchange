"""Input validation helpers."""
import math
import re
from typing import Optional


PAIR_RE = re.compile(r"^[A-Z0-9]+-[A-Z0-9]+_[A-Z0-9]+$")


def is_valid_pair(pair: str) -> bool:
    if not pair or not isinstance(pair, str):
        return False
    return bool(PAIR_RE.match(pair.strip().upper()))


def normalize_pair(pair: str) -> Optional[str]:
    if not pair:
        return None
    p = pair.strip().upper().replace("/", "_")
    if not p.startswith(("B-", "I-", "HB-", "KC-")) and "_" in p:
        # assume B- prefix for Binance-sourced
        parts = p.split("_")
        if len(parts) == 2:
            p = f"B-{parts[0]}_{parts[1]}"
    return p if is_valid_pair(p) else None


def safe_float(val, default: float = 0.0) -> float:
    try:
        value = float(val)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError, OverflowError):
        return default
