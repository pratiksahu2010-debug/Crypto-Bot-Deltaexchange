"""Symbol and strategy cooldown management."""
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional
from utils.logger import logger


class CooldownManager:
    def __init__(self, default_seconds: int = 1800):
        self.default_seconds = default_seconds
        self._cooldowns: Dict[str, datetime] = {}

    def _key(self, pair: str, engine: str, direction: str) -> str:
        return f"{pair}:{engine}:{direction}"

    def is_cooling(self, pair: str, engine: str, direction: str) -> bool:
        key = self._key(pair, engine, direction)
        expiry = self._cooldowns.get(key)
        if expiry is None:
            return False
        now = datetime.now(timezone.utc)
        if now >= expiry:
            del self._cooldowns[key]
            return False
        return True

    def set(self, pair: str, engine: str, direction: str, seconds: Optional[int] = None) -> None:
        key = self._key(pair, engine, direction)
        secs = seconds if seconds is not None else self.default_seconds
        self._cooldowns[key] = datetime.now(timezone.utc) + timedelta(seconds=secs)
        logger.debug(f"Cooldown set {key} for {secs}s")

    def clear(self, pair: str, engine: str, direction: str) -> None:
        key = self._key(pair, engine, direction)
        self._cooldowns.pop(key, None)

    def clear_pair(self, pair: str) -> None:
        keys = [k for k in self._cooldowns if k.startswith(f"{pair}:")]
        for k in keys:
            del self._cooldowns[k]

    def remaining(self, pair: str, engine: str, direction: str) -> float:
        key = self._key(pair, engine, direction)
        expiry = self._cooldowns.get(key)
        if not expiry:
            return 0.0
        delta = (expiry - datetime.now(timezone.utc)).total_seconds()
        return max(0.0, delta)

    def active_count(self) -> int:
        now = datetime.now(timezone.utc)
        expired = [k for k, v in self._cooldowns.items() if v <= now]
        for k in expired:
            del self._cooldowns[k]
        return len(self._cooldowns)
