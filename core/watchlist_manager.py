"""Independent watchlist management for Normal and Momentum engines."""
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from utils.logger import logger
from utils.time_utils import utc_iso


class WatchlistManager:
    def __init__(self, max_entries: int = 50, ttl_minutes: int = 60):
        self.max_entries = max_entries
        self.ttl = timedelta(minutes=ttl_minutes)
        self._entries: Dict[str, dict] = {}

    def _key(self, pair: str, engine: str, direction: str) -> str:
        return f"{pair}:{engine}:{direction}"

    def add(self, signal: dict) -> bool:
        self._purge_expired()
        pair = signal["pair"]
        engine = signal["engine"]
        direction = signal["direction"]
        key = self._key(pair, engine, direction)
        now = datetime.now(timezone.utc)
        signal = dict(signal)
        previous = self._entries.get(key)
        # Refresh the latest observation without extending the original setup
        # lifetime on every intrabar update. This prevents stale setups living
        # forever merely because a feed keeps sending updates.
        if previous and not self._expired(previous):
            signal["added_at"] = previous.get("added_at", utc_iso(now))
            signal["expires_at"] = previous.get("expires_at", utc_iso(now + self.ttl))
        else:
            signal["added_at"] = utc_iso(now)
            signal["expires_at"] = utc_iso(now + self.ttl)
        self._entries[key] = signal
        self._evict()
        logger.info(f"Watchlist add {key} score={signal.get('score')}")
        return True

    def remove(self, pair: str, engine: str, direction: str) -> bool:
        key = self._key(pair, engine, direction)
        if key in self._entries:
            del self._entries[key]
            return True
        return False

    def remove_pair(self, pair: str, engine: str) -> int:
        """Remove all directional watch entries for a pair in one engine."""
        prefix = f"{pair}:{engine}:"
        keys = [key for key in self._entries if key.startswith(prefix)]
        for key in keys:
            del self._entries[key]
        return len(keys)

    def get(self, pair: str, engine: str, direction: str) -> Optional[dict]:
        key = self._key(pair, engine, direction)
        entry = self._entries.get(key)
        if not entry:
            return None
        if self._expired(entry):
            del self._entries[key]
            return None
        return entry

    def list_all(self, engine: Optional[str] = None) -> List[dict]:
        self._purge_expired()
        items = list(self._entries.values())
        if engine:
            items = [e for e in items if e.get("engine") == engine]
        return sorted(items, key=lambda x: x.get("score", 0), reverse=True)

    def promote_candidates(self, threshold: float = 7.0) -> List[dict]:
        self._purge_expired()
        return [e for e in self._entries.values() if e.get("score", 0) >= threshold]

    def count(self, engine: Optional[str] = None) -> int:
        self._purge_expired()
        if engine is None:
            return len(self._entries)
        return sum(1 for e in self._entries.values() if e.get("engine") == engine)

    def _expired(self, entry: dict) -> bool:
        exp = entry.get("expires_at")
        if not exp:
            return True
        try:
            if isinstance(exp, datetime):
                parsed = exp
            else:
                parsed = datetime.fromisoformat(str(exp).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed <= datetime.now(timezone.utc)
        except (TypeError, ValueError, OverflowError):
            # Invalid expiry data must not keep a stale watchlist setup forever.
            return True

    def _purge_expired(self) -> None:
        expired = [k for k, v in self._entries.items() if self._expired(v)]
        for k in expired:
            del self._entries[k]

    def _evict(self) -> None:
        if len(self._entries) <= self.max_entries:
            return
        sorted_keys = sorted(self._entries.keys(), key=lambda k: self._entries[k].get("score", 0))
        for k in sorted_keys[: len(self._entries) - self.max_entries]:
            del self._entries[k]
