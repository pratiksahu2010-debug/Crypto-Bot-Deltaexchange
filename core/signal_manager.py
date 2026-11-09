"""Signal lifecycle with terminal-state cleanup and re-entry support."""
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from utils.logger import logger
from utils.time_utils import utc_iso

TERMINAL = {"closed_tp", "closed_sl", "expired", "invalidated", "evicted"}

class SignalManager:
    def __init__(self, max_active: int = 30, ttl_hours: int = 4):
        self.max_active = max_active
        self.ttl = timedelta(hours=ttl_hours)
        self._signals: Dict[str, dict] = {}
        self._evicted: List[dict] = []

    def _key(self, pair: str, engine: str, direction: str) -> str:
        return f"{pair}:{engine}:{direction}"

    def add(self, signal: dict) -> bool:
        self._purge()
        key = self._key(signal["pair"], signal["engine"], signal["direction"])
        old = self._signals.get(key)
        if old and old.get("status") == "active":
            logger.debug(f"Signal already active: {key}")
            return False
        now = datetime.now(timezone.utc)
        item = dict(signal)
        item.update(status="active", created_at=utc_iso(now), expires_at=utc_iso(now+self.ttl),
                    tp1_hit=False, tp2_hit=False, tp3_hit=False, sl_hit=False)
        self._signals[key] = item
        self._evict()
        logger.info(f"Signal ACTIVE {key} score={item.get('score')}")
        return item.get("status") == "active"

    def restore(self, signal: dict) -> bool:
        """Restore a persisted active signal without resetting its original TTL."""
        key = self._key(signal["pair"], signal["engine"], signal["direction"])
        existing = self._signals.get(key)
        if existing and existing.get("status") == "active":
            # The loader returns newest records first; keep the newest active
            # record instead of allowing an older duplicate row to overwrite it.
            return False
        item = dict(signal)
        try:
            expires_at = datetime.fromisoformat(
                str(item["expires_at"]).replace("Z", "+00:00")
            )
            if expires_at <= datetime.now(timezone.utc):
                return False
        except (KeyError, TypeError, ValueError):
            return False
        item["status"] = "active"
        self._signals[key] = item
        self._evict()
        return item.get("status") == "active"

    def mark_tp(self, pair, engine, direction, level):
        sig = self._signals.get(self._key(pair, engine, direction))
        if not sig or sig.get("status") != "active" or level not in (1,2,3): return False
        sig[f"tp{level}_hit"] = True
        if level == 3: sig["status"] = "closed_tp"
        return True

    def mark_sl(self, pair, engine, direction):
        sig = self._signals.get(self._key(pair, engine, direction))
        if not sig or sig.get("status") != "active": return False
        sig["sl_hit"] = True; sig["status"] = "closed_sl"; return True

    def invalidate(self, pair, engine, direction):
        sig = self._signals.get(self._key(pair, engine, direction))
        if not sig or sig.get("status") != "active": return False
        sig["status"] = "invalidated"; return True

    def get(self, pair, engine, direction): return self._signals.get(self._key(pair,engine,direction))

    def list_active(self, engine=None):
        self._purge(); items=[s for s in self._signals.values() if s.get("status")=="active"]
        if engine: items=[s for s in items if s.get("engine")==engine]
        return sorted(items,key=lambda x:x.get("score",0),reverse=True)

    def count(self): return len(self.list_active())

    def _purge(self):
        now=datetime.now(timezone.utc)
        for key,sig in list(self._signals.items()):
            if sig.get("status") != "active":
                # Keep terminal records briefly for diagnostics, but never let them block re-entry.
                try:
                    created=datetime.fromisoformat(sig.get("created_at","").replace("Z","+00:00"))
                    if now-created > self.ttl*2: self._signals.pop(key,None)
                except Exception: self._signals.pop(key,None)
                continue
            try:
                exp=datetime.fromisoformat(sig["expires_at"].replace("Z","+00:00"))
                if exp <= now: sig["status"]="expired"
            except Exception: sig["status"]="expired"

    def drain_evicted(self) -> List[dict]:
        """Return and clear active signals evicted by the in-memory capacity limit."""
        items = self._evicted
        self._evicted = []
        return items

    def _evict(self):
        active = [k for k, signal in self._signals.items() if signal.get("status") == "active"]
        if len(active) > self.max_active:
            victims = sorted(
                active, key=lambda key: self._signals[key].get("score", 0)
            )[:len(active) - self.max_active]
            for key in victims:
                signal = self._signals[key]
                signal["status"] = "evicted"
                self._evicted.append(dict(signal))
