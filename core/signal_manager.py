"""Signal lifecycle with same-candle deduplication and TP1 re-entry support."""
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

    def _key(self, pair: str, engine: str, direction: str, strategy: Optional[str] = None) -> str:
        base = f"{pair}:{engine}:{direction}"
        return f"{base}:{strategy}" if strategy else base

    def _record_key(self, signal: dict) -> str:
        base = self._key(signal["pair"], signal["engine"], signal["direction"], signal.get("strategy"))
        candle = signal.get("candle_time_ms") or signal.get("candle_ts")
        return f"{base}:{int(candle)}" if candle is not None else base

    def _matching(self, pair: str, engine: str, direction: str, strategy: Optional[str] = None) -> List[dict]:
        return [
            item for item in self._signals.values()
            if item.get("pair") == pair
            and item.get("engine") == engine
            and item.get("direction") == direction
            and (strategy is None or item.get("strategy") == strategy)
        ]

    @staticmethod
    def _signal_id(signal: dict) -> str:
        candle = signal.get("candle_time_ms") or signal.get("candle_ts")
        if candle is not None:
            return f"{signal['pair']}:{signal['engine']}:{signal['direction']}:{signal.get('strategy', '')}:{int(candle)}"
        return f"{signal['pair']}:{signal['engine']}:{signal['direction']}:{signal.get('strategy', '')}:{signal.get('created_at', '')}"

    def can_reenter(self, signal: dict) -> bool:
        """Allow a later-candle setup only after every prior active setup hit TP1."""
        candle = signal.get("candle_time_ms") or signal.get("candle_ts")
        if candle is None:
            return not any(
                old.get("status") == "active"
                for old in self._matching(signal["pair"], signal["engine"], signal["direction"], signal.get("strategy"))
            )
        try:
            candle = int(candle)
        except (TypeError, ValueError, OverflowError):
            return False
        active = [
            old for old in self._matching(
                signal["pair"], signal["engine"], signal["direction"], signal.get("strategy")
            )
            if old.get("status") == "active"
        ]
        for old in active:
            old_candle = old.get("candle_time_ms") or old.get("candle_ts")
            if not old.get("tp1_hit") or old_candle is None:
                return False
            try:
                if candle <= int(old_candle):
                    return False
            except (TypeError, ValueError, OverflowError):
                return False
        return True

    def add(self, signal: dict) -> bool:
        self._purge()
        key = self._record_key(signal)
        if key in self._signals:
            logger.debug(f"Signal duplicate candle: {key}")
            return False
        if not self.can_reenter(signal):
            logger.debug(f"Signal blocked by active setup before TP1: {key}")
            return False

        now = datetime.now(timezone.utc)
        item = dict(signal)
        item.update(
            status="active", created_at=utc_iso(now),
            expires_at=utc_iso(now + self.ttl),
            tp1_hit=False, tp2_hit=False, tp3_hit=False, sl_hit=False,
        )
        item["signal_id"] = self._signal_id(item)
        signal["signal_id"] = item["signal_id"]
        self._signals[key] = item
        self._evict()
        logger.info(f"Signal ACTIVE {key} score={item.get('score')}")
        return item.get("status") == "active"

    def restore(self, signal: dict) -> bool:
        """Restore one persisted signal while preserving its original TTL and identity."""
        item = dict(signal)
        key = self._record_key(item)
        if key in self._signals and self._signals[key].get("status") == "active":
            return False
        try:
            expires_at = datetime.fromisoformat(
                str(item["expires_at"]).replace("Z", "+00:00")
            )
            if expires_at <= datetime.now(timezone.utc):
                return False
        except (KeyError, TypeError, ValueError):
            return False
        item["status"] = "active"
        item.setdefault("signal_id", self._signal_id(item))
        self._signals[key] = item
        self._evict()
        return item.get("status") == "active"

    def _find(self, pair, engine, direction, signal_id=None, strategy=None):
        matches = self._matching(pair, engine, direction, strategy)
        if signal_id is not None:
            return next((s for s in matches if s.get("signal_id") == signal_id), None)
        active = [s for s in matches if s.get("status") == "active"]
        candidates = active or matches
        return max(candidates, key=lambda s: s.get("created_at", ""), default=None)

    def mark_tp(self, pair, engine, direction, level, signal_id=None):
        sig = self._find(pair, engine, direction, signal_id)
        if not sig or sig.get("status") != "active" or level not in (1, 2, 3):
            return False
        sig[f"tp{level}_hit"] = True
        if level == 3:
            sig["status"] = "closed_tp"
        return True

    def mark_sl(self, pair, engine, direction, signal_id=None):
        sig = self._find(pair, engine, direction, signal_id)
        if not sig or sig.get("status") != "active":
            return False
        sig["sl_hit"] = True
        sig["status"] = "closed_sl"
        return True

    def invalidate(self, pair, engine, direction, signal_id=None):
        sig = self._find(pair, engine, direction, signal_id)
        if not sig or sig.get("status") != "active":
            return False
        sig["status"] = "invalidated"
        return True

    def get(self, pair, engine, direction, signal_id=None, strategy=None):
        return self._find(pair, engine, direction, signal_id, strategy)

    def list_active(self, engine=None):
        self._purge()
        items = [s for s in self._signals.values() if s.get("status") == "active"]
        if engine:
            items = [s for s in items if s.get("engine") == engine]
        return sorted(items, key=lambda x: x.get("score", 0), reverse=True)

    def count(self):
        return len(self.list_active())

    def _purge(self):
        now = datetime.now(timezone.utc)
        for key, sig in list(self._signals.items()):
            if sig.get("status") != "active":
                try:
                    created = datetime.fromisoformat(sig.get("created_at", "").replace("Z", "+00:00"))
                    if now - created > self.ttl * 2:
                        self._signals.pop(key, None)
                except Exception:
                    self._signals.pop(key, None)
                continue
            try:
                exp = datetime.fromisoformat(sig["expires_at"].replace("Z", "+00:00"))
                if exp <= now:
                    sig["status"] = "expired"
            except Exception:
                sig["status"] = "expired"

    def drain_evicted(self) -> List[dict]:
        items = self._evicted
        self._evicted = []
        return items

    def _evict(self):
        active = [k for k, signal in self._signals.items() if signal.get("status") == "active"]
        if len(active) > self.max_active:
            victims = sorted(active, key=lambda key: self._signals[key].get("score", 0))[
                :len(active) - self.max_active
            ]
            for key in victims:
                signal = self._signals[key]
                signal["status"] = "evicted"
                self._evicted.append(dict(signal))
