"""Signal and watchlist lifecycle manager regression tests."""
from datetime import datetime, timedelta, timezone

from core.signal_manager import SignalManager
from core.watchlist_manager import WatchlistManager


def test_signal_manager_reports_capacity_evictions_for_persistence():
    manager = SignalManager(max_active=1)
    low = {
        "pair": "B-LOW_USDT", "engine": "normal", "direction": "LONG", "score": 6.5,
    }
    high = {
        "pair": "B-HIGH_USDT", "engine": "normal", "direction": "LONG", "score": 9.0,
    }
    assert manager.add(low)
    assert manager.add(high)
    assert manager.get("B-LOW_USDT", "normal", "LONG")["status"] == "evicted"
    evicted = manager.drain_evicted()
    assert len(evicted) == 1
    assert evicted[0]["pair"] == "B-LOW_USDT"
    assert manager.drain_evicted() == []


def test_watchlist_add_purges_expired_entries_before_capacity_eviction():
    manager = WatchlistManager(max_entries=1)
    manager._entries["B-OLD_USDT:normal:LONG"] = {
        "pair": "B-OLD_USDT", "engine": "normal", "direction": "LONG",
        "score": 99.0, "expires_at": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
    }
    assert manager.add({
        "pair": "B-NEW_USDT", "engine": "normal", "direction": "LONG", "score": 1.0,
    })
    assert manager.get("B-OLD_USDT", "normal", "LONG") is None
    assert manager.get("B-NEW_USDT", "normal", "LONG") is not None


def test_watchlist_invalid_or_missing_expiry_is_treated_as_expired():
    manager = WatchlistManager()
    assert manager._expired({"expires_at": "not-a-timestamp"})
    assert manager._expired({})
    assert not manager._expired({
        "expires_at": datetime.now(timezone.utc) + timedelta(minutes=5),
    })


def test_engine_persists_signal_capacity_evictions(monkeypatch):
    import asyncio
    from core.engine import SignalEngine
    from database import repositories

    async def run():
        engine = SignalEngine(object(), object(), object())
        engine.signals["normal"].max_active = 1
        engine.signals["normal"].add({
            "pair": "B-LOW_USDT", "engine": "normal", "direction": "LONG", "score": 6.5,
        })
        engine.signals["normal"].add({
            "pair": "B-HIGH_USDT", "engine": "normal", "direction": "LONG", "score": 9.0,
        })
        updated = []

        async def fake_update(signal, status):
            updated.append((signal["pair"], status))
            return True

        monkeypatch.setattr(repositories, "update_signal_record", fake_update)
        await engine._persist_evicted_signals("normal")
        assert updated == [("B-LOW_USDT", "evicted")]

    asyncio.run(run())


def test_tp1_hit_allows_later_candle_reentry_and_tracks_both_signals():
    manager = SignalManager()
    first = {
        "pair": "B-BTC_USDT", "engine": "momentum", "direction": "LONG",
        "score": 8.5, "candle_time_ms": 1_800_000_000_000,
    }
    second = {
        "pair": "B-BTC_USDT", "engine": "momentum", "direction": "LONG",
        "score": 9.0, "candle_time_ms": 1_800_000_300_000,
    }
    assert manager.add(first)
    first_id = manager.get("B-BTC_USDT", "momentum", "LONG")["signal_id"]
    assert not manager.add({**first, "score": 9.5})
    assert not manager.add(second)
    assert manager.mark_tp("B-BTC_USDT", "momentum", "LONG", 1, first_id)
    assert manager.add(second)
    active = manager.list_active("momentum")
    assert len(active) == 2
    assert manager.get("B-BTC_USDT", "momentum", "LONG", first_id)["tp1_hit"] is True
    assert manager.get("B-BTC_USDT", "momentum", "LONG")["candle_time_ms"] == second["candle_time_ms"]


def test_tp1_does_not_allow_same_or_older_candle_reentry():
    manager = SignalManager()
    first = {
        "pair": "B-ETH_USDT", "engine": "normal", "direction": "SHORT",
        "candle_time_ms": 1_800_000_000_000,
    }
    assert manager.add(first)
    first_id = manager.get("B-ETH_USDT", "normal", "SHORT")["signal_id"]
    assert manager.mark_tp("B-ETH_USDT", "normal", "SHORT", 1, first_id)
    assert not manager.add({**first, "score": 10})
    assert not manager.add({**first, "candle_time_ms": 1_799_999_700_000})


def test_signal_manager_blocks_new_setup_until_previous_tp1():
    manager = SignalManager()
    first = {
        "pair": "B-SOL_USDT", "engine": "normal", "direction": "LONG",
        "candle_time_ms": 1_800_000_000_000,
    }
    second = {**first, "candle_time_ms": 1_800_000_300_000}
    assert manager.add(first)
    assert not manager.add(second)
    assert len(manager.list_active()) == 1
