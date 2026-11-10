"""Focused async tests for pending-retest direction exclusivity and persistence fail-closed."""
import pytest

from core.engine import SignalEngine
import database.repositories as repositories


@pytest.mark.asyncio
async def test_new_pending_retest_replaces_opposite_direction_after_durable_save(monkeypatch):
    engine = SignalEngine.__new__(SignalEngine)
    engine.pending_retests = {
        "BTC:normal:LONG": {
            "signal": {"pair": "BTC_USDT", "engine": "normal", "direction": "LONG"},
        }
    }
    saved = []
    deleted = []

    async def save(key, pending):
        saved.append(key)
        return True

    async def delete(key):
        deleted.append(key)
        return True

    monkeypatch.setattr(repositories, "save_pending_retest", save)
    monkeypatch.setattr(repositories, "delete_pending_retest", delete)

    new_key = "BTC:normal:SHORT"
    pending = {
        "signal": {"pair": "BTC_USDT", "engine": "normal", "direction": "SHORT"},
        "expires_at": 9999999999,
        "entry": 100.0, "zone_low": 99.0, "zone_high": 101.0,
        "stop_loss": 103.0, "candle_ts": 123,
    }

    assert await engine._persist_pending_retest_candidate(new_key, pending)
    assert saved == [new_key]
    assert deleted == ["BTC:normal:LONG"]
    assert set(engine.pending_retests) == {new_key}


@pytest.mark.asyncio
async def test_opposite_pending_retest_delete_failure_keeps_old_candidate(monkeypatch):
    engine = SignalEngine.__new__(SignalEngine)
    old_key = "ETH:momentum:LONG"
    new_key = "ETH:momentum:SHORT"
    engine.pending_retests = {
        old_key: {"signal": {"pair": "ETH_USDT", "engine": "momentum", "direction": "LONG"}}
    }
    saved = []
    deleted = []

    async def save(key, pending):
        saved.append(key)
        return True

    async def delete(key):
        deleted.append(key)
        return False

    monkeypatch.setattr(repositories, "save_pending_retest", save)
    monkeypatch.setattr(repositories, "delete_pending_retest", delete)

    pending = {
        "signal": {"pair": "ETH_USDT", "engine": "momentum", "direction": "SHORT"},
        "expires_at": 9999999999,
        "entry": 100.0, "zone_low": 99.0, "zone_high": 101.0,
        "stop_loss": 103.0, "candle_ts": 456,
    }

    assert not await engine._persist_pending_retest_candidate(new_key, pending)
    assert old_key in engine.pending_retests
    assert new_key not in engine.pending_retests
    # New row save was followed by fail-closed cleanup attempt.
    assert deleted == [old_key, new_key]


@pytest.mark.asyncio
async def test_pending_retest_restore_loads_only_unexpired_records(monkeypatch):
    from utils.time_utils import utc_now

    engine = SignalEngine.__new__(SignalEngine)
    engine.pending_retests = {}
    future = utc_now().timestamp() + 120
    past = utc_now().timestamp() - 120
    records = [
        ("BTC:normal:LONG", {
            "signal": {"pair": "BTC_USDT", "engine": "normal", "direction": "LONG"},
            "expires_at": future,
        }),
        ("ETH:normal:SHORT", {
            "signal": {"pair": "ETH_USDT", "engine": "normal", "direction": "SHORT"},
            "expires_at": past,
        }),
    ]

    async def load(limit=500):
        return records

    monkeypatch.setattr(repositories, "load_pending_retests", load)
    restored = await engine.restore_pending_retests()

    assert restored == 1
    assert set(engine.pending_retests) == {"BTC:normal:LONG"}


@pytest.mark.asyncio
async def test_signal_claim_duplicate_is_blocked_after_database_engine_restart(monkeypatch):
    """Postgres-backed durable dedup check; skipped in local runs without CI DB."""
    import os
    import uuid
    from sqlalchemy import delete
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from database.db import Base
    import database.db as database_db
    from database.models import SignalDedupRecord

    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is not configured (CI PostgreSQL service required)")
    pair = f"TEST_{uuid.uuid4().hex[:12]}_USDT"
    normal_signal = {"pair": pair, "direction": "LONG", "engine": "normal"}
    momentum_signal = {"pair": pair, "direction": "LONG", "engine": "momentum"}
    candle_ms = 1_800_000_000_000

    engine1 = create_async_engine(database_url)
    async with engine1.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    monkeypatch.setattr(database_db, "SessionLocal", async_sessionmaker(
        engine1, expire_on_commit=False,
    ))

    # Normal and Momentum are independent: same pair/direction/candle can
    # produce one signal from each engine, but each engine is deduplicated.
    assert await repositories.claim_signal_once(normal_signal, candle_ms, ttl_hours=4) is True
    assert await repositories.claim_signal_once(momentum_signal, candle_ms, ttl_hours=4) is True
    assert await repositories.claim_signal_once(normal_signal, candle_ms, ttl_hours=4) is False
    assert await repositories.claim_signal_once(momentum_signal, candle_ms, ttl_hours=4) is False

    # Simulate process restart: close the first pool and establish a fresh one.
    await engine1.dispose()
    engine2 = create_async_engine(database_url)
    monkeypatch.setattr(repositories.database_db, "SessionLocal", async_sessionmaker(
        engine2, expire_on_commit=False,
    ))
    assert await repositories.claim_signal_once(normal_signal, candle_ms, ttl_hours=4) is False
    assert await repositories.claim_signal_once(momentum_signal, candle_ms, ttl_hours=4) is False

    async with database_db.SessionLocal() as session:
        await session.execute(delete(SignalDedupRecord).where(
            SignalDedupRecord.dedup_key.in_([
                f"{pair}:normal:LONG:{candle_ms}",
                f"{pair}:momentum:LONG:{candle_ms}",
            ])
        ))
        await session.commit()
    await engine2.dispose()


@pytest.mark.asyncio
async def test_pending_retest_restore_keeps_newest_direction_for_pair_engine(monkeypatch):
    from utils.time_utils import utc_now

    engine = SignalEngine.__new__(SignalEngine)
    engine.pending_retests = {}
    expiry = utc_now().timestamp() + 120
    newest_key = "SOL:normal:SHORT:new"
    older_key = "SOL:normal:LONG:old"
    records = [
        (newest_key, {"signal": {"pair": "SOL_USDT", "engine": "normal", "direction": "SHORT"}, "expires_at": expiry}),
        (older_key, {"signal": {"pair": "SOL_USDT", "engine": "normal", "direction": "LONG"}, "expires_at": expiry}),
    ]
    deleted = []

    async def load(limit=500):
        return records

    async def delete(key):
        deleted.append(key)
        return True

    monkeypatch.setattr(repositories, "load_pending_retests", load)
    monkeypatch.setattr(repositories, "delete_pending_retest", delete)

    assert await engine.restore_pending_retests() == 1
    assert set(engine.pending_retests) == {newest_key}
    assert deleted == [older_key]
