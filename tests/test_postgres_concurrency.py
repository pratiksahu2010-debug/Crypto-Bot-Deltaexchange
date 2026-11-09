"""PostgreSQL integration tests for durable concurrent signal state.

These tests run in CI against its disposable PostgreSQL service. They are skipped
when TEST_DATABASE_URL is not configured (for example, ordinary local runs).
"""
import asyncio
import json
import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

pytestmark = pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL"),
    reason="TEST_DATABASE_URL is required for PostgreSQL concurrency tests",
)



def _run(coro):
    asyncio.run(coro)


def test_concurrent_claim_is_atomic():
    from database import db as database_db
    from database.db import Base
    from database.models import SignalDedupRecord
    from database.repositories import claim_signal_once

    async def run():
        engine = create_async_engine(os.environ["TEST_DATABASE_URL"], pool_size=10)
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        previous = database_db.SessionLocal
        database_db.SessionLocal = factory
        key_prefix = "TEST-" + uuid.uuid4().hex
        signal = {"pair": key_prefix, "engine": "normal", "direction": "LONG"}
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            results = await asyncio.gather(*[
                claim_signal_once(signal, 123456789, ttl_hours=4) for _ in range(24)
            ])
            assert sum(results) == 1
            async with factory() as session:
                row = await session.get(
                    SignalDedupRecord, f"{key_prefix}:LONG:123456789"
                )
                assert row is not None
        finally:
            database_db.SessionLocal = previous
            await engine.dispose()

    _run(run())



def test_signal_dedup_millisecond_timestamp_column_is_upgraded_to_bigint():
    from database import db as database_db
    from database.db import Base, _upgrade_postgres_schema
    from database.models import SignalDedupRecord
    from database.repositories import claim_signal_once

    async def run():
        engine = create_async_engine(os.environ["TEST_DATABASE_URL"])
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        previous = database_db.SessionLocal
        database_db.SessionLocal = factory
        pair = "TEST-" + uuid.uuid4().hex
        timestamp_ms = 1793730600000
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
                # Recreate the production defect on an empty isolated table. Existing
                # millisecond timestamps cannot fit in int4 and are unrelated to the
                # schema conversion being tested.
                await conn.execute(__import__("sqlalchemy").text("DELETE FROM signal_dedup"))
                await conn.execute(
                    __import__("sqlalchemy").text(
                        "ALTER TABLE signal_dedup ALTER COLUMN candle_time_ms TYPE INTEGER"
                    )
                )
                await _upgrade_postgres_schema(conn)
                column_type = await conn.scalar(
                    __import__("sqlalchemy").text(
                        "SELECT data_type FROM information_schema.columns "
                        "WHERE table_name = 'signal_dedup' AND column_name = 'candle_time_ms'"
                    )
                )
                assert column_type == "bigint"

            signal = {"pair": pair, "engine": "normal", "direction": "LONG"}
            assert await claim_signal_once(signal, timestamp_ms)
            async with factory() as session:
                row = await session.get(
                    SignalDedupRecord, f"{pair}:LONG:{timestamp_ms}"
                )
                assert row is not None
                assert row.candle_time_ms == timestamp_ms
        finally:
            database_db.SessionLocal = previous
            await engine.dispose()

    _run(run())


def test_concurrent_pending_retest_upserts_preserve_one_row_per_key():
    from database import db as database_db
    from database.db import Base
    from database.models import PendingRetestRecord
    from database.repositories import save_pending_retest

    async def run():
        engine = create_async_engine(os.environ["TEST_DATABASE_URL"], pool_size=10)
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        previous = database_db.SessionLocal
        database_db.SessionLocal = factory
        prefix = "TEST-" + uuid.uuid4().hex
        keys = [f"{prefix}:{i}" for i in range(12)]
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            async def write(index):
                key = keys[index % len(keys)]
                pending = {
                    "expires_at": (datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=5)).timestamp(),
                    "signal": {
                        "pair": f"B-TEST{index}_USDT",
                        "engine": "normal",
                        "direction": "LONG",
                    },
                    "revision": index,
                }
                return await save_pending_retest(key, pending)

            results = await asyncio.gather(*(write(i) for i in range(48)))
            assert all(results)
            async with factory() as session:
                count = await session.scalar(
                    select(func.count()).select_from(PendingRetestRecord)
                    .where(PendingRetestRecord.key.in_(keys))
                )
                assert count == len(keys)
                rows = (await session.execute(
                    select(PendingRetestRecord).where(PendingRetestRecord.key.in_(keys))
                )).scalars().all()
                assert all(json.loads(row.payload)["revision"] >= 0 for row in rows)
        finally:
            database_db.SessionLocal = previous
            await engine.dispose()

    _run(run())
