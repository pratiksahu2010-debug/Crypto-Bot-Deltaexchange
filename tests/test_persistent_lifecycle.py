"""Persistence regression tests for pending retests and signal outcomes."""
import asyncio
from datetime import datetime, timezone
import pytest

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from database import db
from database.db import Base
from database import repositories


def test_pending_retest_round_trip_and_expiry(monkeypatch):
    async def run():
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        monkeypatch.setattr(db, "SessionLocal", async_sessionmaker(engine, expire_on_commit=False))
        pending = {
            "signal": {"pair": "B-BTC_USDT", "engine": "normal", "direction": "LONG"},
            "expires_at": datetime.now(timezone.utc).timestamp() + 300,
            "entry": 100.0, "zone_low": 99.0, "zone_high": 101.0,
            "stop_loss": 98.0, "candle_ts": 1_800_000_000_000,
        }
        assert await repositories.save_pending_retest("btc:normal:long", pending)
        restored = await repositories.load_pending_retests()
        assert len(restored) == 1
        assert restored[0][0] == "btc:normal:long"
        assert restored[0][1]["signal"]["pair"] == "B-BTC_USDT"
        assert await repositories.delete_pending_retest("btc:normal:long")
        assert await repositories.load_pending_retests() == []
        await engine.dispose()
    asyncio.run(run())


def test_signal_outcome_report_groups_by_engine_and_strategy(monkeypatch):
    async def run():
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        monkeypatch.setattr(db, "SessionLocal", async_sessionmaker(engine, expire_on_commit=False))
        signal = {"pair": "B-BTC_USDT", "engine": "momentum", "strategy": "EXPLOSION",
                  "direction": "LONG", "price": 100.0, "live_price": 100.0,
                  "candle_time_ms": 1_800_000_000_000}
        assert await repositories.record_signal_outcome(signal, 102.0, "TP1")
        assert await repositories.record_signal_outcome(signal, 99.0, "TP1")
        report = await repositories.signal_performance_report(days=30)
        row = report["groups"]["momentum:EXPLOSION"]
        assert row["count"] == 1
        assert row["avg_mfe_pct"] == pytest.approx(2.0)
        assert row["avg_mae_pct"] == pytest.approx(-1.0)
        await engine.dispose()
    asyncio.run(run())


def test_signal_outcome_progression_and_performance_rates(tmp_path):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'outcome_progression.db'}")
        old_engine, old_factory = db.engine, db.SessionLocal
        db.engine = engine
        db.SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            signal = {
                "pair": "B-ETH_USDT", "engine": "momentum", "strategy": "EXPLOSION",
                "direction": "LONG", "price": 100.0, "live_price": 100.0,
                "candle_time_ms": 1_800_000_000_000,
            }
            signal["tp1_hit"] = True
            assert await repositories.record_signal_outcome(signal, 101.0, "TP1")
            signal["tp2_hit"] = True
            assert await repositories.record_signal_outcome(signal, 102.0, "TP2")
            signal["tp3_hit"] = True
            assert await repositories.record_signal_outcome(signal, 103.0, "TP3")
            # A later monitoring sample must not downgrade a completed TP3.
            assert await repositories.record_signal_outcome(signal, 99.0, "ACTIVE")
            report = await repositories.signal_performance_report(days=30)
            row = report["groups"]["momentum:EXPLOSION"]
            assert row["count"] == 1
            assert row["tp1_rate"] == pytest.approx(1.0)
            assert row["tp2_rate"] == pytest.approx(1.0)
            assert row["tp3_rate"] == pytest.approx(1.0)
            assert row["avg_mfe_pct"] == pytest.approx(3.0)
            assert row["avg_mae_pct"] == pytest.approx(-1.0)
            assert row["active"] == 0
        finally:
            db.engine, db.SessionLocal = old_engine, old_factory
            await engine.dispose()
    asyncio.run(run())


def test_partial_target_rate_survives_later_stop(tmp_path):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'partial_stop.db'}")
        old_engine, old_factory = db.engine, db.SessionLocal
        db.engine = engine
        db.SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            signal = {
                "pair": "B-SOL_USDT", "engine": "normal", "strategy": "PULLBACK",
                "direction": "LONG", "price": 100.0, "live_price": 100.0,
                "candle_time_ms": 1_800_000_000_001, "tp1_hit": True,
            }
            assert await repositories.record_signal_outcome(signal, 101.0, "TP1")
            signal["sl_hit"] = True
            assert await repositories.record_signal_outcome(signal, 98.0, "STOP")
            report = await repositories.signal_performance_report(days=30)
            row = report["groups"]["normal:PULLBACK"]
            assert row["tp1_rate"] == pytest.approx(1.0)
            assert row["tp2_rate"] == pytest.approx(0.0)
            assert row["stop_rate"] == pytest.approx(1.0)
        finally:
            db.engine, db.SessionLocal = old_engine, old_factory
            await engine.dispose()
    asyncio.run(run())


def test_signal_lifecycle_update_targets_only_matching_candle_after_reentry(tmp_path):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'signal_reentry.db'}")
        old_engine, old_factory = db.engine, db.SessionLocal
        db.engine = engine
        db.SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            first = {
                "pair": "B-BTC_USDT", "engine": "momentum", "direction": "LONG",
                "strategy": "EXPLOSION", "price": 100.0,
                "candle_time_ms": 1_800_000_000_000,
                "expires_at": "2099-01-01T00:00:00+00:00",
            }
            second = {
                **first, "candle_time_ms": 1_800_000_300_000,
                "price": 102.0,
            }
            assert await repositories.save_signal(first)
            assert await repositories.save_signal(second)
            assert await repositories.update_signal_record(first, "closed_tp")
            active = await repositories.load_active_signals()
            assert len(active) == 1
            assert active[0]["candle_time_ms"] == second["candle_time_ms"]
        finally:
            db.engine, db.SessionLocal = old_engine, old_factory
            await engine.dispose()
    asyncio.run(run())
