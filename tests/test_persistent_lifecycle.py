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


def test_orb_mex_candidate_persistence_directional_excursions_and_windows(tmp_path):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'strategy_candidates.db'}")
        old_engine, old_factory = db.engine, db.SessionLocal
        db.engine = engine
        db.SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            base_ts = 1_800_000_000_000
            common = {
                "pair": "B-BTC_USDT", "engine": "momentum",
                "strategy_family": "ORB", "strategy": "ORB_BREAKOUT",
                "candidate_ts": base_ts, "entry_price": 100.0,
                "atr_value": 2.0, "decision": "CANDIDATE",
                "stage": "ENTRY_QUALIFIED", "payload": {},
            }
            assert await repositories.save_strategy_candidate({
                **common, "direction": "LONG",
            })
            # Idempotent insert: the same strategy/candle/direction remains one row.
            assert await repositories.save_strategy_candidate({
                **common, "direction": "LONG", "stage": "DUPLICATE",
            })
            assert await repositories.save_strategy_candidate({
                **common, "direction": "SHORT", "strategy": "ORB_RETEST",
            })
            for index in range(1, 13):
                ts = base_ts + index * 300_000
                updated = await repositories.update_strategy_candidate_outcomes(
                    "B-BTC_USDT", ts, high=102.0, low=97.0, close=100.5,
                    interval_ms=300_000,
                )
                assert updated == 2
            from database.models import StrategyCandidateRecord
            from sqlalchemy import select
            async with db.SessionLocal() as session:
                rows = (await session.execute(select(StrategyCandidateRecord))).scalars().all()
            assert len(rows) == 2
            by_direction = {row.direction: row for row in rows}
            long_row, short_row = by_direction["LONG"], by_direction["SHORT"]
            assert long_row.outcome == "OBSERVED_12"
            assert short_row.outcome == "OBSERVED_12"
            assert long_row.bars_observed == short_row.bars_observed == 12
            assert long_row.mfe_pct == pytest.approx(2.0)
            assert long_row.mae_pct == pytest.approx(-3.0)
            assert short_row.mfe_pct == pytest.approx(3.0)
            assert short_row.mae_pct == pytest.approx(-2.0)
            assert await repositories.strategy_candidate_performance_report(30)
            report = await repositories.strategy_candidate_performance_report(30)
            group = report["groups"]["ORB:ORB_BREAKOUT"]
            assert group["count"] == 1
            assert group["windows"]["3"]["sample_count"] == 1
            assert group["windows"]["6"]["sample_count"] == 1
            assert group["windows"]["12"]["sample_count"] == 1
            assert group["windows"]["12"]["reached_1_atr"] == 1
        finally:
            db.engine, db.SessionLocal = old_engine, old_factory
            await engine.dispose()
    asyncio.run(run())


def test_strategy_candidate_gap_does_not_fake_observation_window(tmp_path):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'candidate_gap.db'}")
        old_engine, old_factory = db.engine, db.SessionLocal
        db.engine = engine
        db.SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            base_ts = 1_800_000_000_000
            candidate = {
                "pair": "B-ETH_USDT", "engine": "momentum",
                "strategy_family": "MEX", "strategy": "PRE_EXPLOSION",
                "direction": "LONG", "candidate_ts": base_ts,
                "entry_price": 100.0, "atr_value": 1.0,
                "decision": "WATCHLIST", "stage": "EARLY_CANDIDATE",
            }
            assert await repositories.save_strategy_candidate(candidate)
            # A 10-minute jump is not counted as one completed 5-minute observation.
            assert await repositories.update_strategy_candidate_outcomes(
                "B-ETH_USDT", base_ts + 600_000, 103.0, 99.0, 102.0,
                interval_ms=300_000,
            ) == 0
            from database.models import StrategyCandidateRecord
            async with db.SessionLocal() as session:
                row = await session.get(
                    StrategyCandidateRecord,
                    "B-ETH_USDT:MEX:PRE_EXPLOSION:LONG:" + str(base_ts),
                )
            assert row is not None
            assert row.bars_observed == 0
            assert row.outcome == "ACTIVE"
            import json
            assert json.loads(row.payload)["data_gap"] is True
        finally:
            db.engine, db.SessionLocal = old_engine, old_factory
            await engine.dispose()
    asyncio.run(run())
