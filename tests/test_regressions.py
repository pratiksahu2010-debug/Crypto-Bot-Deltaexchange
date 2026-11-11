from datetime import datetime, timezone, timedelta
import numpy as np
from core.signal_manager import SignalManager
from core.risk_calculator import calculate_risk
from exchange.candle_builder import CandleBuffer, CandleBuilder


def test_signal_can_reenter_after_terminal_state():
    m = SignalManager(ttl_hours=1)
    s = {"pair":"B-BTC_USDT", "engine":"normal", "direction":"LONG", "score":8}
    assert m.add(s)
    assert not m.add(s)
    assert m.mark_sl(s["pair"], s["engine"], s["direction"])
    assert m.add(s)


def test_signal_expires_and_can_reenter():
    m = SignalManager(ttl_hours=1)
    s = {"pair":"B-ETH_USDT", "engine":"normal", "direction":"SHORT", "score":8}
    assert m.add(s)
    m._signals[m._key(s['pair'],s['engine'],s['direction'])]['expires_at'] = (datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
    assert m.add(s)


def test_candle_buffer_deduplicates_and_ignores_older():
    b=CandleBuffer(10)
    c={"time":1000,"open":1,"high":2,"low":.5,"close":1.5,"volume":10}
    assert b.update(c)
    assert not b.update({**c,"close":1.6})
    assert len(b)==1 and b.last_close()==1.6
    assert not b.update({**c,"time":900})


def test_structure_risk_has_positive_rr():
    r=calculate_risk(100,2,"LONG",swing_low=98.5)
    assert r and r.stop_loss < 100 and r.tp1 > 100
    assert r.rr1 >= 1.0 and r.rr2 > r.rr1 and r.rr3 > r.rr2


def test_short_structure_risk():
    r=calculate_risk(100,2,"SHORT",swing_high=101.5)
    assert r and r.stop_loss > 100 and r.tp1 < 100


def test_rest_client_filters_open_and_duplicate_candles():
    import asyncio
    import time
    from exchange.coindcx_rest import CoinDCXRest

    async def run():
        client = CoinDCXRest()
        now = int(time.time() * 1000)
        closed_open = (now // 300000) * 300000 - 600000
        open_open = (now // 300000) * 300000
        rows = [
            {"time": closed_open + 1000, "open": 100, "high": 104, "low": 99, "close": 103, "volume": 12},
            {"time": closed_open // 1000, "open": 100, "high": 102, "low": 99, "close": 101, "volume": 10},
            {"time": closed_open, "open": 100, "high": 103, "low": 99, "close": 102, "volume": 11},
            {"time": open_open, "open": 102, "high": 104, "low": 101, "close": 103, "volume": 12},
        ]
        async def fake_get(url, params=None): return rows
        client._get = fake_get
        got = await client.get_candles("B-BTC_USDT", "5m", 10)
        assert len(got) == 1
        assert got[0]["time"] == closed_open
        assert got[0]["close"] == 102
    asyncio.run(run())



def test_macd_hist_history_returns_scalar_histogram_series():
    from core.engine import _macd_hist_history
    values = np.linspace(100.0, 125.0, 90)
    history = _macd_hist_history(values, points=4)
    assert len(history) == 4
    assert all(np.isfinite(value) for value in history)


def test_normal_and_momentum_candle_cursors_are_independent():
    from core.engine import SignalEngine
    engine = SignalEngine(rest=object(), symbols=object(), candles=object())
    engine._last_processed["normal"]["B-BTC_USDT"] = 123
    assert engine._last_processed["momentum"].get("B-BTC_USDT") is None
    assert engine.signals["normal"] is not engine.signals["momentum"]
    assert engine.cooldowns["normal"] is not engine.cooldowns["momentum"]
    assert engine.watchlists["normal"] is not engine.watchlists["momentum"]
    assert engine.normal_scoring is not engine.momentum_scoring


def test_fixed_symbol_universe_preserves_allowlist_and_active_status(monkeypatch):
    import asyncio
    from exchange.symbol_manager import SymbolManager

    # Static allowlist order is retained among liquid members. LOW is
    # intentionally present only in this fixture; OFF is exchange-inactive.
    monkeypatch.setattr(
        SymbolManager, "_load_yaml",
        lambda self, path: (
            ["B-BTC_USDT", "B-ETH_USDT", "B-LOW_USDT"]
            if path.endswith("momentum_symbols.yaml")
            else ["B-BTC_USDT", "B-ETH_USDT"]
        ),
    )

    monkeypatch.setattr("config.settings.normal_min_24h_turnover_usdt", 500.0)
    monkeypatch.setattr("config.settings.momentum_min_24h_turnover_usdt", 100.0)

    class FakeRest:
        async def get_market_details(self):
            return [
                {"pair": "B-BTC_USDT", "base_currency_short_name": "USDT", "target_currency_short_name": "BTC", "status": "active"},
                {"pair": "B-ETH_USDT", "base_currency_short_name": "USDT", "target_currency_short_name": "ETH", "status": "active"},
                {"pair": "B-LOW_USDT", "base_currency_short_name": "USDT", "target_currency_short_name": "LOW", "status": "active"},
                {"pair": "B-OFF_USDT", "base_currency_short_name": "USDT", "target_currency_short_name": "OFF", "status": "inactive"},
            ]

        async def get_ticker(self):
            return [
                {"market": "B-BTC_USDT", "base_volume": "10", "last_price": "100"},
                {"market": "B-ETH_USDT", "base_volume": "20", "last_price": "10"},
                {"market": "B-LOW_USDT", "base_volume": "1", "last_price": "1"},
            ]

    async def run():
        manager = SymbolManager(FakeRest())
        await manager.load_and_validate()
        # Turnover is telemetry only: configured active allowlist members stay eligible.
        assert manager.normal_symbols == ["B-BTC_USDT", "B-ETH_USDT"]
        assert manager.momentum_symbols == ["B-BTC_USDT", "B-ETH_USDT", "B-LOW_USDT"]
        assert manager.liquidity["B-ETH_USDT"] == 200.0
        assert manager.liquidity["B-LOW_USDT"] == 1.0
        assert "B-OFF_USDT" not in manager.valid_pairs
        assert "B-LOW_USDT" in manager.valid_pairs

    asyncio.run(run())


def test_dynamic_symbol_universe_excludes_stablecoin_pairs(monkeypatch):
    import asyncio
    from exchange.symbol_manager import SymbolManager
    monkeypatch.setattr("config.settings.normal_min_24h_turnover_usdt", 1.0)
    monkeypatch.setattr("config.settings.momentum_min_24h_turnover_usdt", 1.0)

    class FakeRest:
        async def get_market_details(self):
            return [
                {"pair": "B-BTC_USDT", "base_currency_short_name": "USDT", "target_currency_short_name": "BTC", "status": "active"},
                {"pair": "B-USDC_USDT", "base_currency_short_name": "USDT", "target_currency_short_name": "USDC", "status": "active"},
                {"pair": "B-DAI_USDT", "base_currency_short_name": "USDT", "target_currency_short_name": "DAI", "status": "active"},
                {"pair": "B-ETH_USDT", "base_currency_short_name": "USDT", "target_currency_short_name": "ETH", "status": "active"},
            ]

        async def get_ticker(self):
            return [
                {"market": "B-BTC_USDT", "base_volume": "10", "last_price": "100"},
                {"market": "B-USDC_USDT", "base_volume": "1000000", "last_price": "1"},
                {"market": "B-DAI_USDT", "base_volume": "500000", "last_price": "1"},
                {"market": "B-ETH_USDT", "base_volume": "20", "last_price": "10"},
            ]

    async def run():
        manager = SymbolManager(FakeRest())
        await manager.load_and_validate()
        assert manager.valid_pairs == {"B-BTC_USDT", "B-ETH_USDT"}
        assert "B-USDC_USDT" not in manager.normal_symbols
        assert "B-USDC_USDT" not in manager.momentum_symbols
        assert "B-DAI_USDT" not in manager.normal_symbols
        assert "B-DAI_USDT" not in manager.momentum_symbols
        assert manager.skipped_reasons["B-USDC_USDT"].startswith("stablecoin-to-USDT pair")
        assert manager.skipped_reasons["B-DAI_USDT"].startswith("stablecoin-to-USDT pair")

    asyncio.run(run())


def test_persistent_signal_claim_blocks_restart_duplicate(tmp_path, monkeypatch):
    import asyncio
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    from database import db as database_db
    from database.db import Base
    from database import models  # noqa: F401
    from database.repositories import claim_signal_once

    async def run():
        test_engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'dedup.db'}")
        old_engine, old_factory = database_db.engine, database_db.SessionLocal
        database_db.engine = test_engine
        database_db.SessionLocal = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
        try:
            async with test_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            signal = {"pair": "B-BTC_USDT", "engine": "normal", "direction": "LONG", "score": 8.0}
            assert await claim_signal_once(signal, 1_800_000, ttl_hours=4)
            # A second engine must not send the same pair/direction/candle.
            momentum_signal = {**signal, "engine": "momentum"}
            assert not await claim_signal_once(momentum_signal, 1_800_000, ttl_hours=4)
            # A genuinely new candle is not blocked by the previous candle's
            # deduplication record, even while its record remains unexpired.
            assert await claim_signal_once(signal, 2_100_000, ttl_hours=4)
        finally:
            database_db.engine, database_db.SessionLocal = old_engine, old_factory
            await test_engine.dispose()

    asyncio.run(run())


def test_candle_age_is_measured_from_close_not_open():
    from core.candle_timing import candle_age_seconds
    # 5m candle opened at 12:00 and closed at 12:05; now is 12:06:30.
    assert candle_age_seconds(12 * 60 * 60 * 1000 + 6 * 60 * 1000 + 30 * 1000,
                              12 * 60 * 60 * 1000 + 0, 5 * 60 * 1000) == 90.0


def test_15m_context_never_uses_a_candle_newer_than_5m_decision():
    from core.candle_timing import latest_asof_candle_index
    # Entry 5m candle closes at 12:35. Context 12:30-12:45 is future data;
    # the latest eligible context is 12:15-12:30.
    times = [12 * 60 * 60 * 1000 + minute * 60 * 1000 for minute in (0, 15, 30)]
    idx = latest_asof_candle_index(times, 15 * 60 * 1000,
                                   12 * 60 * 60 * 1000 + 35 * 60 * 1000)
    assert idx == 1


def test_15m_context_rejects_missing_or_too_old_history():
    from core.candle_timing import latest_asof_candle_index
    base = 12 * 60 * 60 * 1000
    assert latest_asof_candle_index([base + 30 * 60 * 1000], 15 * 60 * 1000,
                                    base + 35 * 60 * 1000) is None
    assert latest_asof_candle_index([base], 15 * 60 * 1000,
                                    base + 35 * 60 * 1000) is None


def test_rsi_is_neutral_for_a_flat_market():
    from indicators.rsi import rsi
    assert rsi(np.full(40, 100.0), 14) == 50.0


def test_entry_price_must_remain_inside_planned_zone():
    from core.risk_calculator import is_entry_price_actionable
    assert is_entry_price_actionable(100.05, 99.85, 100.10)
    assert not is_entry_price_actionable(100.20, 99.85, 100.10)
    assert not is_entry_price_actionable(0, 99.85, 100.10)


def test_ticker_price_lookup_matches_coin_dcx_pair_aliases():
    import asyncio
    from exchange.coindcx_rest import CoinDCXRest

    async def run():
        client = CoinDCXRest()
        import time
        now_ms = int(time.time() * 1000)
        async def fake_ticker():
            return [
                {"market": "B-ETH_USDT", "last_price": "2500.5", "timestamp": now_ms},
                {"market": "B-BTC_USDT", "last_price": "65000", "timestamp": now_ms},
            ]
        client.get_ticker = fake_ticker
        assert await client.get_last_price("B-BTC_USDT") == 65000.0
        assert await client.get_last_price("B-MISSING_USDT") is None
    asyncio.run(run())


def test_scan_wait_aligns_to_next_completed_candle_boundary():
    from core.candle_timing import seconds_until_next_candle_boundary
    interval = 5 * 60 * 1000
    grace = 3 * 1000
    assert seconds_until_next_candle_boundary(12*60*60*1000 + 2*60*1000, interval, grace) == 183.0
    assert seconds_until_next_candle_boundary(12*60*60*1000 + 5*60*1000 + 1000, interval, grace) == 2.0
    assert seconds_until_next_candle_boundary(12*60*60*1000 + 5*60*1000 + 4000, interval, grace) == 299.0


def test_rest_client_rejects_historical_stale_market_candles():
    import asyncio
    from exchange.coindcx_rest import CoinDCXRest
    async def run():
        client = CoinDCXRest()
        rows = [{"time": 1_708_397_700_000, "open": 100, "high": 102, "low": 99,
                 "close": 101, "volume": 10}]
        async def fake_get(url, params=None): return rows
        client._get = fake_get
        assert await client.get_candles("B-XMR_USDT", "5m", 10) == []
    asyncio.run(run())



def test_stale_market_quarantine_filters_both_engine_universes_and_restores(monkeypatch):
    import asyncio
    import time
    from exchange.symbol_manager import SymbolManager
    from config import settings

    # Include BAD explicitly in both static lists; SOL is the next eligible
    # configured symbol and backfills a quarantined slot within the static list.
    monkeypatch.setattr(
        SymbolManager, "_load_yaml",
        lambda self, path: ["B-BAD_USDT", "B-BTC_USDT", "B-ETH_USDT", "B-SOL_USDT"],
    )

    old_normal_floor = settings.normal_min_24h_turnover_usdt
    old_momentum_floor = settings.momentum_min_24h_turnover_usdt
    settings.normal_min_24h_turnover_usdt = 1.0
    settings.momentum_min_24h_turnover_usdt = 1.0

    class FakeRest:
        def __init__(self):
            self.stale = {"B-BAD_USDT": {"5m": 2, "15m": 1}}
            self.healthy = False

        async def get_market_details(self):
            return [
                {"pair": f"B-{name}_USDT", "base_currency_short_name": "USDT",
                 "target_currency_short_name": name, "status": "active"}
                for name in ("BTC", "ETH", "SOL", "BAD")
            ]

        async def get_ticker(self):
            return [
                {"market": f"B-{name}_USDT", "base_volume": str(volume), "last_price": "10"}
                for name, volume in (("BAD", 1000), ("BTC", 900), ("ETH", 800), ("SOL", 700))
            ]

        def get_stale_market_timeframes(self):
            return {pair: dict(value) for pair, value in self.stale.items()}

        async def get_candles(self, pair, interval, limit=100):
            if not self.healthy:
                return []
            return [{"time": i, "open": 1, "high": 2, "low": .5, "close": 1.5, "volume": 10}
                    for i in range(100 if interval == "5m" else 110)]

    async def run():
        old_normal, old_momentum = settings.normal_symbol_limit, settings.momentum_symbol_limit
        settings.normal_symbol_limit, settings.momentum_symbol_limit = 2, 3
        try:
            rest = FakeRest()
            manager = SymbolManager(rest)
            await manager.load_and_validate()
            assert manager.normal_symbols == ["B-BAD_USDT", "B-BTC_USDT"]
            assert manager.momentum_symbols == ["B-BAD_USDT", "B-BTC_USDT", "B-ETH_USDT"]

            manager._refresh_quarantine_from_rest()
            manager._select_active_symbols()
            assert "B-BAD_USDT" not in manager.normal_symbols
            assert "B-BAD_USDT" not in manager.momentum_symbols
            assert len(manager.normal_symbols) == 2
            # Backfill comes only from the remaining configured static allowlist.
            assert manager.momentum_symbols == ["B-BTC_USDT", "B-ETH_USDT", "B-SOL_USDT"]
            assert "B-BAD_USDT" in manager.skipped_reasons

            # Simulate the 15-minute retry deadline and recovery of both feeds.
            rest.stale.clear()
            rest.healthy = True
            manager._quarantined["B-BAD_USDT"]["retry_at"] = time.time() - 1
            await manager.refresh_universes()
            assert "B-BAD_USDT" not in manager._quarantined
            assert manager.normal_symbols[0] == "B-BAD_USDT"
            assert manager.momentum_symbols == ["B-BAD_USDT", "B-BTC_USDT", "B-ETH_USDT"]
            assert "B-BAD_USDT" not in manager.skipped_reasons
        finally:
            settings.normal_symbol_limit, settings.momentum_symbol_limit = old_normal, old_momentum
            settings.normal_min_24h_turnover_usdt = old_normal_floor
            settings.momentum_min_24h_turnover_usdt = old_momentum_floor

    asyncio.run(run())


def test_stale_market_quarantine_requires_repeated_primary_timeframe_observations():
    from exchange.symbol_manager import SymbolManager

    class FakeRest:
        def __init__(self):
            self.stale = {
                "B-ONE_USDT": {"5m": 1},
                "B-TWO_USDT": {"5m": 1, "15m": 3},
            }
        def get_stale_market_timeframes(self):
            return self.stale

    manager = SymbolManager(FakeRest())
    manager._refresh_quarantine_from_rest()
    assert "B-ONE_USDT" not in manager._quarantined
    assert "B-TWO_USDT" not in manager._quarantined
    manager.rest.stale["B-ONE_USDT"]["5m"] = 2
    manager._refresh_quarantine_from_rest()
    assert "B-ONE_USDT" in manager._quarantined


def test_rest_client_rejects_future_open_candles(monkeypatch):
    import asyncio
    from exchange.coindcx_rest import CoinDCXRest

    async def run():
        client = CoinDCXRest()
        now_ms = 1_800_000_000_000
        interval_ms = 5 * 60 * 1000
        future_open = ((now_ms // interval_ms) + 1) * interval_ms
        rows = [{
            "time": future_open, "open": 100, "high": 102,
            "low": 99, "close": 101, "volume": 10,
        }]

        async def fake_get(url, params=None):
            return rows

        client._get = fake_get
        monkeypatch.setattr("exchange.coindcx_rest.time.time", lambda: now_ms / 1000)
        assert await client.get_candles("B-BTC_USDT", "5m", 10) == []

    asyncio.run(run())


def test_score_disposition_keeps_watchlist_below_signal_threshold():
    from core.engine import _score_disposition

    # Momentum watchlist floor is 5.5; scores below the floor are rejected.
    # Scores between the watchlist floor and signal threshold remain watchlist-only.
    assert _score_disposition(5.49, 7.0) == "REJECT"
    assert _score_disposition(5.5, 7.0) == "WATCHLIST"
    assert _score_disposition(5.99, 7.0) == "WATCHLIST"
    assert _score_disposition(6.0, 7.0) == "WATCHLIST"
    assert _score_disposition(6.99, 7.0) == "WATCHLIST"
    assert _score_disposition(7.0, 7.0) == "WATCHLIST"
    assert _score_disposition(7.99, 8.0) == "WATCHLIST"
    assert _score_disposition(8.0, 8.0) == "SIGNAL"


def test_candle_evaluation_diagnostic_uses_primary_candle_and_gate_reasons(monkeypatch):
    from core.engine import SignalEngine
    from utils import logger as logger_module

    messages = []
    monkeypatch.setattr(logger_module.logger, "info", lambda message: messages.append(message))
    engine = SignalEngine(rest=object(), symbols=object(), candles=object())
    engine._log_candle_evaluation(
        pair="B-ZEC_USDT", engine="normal", direction="SHORT",
        candle_ts=1_800_000_000_000, candle_close_ms=1_800_000_300_000,
        confirm_candle_time_ms=None,
        indicators={
            "close": 100.0, "ema9": 99.0, "ema21": 101.0,
            "ema50": 102.0, "vwap": 101.0, "rsi": 42.0,
            "adx": 31.0, "plus_di": 12.0, "minus_di": 25.0,
            "macd_hist": -1.2, "rvol": 0.8, "atr_pct": 0.4,
            "ema_extension_atr": 0.5,
            "adx_rising": True, "directional_candle": True,
        },
        decision="REJECT", failed=["rsi_direction"], strategy="TREND_CONTINUATION",
        score=6.8, score_parts="EMA Trend:1/1,RSI:1/1",
    )
    assert len(messages) == 1
    message = messages[0]
    assert "pair=B-ZEC_USDT" in message
    assert "candle_close_ms=1800000300000" in message
    assert "confirm_15m" not in message
    assert "failed=rsi_direction" in message
    assert "score_parts=[EMA Trend:1/1,RSI:1/1]" in message



def test_scan_loop_starts_normal_and_momentum_as_independent_tasks(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from core.engine import SignalEngine
    from config import settings

    async def run():
        engine = SignalEngine(rest=object(), symbols=SimpleNamespace(refresh_universes=None), candles=object())
        called = []
        async def refresh():
            called.append("refresh")
        async def engine_loop(app, name):
            if name == "normal":
                await asyncio.sleep(0.02)
            called.append(name)

        monkeypatch.setattr(engine.symbols, "refresh_universes", refresh)
        monkeypatch.setattr(engine, "_run_engine_loop", engine_loop)
        old_normal, old_momentum = settings.enable_normal_engine, settings.enable_momentum_engine
        settings.enable_normal_engine = True
        settings.enable_momentum_engine = True
        try:
            await engine.run_scan_loop(SimpleNamespace())
            assert called[0] == "refresh"
            assert called.index("momentum") < called.index("normal")
        finally:
            settings.enable_normal_engine = old_normal
            settings.enable_momentum_engine = old_momentum

    asyncio.run(run())


def test_persistent_dedup_claim_prunes_expired_rows(tmp_path):
    import asyncio
    from datetime import datetime, timedelta
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    from database import db as database_db
    from database.db import Base
    from database import models  # noqa: F401
    from database.models import SignalDedupRecord
    from database.repositories import claim_signal_once

    async def run():
        test_engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'dedup_cleanup.db'}")
        old_engine, old_factory = database_db.engine, database_db.SessionLocal
        database_db.engine = test_engine
        database_db.SessionLocal = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
        try:
            async with test_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            async with database_db.SessionLocal() as session:
                session.add(SignalDedupRecord(
                    dedup_key="expired:SHORT:1", candle_time_ms=1, payload="{}",
                    created_at=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=5),
                    expires_at=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=1),
                ))
                await session.commit()
            signal = {"pair": "B-ETH_USDT", "engine": "normal", "direction": "LONG", "score": 8.0}
            assert await claim_signal_once(signal, 2_000_000, ttl_hours=4)
            async with database_db.SessionLocal() as session:
                result = await session.execute(select(SignalDedupRecord).where(
                    SignalDedupRecord.dedup_key == "expired:SHORT:1"
                ))
                assert result.scalar_one_or_none() is None
        finally:
            database_db.engine, database_db.SessionLocal = old_engine, old_factory
            await test_engine.dispose()

    asyncio.run(run())


def test_ticker_price_lookup_rejects_stale_or_untimestamped_prices():
    import asyncio
    import time
    from exchange.coindcx_rest import CoinDCXRest

    async def run():
        client = CoinDCXRest()
        old_ms = int(time.time() * 1000) - 60_000
        async def stale_ticker():
            return [{"market": "B-BTC_USDT", "last_price": "65000", "timestamp": old_ms}]
        client.get_ticker = stale_ticker
        assert await client.get_last_price("B-BTC_USDT") is None

        async def missing_timestamp_ticker():
            return [{"market": "B-BTC_USDT", "last_price": "65000"}]
        client.get_ticker = missing_timestamp_ticker
        assert await client.get_last_price("B-BTC_USDT") is None

    asyncio.run(run())


def test_active_signal_persistence_restore_and_terminal_update(tmp_path):
    import asyncio
    from datetime import datetime, timezone, timedelta
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    from database import db as database_db
    from database.db import Base
    from database import models  # noqa: F401
    from database.repositories import save_signal, load_active_signals, update_signal_record
    from core.signal_manager import SignalManager

    async def run():
        test_engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'signals.db'}")
        old_engine, old_factory = database_db.engine, database_db.SessionLocal
        database_db.engine = test_engine
        database_db.SessionLocal = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
        try:
            async with test_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            now = datetime.now(timezone.utc)
            signal = {
                "pair": "B-BTC_USDT", "engine": "normal", "direction": "LONG",
                "score": 8.0, "grade": "STRONG", "price": 100.0,
                "status": "active", "created_at": now.isoformat(),
                "expires_at": (now + timedelta(hours=3)).isoformat(),
                "risk": {"stop_loss": 95.0, "tp1": 105.0, "tp2": 110.0, "tp3": 115.0},
            }
            assert await save_signal(signal)
            restored = await load_active_signals()
            assert len(restored) == 1 and restored[0]["pair"] == "B-BTC_USDT"
            manager = SignalManager()
            assert manager.restore(restored[0])
            assert manager.count() == 1
            assert await update_signal_record(restored[0], "closed_tp")
            assert await load_active_signals() == []
        finally:
            database_db.engine, database_db.SessionLocal = old_engine, old_factory
            await test_engine.dispose()

    asyncio.run(run())


def test_tp_outcome_is_persisted_only_once_per_target(monkeypatch):
    import asyncio
    from core.engine import SignalEngine
    from database import repositories

    async def run():
        engine = SignalEngine(rest=object(), symbols=object(), candles=object())
        signal = {
            "pair": "B-BTC_USDT", "engine": "normal", "direction": "LONG",
            "score": 8.0, "risk": {
                "stop_loss": 95.0, "tp1": 105.0, "tp2": 110.0, "tp3": 115.0,
            },
        }
        assert engine.signals["normal"].add(signal)
        calls = []
        async def persist(item, status):
            calls.append((status, item.get("tp1_hit")))
            return True
        monkeypatch.setattr(repositories, "update_signal_record", persist)
        await engine._update_signal_outcomes("B-BTC_USDT", "normal", high=106.0, low=99.0)
        await engine._update_signal_outcomes("B-BTC_USDT", "normal", high=106.0, low=99.0)
        assert calls == [("active", True)]

    asyncio.run(run())


def test_liquidity_universe_preserves_static_order_when_24h_volume_changes(monkeypatch):
    import asyncio
    from exchange.symbol_manager import SymbolManager
    monkeypatch.setattr("config.settings.normal_min_24h_turnover_usdt", 1.0)
    monkeypatch.setattr("config.settings.momentum_min_24h_turnover_usdt", 1.0)

    # Both engines retain the configured static order even when live volume changes.
    monkeypatch.setattr(
        SymbolManager, "_load_yaml",
        lambda self, path: ["B-BTC_USDT", "B-ETH_USDT", "B-SOL_USDT"],
    )

    class FakeRest:
        def __init__(self):
            self.volumes = {"BTC": 300, "ETH": 200, "SOL": 100}
        async def get_market_details(self):
            return [
                {"pair": f"B-{name}_USDT", "base_currency_short_name": "USDT",
                 "target_currency_short_name": name, "coindcx_name": f"{name}USDT",
                 "status": "active"}
                for name in ("BTC", "ETH", "SOL")
            ]
        async def get_ticker(self):
            return [
                {"market": f"{name}USDT", "base_volume": str(volume), "last_price": "10"}
                for name, volume in self.volumes.items()
            ]

    async def run():
        rest = FakeRest()
        manager = SymbolManager(rest)
        await manager.load_and_validate()
        assert manager.normal_symbols == ["B-BTC_USDT", "B-ETH_USDT", "B-SOL_USDT"]
        rest.volumes = {"BTC": 50, "ETH": 500, "SOL": 250}
        manager._last_liquidity_refresh = 0
        await manager.refresh_universes()
        assert manager.normal_symbols == ["B-BTC_USDT", "B-ETH_USDT", "B-SOL_USDT"]
        assert manager.momentum_symbols == ["B-BTC_USDT", "B-ETH_USDT", "B-SOL_USDT"]

    asyncio.run(run())


def test_restore_marks_payload_expired_even_if_db_row_is_recent(tmp_path):
    import asyncio
    import json
    from datetime import datetime, timezone, timedelta
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    from database import db as database_db
    from database.db import Base
    from database import models  # noqa: F401
    from database.models import SignalRecord
    from database.repositories import load_active_signals

    async def run():
        test_engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'expired_signal.db'}")
        old_engine, old_factory = database_db.engine, database_db.SessionLocal
        database_db.engine = test_engine
        database_db.SessionLocal = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
        try:
            async with test_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            now = datetime.now(timezone.utc)
            expired_payload = {
                "pair": "B-ETH_USDT", "engine": "normal", "direction": "SHORT",
                "created_at": (now - timedelta(hours=2)).isoformat(),
                "expires_at": (now - timedelta(minutes=1)).isoformat(),
            }
            async with database_db.SessionLocal() as session:
                session.add(SignalRecord(
                    pair="B-ETH_USDT", engine="normal", direction="SHORT", score=7.5,
                    grade="SIGNAL", price=100.0, status="active",
                    payload=json.dumps(expired_payload), created_at=datetime.now(timezone.utc).replace(tzinfo=None),
                ))
                await session.commit()
            assert await load_active_signals() == []
            async with database_db.SessionLocal() as session:
                result = await session.execute(select(SignalRecord))
                assert result.scalar_one().status == "expired"
        finally:
            database_db.engine, database_db.SessionLocal = old_engine, old_factory
            await test_engine.dispose()

    asyncio.run(run())


def test_liquidity_refresh_retains_previous_universe_when_ticker_aliases_do_not_match():
    import asyncio
    from exchange.symbol_manager import SymbolManager

    class FakeRest:
        def __init__(self):
            self.rows = [
                {"market": "BTCUSDT", "volume": "300", "last_price": "10"},
                {"market": "ETHUSDT", "volume": "200", "last_price": "10"},
            ]
        async def get_market_details(self):
            return [
                {"pair": f"B-{name}_USDT", "base_currency_short_name": "USDT",
                 "target_currency_short_name": name, "coindcx_name": f"{name}USDT",
                 "status": "active"}
                for name in ("BTC", "ETH")
            ]
        async def get_ticker(self):
            return self.rows

    async def run():
        rest = FakeRest()
        manager = SymbolManager(rest)
        await manager.load_and_validate()
        before = list(manager.normal_symbols)
        rest.rows = [{"market": "UNRELATEDUSDT", "volume": "999", "last_price": "1"}]
        manager._last_liquidity_refresh = 0
        await manager.refresh_universes()
        assert manager.normal_symbols == before

    asyncio.run(run())


def test_scan_loop_waits_for_fresh_boundary_before_first_engine_scan(monkeypatch):
    import asyncio
    from core.engine import SignalEngine
    from config import settings

    class FakeSymbols:
        async def refresh_universes(self):
            events.append("refresh")

    async def run():
        engine = SignalEngine(rest=object(), symbols=FakeSymbols(), candles=object())

        async def wait_for_boundary():
            events.append("boundary")

        engine._wait_for_next_primary_boundary = wait_for_boundary
        old_normal = settings.enable_normal_engine
        old_momentum = settings.enable_momentum_engine
        settings.enable_normal_engine = False
        settings.enable_momentum_engine = False
        try:
            await engine.run_scan_loop(app=None)
        finally:
            settings.enable_normal_engine = old_normal
            settings.enable_momentum_engine = old_momentum

    events = []
    asyncio.run(run())
    assert events == ["refresh", "boundary"]

def test_watchlist_refresh_does_not_extend_original_setup_expiry():
    from datetime import datetime, timezone, timedelta
    from core.watchlist_manager import WatchlistManager

    manager = WatchlistManager(ttl_minutes=60)
    item = {"pair": "B-BTC_USDT", "engine": "momentum", "direction": "LONG", "score": 5.5}
    manager.add(item)
    key = manager._key(item["pair"], item["engine"], item["direction"])
    original_expiry = manager._entries[key]["expires_at"]
    manager.add({**item, "score": 6.0, "price": 101.0})
    assert manager._entries[key]["expires_at"] == original_expiry
    assert manager._entries[key]["score"] == 6.0


def test_normal_engine_has_no_macd_gate():
    from core.engine import _normal_threshold_gates

    base = {
        "ema9": 102.0, "ema21": 100.0, "close": 103.0, "vwap": 101.0,
        "adx": 30.0, "confirm_15m": True, "rsi": 58.0, "atr": 0.5,
        "directional_candle": True,
    }
    # Missing or opposing MACD data must not affect Normal Engine approval.
    without_macd = _normal_threshold_gates(base, "LONG", 25.0)
    opposing_macd = _normal_threshold_gates({**base, "macd_hist": -999.0}, "LONG", 25.0)
    assert len(without_macd) == 5
    assert all(without_macd.values())
    assert opposing_macd == without_macd


def test_normal_scoring_has_only_indicator_components_without_15m_context():
    import pytest
    from core.scoring_engine import ScoringEngine

    indicators = {
        "ema9": 102.0, "ema21": 100.0, "close": 103.0, "vwap": 101.0,
        "adx": 35.0, "rsi": 60.0, "atr": 0.8, "confirm_15m": True,
        "macd_hist": -999.0,
    }
    result = ScoringEngine().score_normal(indicators, "LONG")
    assert len(result.components) == 5
    assert sum(component.max_points for component in result.components) == pytest.approx(8.5)
    assert result.total == pytest.approx(8.5)


def test_confirmed_swing_levels_ignore_unconfirmed_last_two_bars():
    from core.engine import _confirmed_swing_levels

    highs = [10, 11, 12, 11, 10, 11, 13, 12, 11, 12]
    lows =  [8, 7,  6,  7,  8,  7,  5,  6,  7,  6]
    swing_high, swing_low = _confirmed_swing_levels(highs, lows, lookback=20, left=2, right=2)
    assert swing_high == 13.0
    assert swing_low == 5.0


def test_risk_calculator_ignores_legacy_execution_cost_settings():
    from core.risk_calculator import calculate_risk

    no_cost = calculate_risk(100, 2, "LONG", swing_low=98.5, cost_bps=0)
    high_cost = calculate_risk(
        100, 2, "LONG", swing_low=98.5, cost_bps=100, min_net_rr=10,
    )
    assert no_cost is not None and high_cost is not None
    assert high_cost == no_cost
    assert no_cost.rr1 == 2.0


def test_candle_buffer_rejects_nonfinite_and_inconsistent_ohlc():
    buffer = CandleBuffer(10)
    valid = {"time": 1000, "open": 10, "high": 12, "low": 9, "close": 11, "volume": 5}
    assert buffer.update(valid)
    assert not buffer.update({**valid, "time": 2000, "close": float("nan")})
    assert not buffer.update({**valid, "time": 3000, "high": 8})
    assert not buffer.update({**valid, "time": 4000, "volume": -1})
    assert len(buffer) == 1
    assert buffer.last_close() == 11.0


def test_futures_turnover_estimate_uses_base_volume_times_last_price():
    from exchange.symbol_manager import SymbolManager

    manager = SymbolManager(object())
    values = manager._ticker_liquidity([
        {"market": "B-ETH_USDT", "base_volume": "25000", "last_price": "3.2"},
        {"market": "B-LOW_USDT", "v": "100", "ls": "0.5"},
        # A generic volume without an LTP is not treated as USDT turnover.
        {"market": "B-UNKNOWN_USDT", "volume": "999999999"},
    ])
    assert values["ETHUSDT"] == 80000.0
    assert values["LOWUSDT"] == 50.0
    assert "UNKNOWNUSDT" not in values


def test_static_universe_fails_closed_when_turnover_data_is_missing(monkeypatch):
    import asyncio
    from exchange.symbol_manager import SymbolManager

    monkeypatch.setattr("config.settings.normal_min_24h_turnover_usdt", 1000.0)
    monkeypatch.setattr("config.settings.momentum_min_24h_turnover_usdt", 500.0)
    monkeypatch.setattr(
        SymbolManager, "_load_yaml",
        lambda self, path: ["B-BTC_USDT"],
    )

    class FakeRest:
        async def get_market_details(self):
            return [{
                "pair": "B-BTC_USDT", "base_currency_short_name": "USDT",
                "target_currency_short_name": "BTC", "status": "active",
            }]

        async def get_ticker(self):
            return []

    async def run():
        manager = SymbolManager(FakeRest())
        await manager.load_and_validate()
        assert manager.normal_symbols == ["B-BTC_USDT"]
        assert manager.momentum_symbols == ["B-BTC_USDT"]
        assert manager.liquidity["B-BTC_USDT"] == 0.0

    asyncio.run(run())



def test_intrabar_momentum_rvol_uses_live_candle_volume(monkeypatch):
    from core.engine import SignalEngine
    from config import settings

    class CandleStub:
        def get_arrays(self, pair, timeframe):
            return {
                "close": np.full(30, 100.0),
                "volume": np.full(30, 10.0),
            }

    class SymbolStub:
        def display_name(self, pair):
            return "BTCUSDT"

    engine = SignalEngine.__new__(SignalEngine)
    engine.candles = CandleStub()
    engine.symbols = SymbolStub()
    from core.watchlist_manager import WatchlistManager

    engine.watchlists = {"momentum": WatchlistManager()}
    monkeypatch.setattr(settings, "candle_primary_tf", "5m")

    engine.on_intrabar_candle("B-BTC_USDT", "5m", {
        "open": 100.0, "high": 102.0, "low": 99.0,
        "close": 101.0, "volume": 100.0,
    })

    entries = engine.watchlists["momentum"].list_all("momentum")
    assert len(entries) == 1
    assert entries[0]["indicators"]["live_rvol"] == 10.0



def test_candle_builder_keeps_existing_buffer_when_rest_seed_has_no_valid_rows():
    builder = CandleBuilder()
    original = {
        "time": 1_000, "open": 100, "high": 102,
        "low": 99, "close": 101, "volume": 10,
    }
    builder.seed("B-BTC_USDT", "5m", [original])
    buffer = builder.get_buffer("B-BTC_USDT", "5m")

    builder.seed("B-BTC_USDT", "5m", [])
    assert len(buffer) == 1
    assert buffer.last_close() == 101.0

    builder.seed("B-BTC_USDT", "5m", [
        {**original, "time": 2_000, "close": float("nan")},
        {"time": 3_000, "open": 100},
    ])
    assert len(buffer) == 1
    assert buffer.last_close() == 101.0


def test_candle_builder_replaces_buffer_with_valid_rest_refresh():
    builder = CandleBuilder()
    builder.seed("B-BTC_USDT", "5m", [
        {"time": 1_000, "open": 100, "high": 102, "low": 99, "close": 101, "volume": 10},
        {"time": 2_000, "open": 101, "high": 103, "low": 100, "close": 102, "volume": 11},
    ])
    builder.seed("B-BTC_USDT", "5m", [
        {"time": 3_000, "open": 102, "high": 104, "low": 101, "close": 103, "volume": 12},
    ])
    buffer = builder.get_buffer("B-BTC_USDT", "5m")
    assert len(buffer) == 1
    assert buffer.candles[0]["time"] == 3_000
    assert buffer.last_close() == 103.0

def test_runtime_config_rejects_non_finite_safety_thresholds(monkeypatch):
    import pytest
    from app.lifespan import validate_runtime_config

    monkeypatch.setattr("config.settings.signal_threshold", float("nan"))
    with pytest.raises(ValueError, match="SIGNAL_THRESHOLD must be a finite number"):
        validate_runtime_config()


def test_runtime_config_ignores_deprecated_execution_quality_settings(monkeypatch):
    from app.lifespan import validate_runtime_config

    monkeypatch.setattr("config.settings.futures_round_trip_cost_bps", -1.0)
    monkeypatch.setattr("config.settings.futures_max_spread_bps", 0.0)
    monkeypatch.setattr("config.settings.futures_min_side_depth_usdt", -1.0)
    validate_runtime_config()


def test_runtime_config_rejects_out_of_range_orb_adx(monkeypatch):
    import pytest
    from app.lifespan import validate_runtime_config

    monkeypatch.setattr("config.settings.orb_adx_min", 101.0)
    with pytest.raises(ValueError, match="ORB_ADX_MIN must be between 0 and 100"):
        validate_runtime_config()

