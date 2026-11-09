"""Regression coverage for Delta candle feeds, scanner readiness and Telegram outbox."""
import asyncio
import json
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest


def test_momentum_allowlist_has_100_unique_listed_replacements():
    from pathlib import Path
    import yaml

    path = Path(__file__).resolve().parents[1] / "config" / "momentum_symbols.yaml"
    symbols = yaml.safe_load(path.read_text(encoding="utf-8"))["symbols"]
    assert len(symbols) == 100
    assert len(set(symbols)) == 100
    assert not {"B-BONK_USDT", "B-KORU_USDT", "B-LITE_USDT"}.intersection(symbols)
    assert {"B-1000BONK_USDT", "B-FARTCOIN_USDT", "B-ORDI_USDT"}.issubset(symbols)


def test_websocket_tracks_accepted_updates_per_symbol_and_timeframe(monkeypatch):
    from exchange.delta_ws import DeltaFuturesStream
    from exchange.candle_builder import CandleBuilder
    from exchange.delta_rest import DeltaExchangeRest

    now_ms = (int(time.time() * 1000) // 300_000) * 300_000 + 1_000
    monkeypatch.setattr("exchange.delta_ws.time.time", lambda: now_ms / 1000)

    class FakeRest:
        _timestamp_ms = staticmethod(DeltaExchangeRest._timestamp_ms)

        @staticmethod
        def canonical_pair_for_symbol(symbol):
            return "B-BTC_USDT" if symbol == "BTCUSD" else ""

        @staticmethod
        def set_ws_prices(*args, **kwargs):
            return None

    stream = DeltaFuturesStream(CandleBuilder(), FakeRest())
    message = {
        "type": "candlestick_5m",
        "res": "5m",
        "sy": "BTCUSD",
        "ts": now_ms * 1000,
        "o": "100", "h": "102", "l": "99", "c": "101", "v": "25",
    }
    stream._handle_candlestick(message)
    assert stream.symbol_feed_age_seconds["B-BTC_USDT"]["5m"] == 0.0
    assert len(stream.candles.get_buffer("B-BTC_USDT", "5m")) == 1

    invalid = {**message, "sy": "UNKNOWNUSD", "ts": (now_ms + 60_000) * 1000}
    stream._handle_candlestick(invalid)
    assert set(stream._last_symbol_update_ms) == {"B-BTC_USDT"}


def test_websocket_reconnect_backoff_grows_and_is_capped():
    from exchange.delta_ws import DeltaFuturesStream

    assert [DeltaFuturesStream.reconnect_delay(n) for n in range(1, 8)] == [
        1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0,
    ]


@pytest.mark.asyncio
async def test_delta_rest_candles_are_completed_aligned_and_report_freshness(monkeypatch):
    from exchange.delta_rest import DeltaExchangeRest

    now_ms = (int(time.time() * 1000) // 900_000) * 900_000 + 5_000
    monkeypatch.setattr("exchange.delta_rest.time.time", lambda: now_ms / 1000)
    client = DeltaExchangeRest()
    client._canonical_to_delta["B-BTC_USDT"] = "BTCUSD"

    rows_by_interval = {}
    for interval, duration in (("5m", 300_000), ("15m", 900_000)):
        latest_closed_open = (now_ms // duration) * duration - duration
        rows_by_interval[interval] = [
            {
                "time": (latest_closed_open - (59 - i) * duration) / 1000,
                "open": 100, "high": 102, "low": 99, "close": 101, "volume": 10,
            }
            for i in range(60)
        ]

    async def fake_get(url, params=None):
        return {"result": rows_by_interval[params["resolution"]]}

    client._get = fake_get
    for interval, duration in (("5m", 300_000), ("15m", 900_000)):
        candles = await client.get_candles("B-BTC_USDT", interval, limit=100)
        assert len(candles) == 60
        assert all(candle["time"] % duration == 0 for candle in candles)
        assert candles[-1]["time"] + duration + 3_000 <= now_ms
        assert candles[-1]["time"] == rows_by_interval[interval][-1]["time"] * 1000

    status = client.get_candle_feed_status()["B-BTC_USDT"]
    assert status["5m"]["status"] == "fresh"
    assert status["15m"]["status"] == "fresh"
    assert status["5m"]["last_completed_close_ms"] == (
        status["5m"]["last_completed_open_ms"] + 300_000
    )


@pytest.mark.asyncio
async def test_health_rejects_incomplete_universe_and_missing_per_symbol_candles(monkeypatch):
    from app.health import health
    from config import settings

    monkeypatch.setattr(type(settings), "is_production", property(lambda self: True))
    monkeypatch.setattr(settings, "enable_normal_engine", True)
    monkeypatch.setattr(settings, "enable_momentum_engine", True)
    monkeypatch.setattr(settings, "normal_symbol_limit", 1)
    monkeypatch.setattr(settings, "momentum_symbol_limit", 2)

    now = datetime.now(timezone.utc)
    state = {
        "database": "postgres",
        "normal_engine": "running",
        "momentum_engine": "running",
        "valid_symbols_normal": 1,
        "valid_symbols_momentum": 1,
        "scan_task_expected": True,
        "scan_task_status": "running",
        "retest_task_status": "running",
        "telegram_delivery_task_expected": True,
        "telegram_delivery_task_status": "running",
        "last_scan_normal": now.isoformat(),
        "last_scan_momentum": now.isoformat(),
    }
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        bot_state=state,
        telegram=SimpleNamespace(bot=object(), admin_ids=[123]),
        rest=SimpleNamespace(get_candle_feed_status=lambda: {}),
        symbol_mgr=SimpleNamespace(normal_symbols=["B-BTC_USDT"], momentum_symbols=["B-BTC_USDT", "B-ETH_USDT"]),
        websocket=SimpleNamespace(status="connected", reconnect_count=0, symbol_feed_age_seconds={}),
        start_time=now,
    )))
    response = await health(request)
    assert response.status_code == 503
    body = json.loads(response.body)
    assert "momentum_universe_incomplete" in body["reasons"]
    assert "per_symbol_candle_data_not_ready" in body["reasons"]


@pytest.mark.asyncio
async def test_telegram_outbox_duplicate_enqueue_and_delivery_are_idempotent(tmp_path):
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    from database import db as database_db
    from database.db import Base
    from database import models  # noqa: F401
    from database.models import SignalDeliveryRecord
    from database.repositories import enqueue_signal_deliveries
    from tgbot.delivery_worker import TelegramDeliveryWorker

    test_engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'outbox.db'}")
    old_engine, old_factory = database_db.engine, database_db.SessionLocal
    database_db.engine = test_engine
    database_db.SessionLocal = async_sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    try:
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        signal = {
            "pair": "B-BTC_USDT", "engine": "normal", "direction": "LONG",
            "candle_time_ms": 1_800_000_000_000, "signal_age_seconds": 0,
            "price": 100.0,
        }
        assert await enqueue_signal_deliveries(signal, [123]) == 1
        # INSERT DO NOTHING must still report the existing durable item as accepted.
        assert await enqueue_signal_deliveries(signal, [123]) == 1

        class FakeTelegram:
            bot = object()
            admin_ids = [123]

            def __init__(self):
                self.sent = []

            async def send_signal_to_chat(self, payload, chat_id):
                self.sent.append((payload["pair"], chat_id))

        telegram = FakeTelegram()
        worker = TelegramDeliveryWorker(telegram)
        assert await worker.run_once() == 1
        assert await worker.run_once() == 0
        assert telegram.sent == [("B-BTC_USDT", 123)]

        async with database_db.SessionLocal() as session:
            rows = (await session.execute(select(SignalDeliveryRecord))).scalars().all()
            assert len(rows) == 1
            assert rows[0].status == "delivered"
            assert rows[0].attempts == 1
    finally:
        database_db.engine, database_db.SessionLocal = old_engine, old_factory
        await test_engine.dispose()


@pytest.mark.asyncio
async def test_health_is_ready_only_after_every_enabled_symbol_has_fresh_5m_and_15m_data(monkeypatch):
    from app.health import health
    from config import settings

    monkeypatch.setattr(type(settings), "is_production", property(lambda self: True))
    monkeypatch.setattr(settings, "enable_normal_engine", True)
    monkeypatch.setattr(settings, "enable_momentum_engine", True)
    monkeypatch.setattr(settings, "normal_symbol_limit", 1)
    monkeypatch.setattr(settings, "momentum_symbol_limit", 2)

    now = datetime.now(timezone.utc)
    now_ms = int(now.timestamp() * 1000)
    feed = {
        pair: {
            tf: {"status": "fresh", "observed_at_ms": now_ms}
            for tf in ("5m", "15m")
        }
        for pair in ("B-BTC_USDT", "B-ETH_USDT")
    }
    state = {
        "database": "postgres",
        "normal_engine": "running",
        "momentum_engine": "running",
        "valid_symbols_normal": 1,
        "valid_symbols_momentum": 2,
        "scan_task_expected": True,
        "scan_task_status": "running",
        "retest_task_status": "running",
        "telegram_delivery_task_expected": True,
        "telegram_delivery_task_status": "running",
        "last_scan_normal": now.isoformat(),
        "last_scan_momentum": now.isoformat(),
    }
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        bot_state=state,
        telegram=SimpleNamespace(bot=object(), admin_ids=[123]),
        rest=SimpleNamespace(get_candle_feed_status=lambda: feed),
        symbol_mgr=SimpleNamespace(
            normal_symbols=["B-BTC_USDT"],
            momentum_symbols=["B-BTC_USDT", "B-ETH_USDT"],
        ),
        websocket=SimpleNamespace(
            status="reconnecting", reconnect_count=2, symbol_feed_age_seconds={}
        ),
        start_time=now,
    )))
    response = await health(request)
    assert response["ok"] is True
    assert response["market_data_freshness"]["expected"] == 4
    assert response["market_data_freshness"]["fresh"] == 4
    assert response["market_data_freshness"]["feeds"]["B-BTC_USDT"]["5m"]["status"] == "fresh"
    assert response["market_data_freshness"]["feeds"]["B-ETH_USDT"]["15m"]["last_completed_close_ms"] is None
    # WebSocket loss is diagnostic only because REST closed-candle data is authoritative.
    assert response["websocket"]["status"] == "reconnecting"


@pytest.mark.asyncio
async def test_delta_rest_request_failure_marks_symbol_timeframe_stale(monkeypatch):
    from exchange.delta_rest import DeltaExchangeRest

    now_ms = (int(time.time() * 1000) // 300_000) * 300_000 + 5_000
    monkeypatch.setattr("exchange.delta_rest.time.time", lambda: now_ms / 1000)
    client = DeltaExchangeRest()
    client._canonical_to_delta["B-BTC_USDT"] = "BTCUSD"

    async def fail_request(url, params=None):
        raise RuntimeError("temporary exchange failure")

    client._get = fail_request
    with pytest.raises(RuntimeError, match="temporary exchange failure"):
        await client.get_candles("B-BTC_USDT", "5m", limit=100)
    status = client.get_candle_feed_status()["B-BTC_USDT"]["5m"]
    assert status["status"] == "stale"
    assert status["error"] == "rest_request_failed:RuntimeError"


@pytest.mark.asyncio
async def test_telegram_outbox_retries_a_transient_send_failure(tmp_path):
    from sqlalchemy import select, update
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    from database import db as database_db
    from database.db import Base
    from database import models  # noqa: F401
    from database.models import SignalDeliveryRecord
    from database.repositories import enqueue_signal_deliveries
    from tgbot.delivery_worker import TelegramDeliveryWorker

    test_engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'outbox_retry.db'}")
    old_engine, old_factory = database_db.engine, database_db.SessionLocal
    database_db.engine = test_engine
    database_db.SessionLocal = async_sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    try:
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        signal = {
            "pair": "B-ETH_USDT", "engine": "momentum", "direction": "SHORT",
            "candle_time_ms": 1_800_000_300_000, "signal_age_seconds": 0,
            "price": 200.0,
        }
        assert await enqueue_signal_deliveries(signal, [456]) == 1

        class FlakyTelegram:
            bot = object()
            admin_ids = [456]

            def __init__(self):
                self.calls = 0

            async def send_signal_to_chat(self, payload, chat_id):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("temporary Telegram timeout")

        telegram = FlakyTelegram()
        worker = TelegramDeliveryWorker(telegram)
        assert await worker.run_once() == 0
        async with database_db.SessionLocal() as session:
            row = (await session.execute(select(SignalDeliveryRecord))).scalar_one()
            assert row.status == "retry"
            assert row.attempts == 1
            await session.execute(
                update(SignalDeliveryRecord).where(
                    SignalDeliveryRecord.delivery_key == row.delivery_key
                ).values(next_attempt_at=datetime.now(timezone.utc).replace(tzinfo=None))
            )
            await session.commit()

        assert await worker.run_once() == 1
        assert telegram.calls == 2
        async with database_db.SessionLocal() as session:
            row = (await session.execute(select(SignalDeliveryRecord))).scalar_one()
            assert row.status == "delivered"
            assert row.attempts == 2
    finally:
        database_db.engine, database_db.SessionLocal = old_engine, old_factory
        await test_engine.dispose()


def test_websocket_is_not_ready_until_delta_acknowledges_all_subscriptions():
    from exchange.delta_ws import DeltaFuturesStream
    from exchange.candle_builder import CandleBuilder

    stream = DeltaFuturesStream(CandleBuilder(), rest_client=object())
    stream._channels = {
        "candlestick_5m": ["BTCUSD", "ETHUSD"],
        "candlestick_15m": ["BTCUSD", "ETHUSD"],
    }
    stream._handle_message({
        "type": "subscriptions",
        "channels": [
            {"name": "candlestick_5m", "symbols": ["BTCUSD", "ETHUSD"]},
            {"name": "candlestick_15m", "symbols": ["BTCUSD", "ETHUSD"]},
        ],
    })
    assert stream.connected is True
    assert stream._connected_event.is_set()
    assert stream._subscribed_channels == {"candlestick_5m", "candlestick_15m"}

    rejected = DeltaFuturesStream(CandleBuilder(), rest_client=object())
    rejected._channels = {"candlestick_5m": ["BTCUSD", "ETHUSD"]}
    rejected._handle_message({
        "type": "subscriptions",
        "channels": [{"name": "candlestick_5m", "symbols": ["BTCUSD"]}],
    })
    assert rejected.connected is False
    assert not rejected._connected_event.is_set()
    assert "ETHUSD" in rejected._subscription_error


def test_websocket_heartbeat_updates_heartbeat_clock(monkeypatch):
    from exchange.delta_ws import DeltaFuturesStream
    from exchange.candle_builder import CandleBuilder

    now_ms = 1_800_000_000_000
    monkeypatch.setattr("exchange.delta_ws.time.time", lambda: now_ms / 1000)
    stream = DeltaFuturesStream(CandleBuilder(), rest_client=object())
    stream._handle_message({"type": "heartbeat"})
    assert stream._last_heartbeat_ms == now_ms


def test_replacement_symbols_map_to_delta_canonical_usd_products():
    from exchange.delta_rest import DeltaExchangeRest

    for product_symbol, asset in (
        ("1000BONKUSD", "1000BONK"),
        ("FARTCOINUSD", "FARTCOIN"),
        ("ORDIUSD", "ORDI"),
    ):
        product = {
            "symbol": product_symbol,
            "quoting_asset": {"symbol": "USD"},
        }
        assert DeltaExchangeRest._asset_symbol(product) == asset
        assert DeltaExchangeRest._canonical_for_asset(asset) == f"B-{asset}_USDT"
