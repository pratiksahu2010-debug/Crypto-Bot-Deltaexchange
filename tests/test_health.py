"""Basic health endpoint tests."""
import pytest
from httpx import AsyncClient, ASGITransport
from main import app


@pytest.mark.asyncio
async def test_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_ping_is_empty_keepalive_only():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ping")
        assert resp.status_code == 204
        assert resp.content == b""
        assert resp.text == ""



@pytest.mark.asyncio
async def test_ping_head_is_empty():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.head("/ping")
        assert resp.status_code == 204
        assert resp.content == b""


@pytest.mark.asyncio
async def test_ping_remains_alive_when_database_is_degraded():
    old_state = getattr(app.state, "bot_state", None)
    app.state.bot_state = {
        "database": "unavailable",
        "normal_engine": "paused_database_unavailable",
        "momentum_engine": "paused_database_unavailable",
    }
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/ping")
            assert resp.status_code == 204
            assert resp.content == b""
    finally:
        if old_state is None:
            delattr(app.state, "bot_state")
        else:
            app.state.bot_state = old_state


@pytest.mark.asyncio
async def test_health_reports_database_degraded_state():
    old_state = getattr(app.state, "bot_state", None)
    app.state.bot_state = {
        "database": "unavailable",
        "normal_engine": "paused_database_unavailable",
        "momentum_engine": "paused_database_unavailable",
    }
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 503
            assert resp.json()["status"] == "degraded"
    finally:
        if old_state is None:
            delattr(app.state, "bot_state")
        else:
            app.state.bot_state = old_state


@pytest.mark.asyncio
async def test_health_reports_empty_enabled_universe_as_degraded():
    old_state = getattr(app.state, "bot_state", None)
    app.state.bot_state = {
        "database": "postgres",
        "normal_engine": "starting",
        "momentum_engine": "starting",
        "valid_symbols_normal": 0,
        "valid_symbols_momentum": 80,
    }
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 503
            assert "normal_universe_empty" in resp.json()["reasons"]
    finally:
        if old_state is None:
            delattr(app.state, "bot_state")
        else:
            app.state.bot_state = old_state


@pytest.mark.asyncio
async def test_production_health_degrades_when_telegram_delivery_is_unavailable(monkeypatch):
    from types import SimpleNamespace
    from config import settings

    monkeypatch.setattr(type(settings), "is_production", property(lambda self: True))
    old_state = getattr(app.state, "bot_state", None)
    old_telegram = getattr(app.state, "telegram", None)
    app.state.bot_state = {
        "database": "postgres",
        "normal_engine": "running",
        "momentum_engine": "running",
        "valid_symbols_normal": 40,
        "valid_symbols_momentum": 80,
    }
    app.state.telegram = SimpleNamespace(bot=None, admin_ids=[])
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 503
            assert "telegram_delivery_unavailable" in resp.json()["reasons"]
    finally:
        if old_state is None:
            delattr(app.state, "bot_state")
        else:
            app.state.bot_state = old_state
        if old_telegram is None:
            if hasattr(app.state, "telegram"):
                delattr(app.state, "telegram")
        else:
            app.state.telegram = old_telegram


@pytest.mark.asyncio
async def test_production_health_accepts_configured_telegram_delivery(monkeypatch):
    from types import SimpleNamespace
    from config import settings

    monkeypatch.setattr(type(settings), "is_production", property(lambda self: True))
    old_state = getattr(app.state, "bot_state", None)
    old_telegram = getattr(app.state, "telegram", None)
    app.state.bot_state = {
        "database": "postgres",
        "normal_engine": "running",
        "momentum_engine": "running",
        "valid_symbols_normal": 40,
        "valid_symbols_momentum": 80,
    }
    app.state.telegram = SimpleNamespace(bot=object(), admin_ids=[123456])
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["telegram_delivery_ready"] is True
    finally:
        if old_state is None:
            delattr(app.state, "bot_state")
        else:
            app.state.bot_state = old_state
        if old_telegram is None:
            if hasattr(app.state, "telegram"):
                delattr(app.state, "telegram")
        else:
            app.state.telegram = old_telegram



@pytest.mark.asyncio
async def test_health_reports_unexpected_scan_task_exit(monkeypatch):
    from config import settings

    monkeypatch.setattr(type(settings), "is_production", property(lambda self: False))
    old_state = getattr(app.state, "bot_state", None)
    app.state.bot_state = {
        "database": "postgres",
        "normal_engine": "running",
        "momentum_engine": "running",
        "valid_symbols_normal": 40,
        "valid_symbols_momentum": 80,
        "scan_task_expected": True,
        "scan_task_status": "failed",
        "scan_task_error": "RuntimeError: universe refresh failed",
    }
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 503
            data = resp.json()
            assert "scan_loop_failed" in data["reasons"]
            assert data["scan_task_status"] == "failed"
            assert "universe refresh failed" in data["scan_task_error"]
    finally:
        if old_state is None:
            delattr(app.state, "bot_state")
        else:
            app.state.bot_state = old_state


@pytest.mark.asyncio
async def test_health_reports_unexpected_telegram_worker_exit(monkeypatch):
    from config import settings

    monkeypatch.setattr(type(settings), "is_production", property(lambda self: False))
    old_state = getattr(app.state, "bot_state", None)
    app.state.bot_state = {
        "database": "postgres",
        "normal_engine": "running",
        "momentum_engine": "running",
        "valid_symbols_normal": 40,
        "valid_symbols_momentum": 80,
        "telegram_delivery_task_expected": True,
        "telegram_delivery_task_status": "stopped",
        "telegram_delivery_task_error": "task_exited",
    }
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 503
            assert "telegram_delivery_worker_failed" in resp.json()["reasons"]
    finally:
        if old_state is None:
            delattr(app.state, "bot_state")
        else:
            app.state.bot_state = old_state



@pytest.mark.asyncio
async def test_health_reports_unexpected_pending_retest_monitor_exit(monkeypatch):
    from config import settings

    monkeypatch.setattr(type(settings), "is_production", property(lambda self: False))
    old_state = getattr(app.state, "bot_state", None)
    app.state.bot_state = {
        "database": "postgres",
        "normal_engine": "running",
        "momentum_engine": "running",
        "valid_symbols_normal": 40,
        "valid_symbols_momentum": 80,
        "retest_task_status": "failed",
        "retest_task_error": "RuntimeError: retest monitor crashed",
    }
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 503
            data = resp.json()
            assert "pending_retest_monitor_failed" in data["reasons"]
            assert data["retest_task_status"] == "failed"
            assert "retest monitor crashed" in data["retest_task_error"]
    finally:
        if old_state is None:
            delattr(app.state, "bot_state")
        else:
            app.state.bot_state = old_state



@pytest.mark.asyncio
async def test_health_degrades_when_enabled_engine_scan_is_stale(monkeypatch):
    from datetime import datetime, timedelta, timezone
    from config import settings
    from utils.time_utils import parse_interval

    monkeypatch.setattr(type(settings), "is_production", property(lambda self: False))
    old_state = getattr(app.state, "bot_state", None)
    old_start = getattr(app.state, "start_time", None)
    now = datetime.now(timezone.utc)
    stale_after = max(
        300.0,
        parse_interval(settings.candle_primary_tf).total_seconds() * 2
        + settings.candle_close_grace_seconds
        + settings.max_signal_age_seconds,
    )
    app.state.start_time = now - timedelta(minutes=1)
    app.state.bot_state = {
        "database": "postgres",
        "normal_engine": "running",
        "momentum_engine": "running",
        "valid_symbols_normal": 40,
        "valid_symbols_momentum": 80,
        "scan_task_expected": True,
        "scan_task_status": "running",
        "last_scan_normal": (now - timedelta(seconds=stale_after + 30)).isoformat(),
        "last_scan_momentum": now.isoformat(),
    }
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 503
            data = resp.json()
            assert "scan_cycle_stale" in data["reasons"]
            assert data["scan_freshness"]["normal"]["status"] == "stale"
            assert data["scan_freshness"]["momentum"]["status"] == "fresh"
    finally:
        if old_state is None:
            delattr(app.state, "bot_state")
        else:
            app.state.bot_state = old_state
        if old_start is None:
            if hasattr(app.state, "start_time"):
                delattr(app.state, "start_time")
        else:
            app.state.start_time = old_start


@pytest.mark.asyncio
async def test_health_allows_scan_startup_grace_before_first_cycle(monkeypatch):
    from datetime import datetime, timezone
    from config import settings

    monkeypatch.setattr(type(settings), "is_production", property(lambda self: False))
    old_state = getattr(app.state, "bot_state", None)
    old_start = getattr(app.state, "start_time", None)
    app.state.start_time = datetime.now(timezone.utc)
    app.state.bot_state = {
        "database": "postgres",
        "normal_engine": "running",
        "momentum_engine": "running",
        "valid_symbols_normal": 40,
        "valid_symbols_momentum": 80,
        "scan_task_expected": True,
        "scan_task_status": "running",
    }
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["scan_freshness"]["normal"]["status"] == "waiting"
            assert resp.json()["scan_freshness"]["momentum"]["status"] == "waiting"
    finally:
        if old_state is None:
            delattr(app.state, "bot_state")
        else:
            app.state.bot_state = old_state
        if old_start is None:
            if hasattr(app.state, "start_time"):
                delattr(app.state, "start_time")
        else:
            app.state.start_time = old_start

@pytest.mark.asyncio
async def test_metrics_includes_telegram_outbox_queue(monkeypatch):
    from database import repositories

    async def fake_queue_stats():
        return {"pending": 2, "retry": 1, "total": 3}

    monkeypatch.setattr(repositories, "signal_delivery_queue_stats", fake_queue_stats)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert resp.json()["telegram_outbox_queue"] == {
        "pending": 2, "retry": 1, "total": 3
    }
