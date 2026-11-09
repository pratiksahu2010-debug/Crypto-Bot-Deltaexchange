"""Health, status, ping, metrics endpoints — lightweight for keep-alive."""
from typing import Any, Dict
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from config import settings
from utils.time_utils import utc_iso, parse_interval
from utils.memory_monitor import check_memory

router = APIRouter()


def _scan_freshness(state: dict, request: Request) -> tuple[dict, bool]:
    """Report per-engine scan freshness and detect a stalled live scan loop."""
    if not state.get("scan_task_expected"):
        return {}, False

    interval_seconds = parse_interval(settings.candle_primary_tf).total_seconds()
    stale_after = max(
        300.0,
        interval_seconds * 2
        + max(0, settings.candle_close_grace_seconds)
        + settings.max_signal_age_seconds,
    )
    now = datetime.now(timezone.utc)
    startup = getattr(request.app.state, "start_time", None)
    if startup is not None and startup.tzinfo is None:
        startup = startup.replace(tzinfo=timezone.utc)
    startup_age = max(0.0, (now - startup).total_seconds()) if startup else 0.0

    freshness = {}
    any_stale = False
    for engine, enabled in (
        ("normal", settings.enable_normal_engine),
        ("momentum", settings.enable_momentum_engine),
    ):
        if not enabled:
            continue
        last_scan = state.get(f"last_scan_{engine}")
        age = None
        if last_scan:
            try:
                scanned_at = datetime.fromisoformat(str(last_scan).replace("Z", "+00:00"))
                if scanned_at.tzinfo is None:
                    scanned_at = scanned_at.replace(tzinfo=timezone.utc)
                age = (now - scanned_at).total_seconds()
            except (TypeError, ValueError, OverflowError):
                age = None

        if last_scan:
            stale = age is None or age < -60.0 or age > stale_after
            status = "stale" if stale else "fresh"
        else:
            stale = startup_age > stale_after
            status = "stale" if stale else "waiting"

        any_stale = any_stale or stale
        freshness[engine] = {
            "status": status,
            "last_scan": last_scan,
            "age_seconds": round(max(0.0, age), 1) if age is not None else None,
            "stale_after_seconds": round(stale_after, 1),
        }
    return freshness, any_stale


@router.get("/")
async def root() -> Dict[str, Any]:
    return {
        "service": "delta-india-signal-bot",
        "version": "2.0",
        "docs": "/docs",
        "health": "/health",
        "ping": "/ping",
    }


@router.get("/health")
async def health(request: Request):
    """Readiness probe; /ping remains process liveness only."""
    state = getattr(request.app.state, "bot_state", None)
    telegram = getattr(request.app.state, "telegram", None)
    telegram_delivery_ready = bool(
        getattr(telegram, "bot", None) and getattr(telegram, "admin_ids", [])
    )

    # In production, a process that has not completed lifespan startup is not ready.
    if state is None and settings.is_production:
        return JSONResponse(
            status_code=503,
            content={"ok": False, "status": "degraded", "reasons": ["startup_incomplete"]},
        )

    scan_freshness, scan_cycle_stale = _scan_freshness(state or {}, request)

    if state is not None:
        database = state.get("database")
        database_paused = (
            (settings.enable_normal_engine and state.get("normal_engine") == "paused_database_unavailable")
            or (settings.enable_momentum_engine and state.get("momentum_engine") == "paused_database_unavailable")
        )
        normal_symbols_missing = (
            settings.enable_normal_engine and int(state.get("valid_symbols_normal") or 0) <= 0
        )
        momentum_symbols_missing = (
            settings.enable_momentum_engine and int(state.get("valid_symbols_momentum") or 0) <= 0
        )
        telegram_unavailable = settings.is_production and not telegram_delivery_ready
        scan_task_failed = (
            bool(state.get("scan_task_expected"))
            and state.get("scan_task_status") in ("failed", "stopped")
        )
        retest_task_failed = (
            database != "unavailable"
            and state.get("retest_task_status") in ("failed", "stopped")
        )
        delivery_task_failed = (
            bool(state.get("telegram_delivery_task_expected"))
            and state.get("telegram_delivery_task_status") in ("failed", "stopped")
        )
        if (
            database == "unavailable"
            or database_paused
            or normal_symbols_missing
            or momentum_symbols_missing
            or telegram_unavailable
            or scan_task_failed
            or retest_task_failed
            or delivery_task_failed
            or scan_cycle_stale
        ):
            reasons = []
            if database == "unavailable" or database_paused:
                reasons.append("database_unavailable")
            if normal_symbols_missing:
                reasons.append("normal_universe_empty")
            if momentum_symbols_missing:
                reasons.append("momentum_universe_empty")
            if telegram_unavailable:
                reasons.append("telegram_delivery_unavailable")
            if scan_task_failed:
                reasons.append("scan_loop_failed")
            if retest_task_failed:
                reasons.append("pending_retest_monitor_failed")
            if delivery_task_failed:
                reasons.append("telegram_delivery_worker_failed")
            if scan_cycle_stale:
                reasons.append("scan_cycle_stale")
            return JSONResponse(
                status_code=503,
                content={
                    "ok": False,
                    "status": "degraded",
                    "database": database,
                    "normal_engine": state.get("normal_engine"),
                    "momentum_engine": state.get("momentum_engine"),
                    "valid_symbols_normal": state.get("valid_symbols_normal", 0),
                    "valid_symbols_momentum": state.get("valid_symbols_momentum", 0),
                    "telegram_delivery_ready": telegram_delivery_ready,
                    "scan_task_status": state.get("scan_task_status", "unknown"),
                    "scan_task_error": state.get("scan_task_error"),
                    "retest_task_status": state.get("retest_task_status", "unknown"),
                    "retest_task_error": state.get("retest_task_error"),
                    "telegram_delivery_task_status": state.get("telegram_delivery_task_status", "unknown"),
                    "telegram_delivery_task_error": state.get("telegram_delivery_task_error"),
                    "scan_freshness": scan_freshness,
                    "reasons": reasons,
                },
            )
    return {
        "ok": True,
        "status": "healthy",
        "telegram_delivery_ready": telegram_delivery_ready,
        "scan_freshness": scan_freshness,
    }


@router.api_route("/ping", methods=["GET", "HEAD"], response_class=Response)
async def ping() -> Response:
    """
    No-op process liveness probe.

    Deliberately does not inspect application state, touch the database,
    call the exchange, scan markets, or send Telegram messages. Always returns
    an empty HTTP 204 response for GET and HEAD keep-alive requests.
    """
    return Response(
        content=b"",
        status_code=204,
        headers={"Cache-Control": "no-store"},
    )


@router.get("/status")
async def status(request: Request) -> Dict[str, Any]:
    app_state = getattr(request.app.state, "bot_state", {})
    mem = check_memory()
    telegram = getattr(request.app.state, "telegram", None)
    telegram_polling = bool(getattr(telegram, "polling_active", False))
    telegram_status = "polling" if telegram_polling else ("send_only" if getattr(telegram, "bot", None) else "disabled")
    telegram_delivery_ready = bool(
        getattr(telegram, "bot", None) and getattr(telegram, "admin_ids", [])
    )
    delivery_worker = getattr(request.app.state, "telegram_delivery_worker", None)
    from database.repositories import signal_delivery_queue_stats
    outbox_stats = await signal_delivery_queue_stats()
    return {
        "ok": True,
        "service": "delta-india-signal-bot",
        "timestamp": utc_iso(),
        "uptime_seconds": app_state.get("uptime_seconds", 0),
        "websocket": app_state.get("websocket_status", "unknown"),
        "websocket_reconnects": app_state.get("ws_reconnects", 0),
        "websocket_price_feed_age_seconds": app_state.get("ws_price_feed_age_seconds"),
        "last_market_update": app_state.get("last_market_update"),
        "normal_engine": app_state.get("normal_engine", "unknown"),
        "momentum_engine": app_state.get("momentum_engine", "unknown"),
        "last_scan": app_state.get("last_scan"),
        "last_scan_normal": app_state.get("last_scan_normal"),
        "last_scan_momentum": app_state.get("last_scan_momentum"),
        "last_scan_duration_ms_normal": app_state.get("last_scan_duration_ms_normal"),
        "last_scan_duration_ms_momentum": app_state.get("last_scan_duration_ms_momentum"),
        "scan_task_expected": app_state.get("scan_task_expected", False),
        "scan_task_status": app_state.get("scan_task_status", "unknown"),
        "scan_task_error": app_state.get("scan_task_error"),
        "retest_task_status": app_state.get("retest_task_status", "unknown"),
        "retest_task_error": app_state.get("retest_task_error"),
        "telegram_delivery_task_status": app_state.get("telegram_delivery_task_status", "unknown"),
        "telegram_delivery_task_error": app_state.get("telegram_delivery_task_error"),
        "active_signals": app_state.get("active_signals", 0),
        "watchlist_count": app_state.get("watchlist_count", 0),
        "scan_errors_normal": app_state.get("scan_errors_normal", 0),
        "scan_errors_momentum": app_state.get("scan_errors_momentum", 0),
        "valid_symbols_normal": app_state.get("valid_symbols_normal", 0),
        "valid_symbols_momentum": app_state.get("valid_symbols_momentum", 0),
        "memory_mb": mem["memory_mb"],
        "database": app_state.get("database", "unknown"),
        "telegram": telegram_status,
        "telegram_polling": telegram_polling,
        "telegram_delivery_ready": telegram_delivery_ready,
        "telegram_admin_count": len(getattr(telegram, "admin_ids", []) or []),
        "eligible_symbols": app_state.get("eligible_symbols", 0),
        "skipped_symbols": app_state.get("skipped_symbols", 0),
        "liquidity_filtered_symbols": app_state.get("liquidity_filtered_symbols", 0),
        "normal_min_24h_turnover_usdt": app_state.get("normal_min_24h_turnover_usdt"),
        "momentum_min_24h_turnover_usdt": app_state.get("momentum_min_24h_turnover_usdt"),
        "telegram_outbox_worker": "running" if getattr(delivery_worker, "running", False) else "stopped",
        "telegram_outbox_delivered": getattr(delivery_worker, "delivered", 0),
        "telegram_outbox_failed": getattr(delivery_worker, "failed", 0),
        "telegram_outbox_expired": getattr(delivery_worker, "expired", 0),
        "telegram_outbox_queue": outbox_stats,
    }


@router.get("/metrics")
async def metrics(request: Request) -> Dict[str, Any]:
    app_state = getattr(request.app.state, "bot_state", {})
    mem = check_memory()
    delivery_worker = getattr(request.app.state, "telegram_delivery_worker", None)
    from database.repositories import signal_delivery_queue_stats
    outbox_stats = await signal_delivery_queue_stats()
    return {
        "uptime_seconds": app_state.get("uptime_seconds", 0),
        "scan_count": app_state.get("scan_count", 0),
        "last_scan_duration_ms": app_state.get("last_scan_duration_ms", 0),
        "valid_symbols": app_state.get("valid_symbols_normal", 0) + app_state.get("valid_symbols_momentum", 0),
        "active_signals": app_state.get("active_signals", 0),
        "watchlist_entries": app_state.get("watchlist_count", 0),
        "scan_errors_normal": app_state.get("scan_errors_normal", 0),
        "scan_errors_momentum": app_state.get("scan_errors_momentum", 0),
        "websocket_reconnects": app_state.get("ws_reconnects", 0),
        "websocket_price_feed_age_seconds": app_state.get("ws_price_feed_age_seconds"),
        "futures_rest_requests": getattr(getattr(request.app.state, "rest", None), "request_count", 0),
        "futures_rest_failures": getattr(getattr(request.app.state, "rest", None), "request_failures", 0),
        "candle_cache_hits": getattr(getattr(request.app.state, "rest", None), "candle_cache_hits", 0),
        "pending_retests": len(getattr(getattr(request.app.state, "engine", None), "pending_retests", {})),
        "pending_retest_checks": getattr(getattr(request.app.state, "engine", None), "pending_retest_checks", 0),
        "pending_retest_failures": getattr(getattr(request.app.state, "engine", None), "pending_retest_failures", 0),
        "telegram_delivery_failures": getattr(getattr(request.app.state, "engine", None), "telegram_delivery_failures", 0),
        "telegram_outbox_worker_running": bool(getattr(delivery_worker, "running", False)),
        "telegram_outbox_delivered": getattr(delivery_worker, "delivered", 0),
        "telegram_outbox_failed": getattr(delivery_worker, "failed", 0),
        "telegram_outbox_expired": getattr(delivery_worker, "expired", 0),
        "telegram_outbox_queue": outbox_stats,
        "memory_mb": mem["memory_mb"],
        "timestamp": utc_iso(),
    }
