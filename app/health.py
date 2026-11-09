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
            # A process with no completed scan is alive but not scanner-ready.
            stale = True
            status = "waiting"

        any_stale = any_stale or stale
        freshness[engine] = {
            "status": status,
            "last_scan": last_scan,
            "age_seconds": round(max(0.0, age), 1) if age is not None else None,
            "stale_after_seconds": round(stale_after, 1),
        }
    return freshness, any_stale


def _market_data_freshness(request: Request) -> dict:
    """Check the last authoritative REST candle observation per active symbol/TF."""
    rest = getattr(request.app.state, "rest", None)
    symbols = getattr(request.app.state, "symbol_mgr", None)
    getter = getattr(rest, "get_candle_feed_status", None)
    if not callable(getter) or symbols is None:
        return {"ready": False, "expected": 0, "fresh": 0, "missing_or_stale": 0, "issues": []}

    pairs = set()
    if settings.enable_normal_engine:
        pairs.update(getattr(symbols, "normal_symbols", []) or [])
    if settings.enable_momentum_engine:
        pairs.update(getattr(symbols, "momentum_symbols", []) or [])
    intervals = tuple(dict.fromkeys((settings.candle_primary_tf, settings.candle_confirm_tf)))
    enabled_age_limits = []
    if settings.enable_normal_engine:
        enabled_age_limits.append(float(settings.normal_max_signal_age_seconds))
    if settings.enable_momentum_engine:
        enabled_age_limits.append(float(settings.max_signal_age_seconds))
    readiness_entry_age_limit = (
        min(enabled_age_limits) if enabled_age_limits
        else float(settings.max_signal_age_seconds)
    )
    feed = getter() or {}
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    expected = len(pairs) * len(intervals)
    fresh = 0
    issues = []
    feed_details = {}
    for pair in sorted(pairs):
        feed_details[pair] = {}
        for interval in intervals:
            item = (feed.get(pair) or {}).get(interval)
            observed = item.get("observed_at_ms") if isinstance(item, dict) else None
            age = (now_ms - int(observed)) / 1000.0 if observed is not None else None
            interval_seconds = parse_interval(interval).total_seconds()
            stale_after = max(interval_seconds * 3.0, interval_seconds + settings.max_signal_age_seconds)
            status = item.get("status", "missing") if isinstance(item, dict) else "missing"
            candle_age = (
                item.get("candle_age_seconds")
                if isinstance(item, dict) else None
            )
            candle_age_limit = (
                readiness_entry_age_limit
                if interval == settings.candle_primary_tf
                else interval_seconds + readiness_entry_age_limit
            )
            healthy = (
                isinstance(item, dict)
                and status == "fresh"
                and age is not None
                and -60.0 <= age <= stale_after
                and candle_age is not None
                and 0.0 <= float(candle_age) <= candle_age_limit
            )
            feed_details[pair][interval] = {
                "status": status,
                "observation_age_seconds": round(max(0.0, age), 1) if age is not None else None,
                "candle_age_seconds": item.get("candle_age_seconds") if isinstance(item, dict) else None,
                "last_completed_open_ms": item.get("last_completed_open_ms") if isinstance(item, dict) else None,
                "last_completed_close_ms": item.get("last_completed_close_ms") if isinstance(item, dict) else None,
                "error": item.get("error") if isinstance(item, dict) else "no_observation",
            }
            if healthy:
                fresh += 1
            else:
                issues.append({
                    "pair": pair,
                    "timeframe": interval,
                    "status": status,
                    "age_seconds": round(max(0.0, age), 1) if age is not None else None,
                })
    return {
        "ready": expected > 0 and fresh == expected,
        "expected": expected,
        "fresh": fresh,
        "missing_or_stale": expected - fresh,
        "issues": issues[:20],
        "feeds": feed_details,
    }


def _websocket_diagnostics(request: Request) -> dict:
    websocket = getattr(request.app.state, "websocket", None)
    if websocket is None:
        return {"status": "unavailable", "reconnects": 0, "symbol_feed_age_seconds": {}}
    return {
        "status": websocket.status,
        "reconnects": websocket.reconnect_count,
        "symbol_feed_age_seconds": websocket.symbol_feed_age_seconds,
    }


@router.get("/")
async def root() -> Dict[str, Any]:
    return {
        "service": "delta-india-signal-bot",
        "version": "3.0",
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

    # Readiness must represent a started scanner in every environment.
    # A live HTTP process before lifespan initialization is only liveness, not readiness.
    if state is None:
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "status": "degraded",
                "reasons": ["startup_incomplete"],
                "liveness": "/ping",
            },
        )

    scan_freshness, scan_cycle_stale = _scan_freshness(state or {}, request)
    market_data = _market_data_freshness(request)
    websocket = _websocket_diagnostics(request)

    if state is not None:
        database = state.get("database")
        database_paused = (
            (settings.enable_normal_engine and state.get("normal_engine") == "paused_database_unavailable")
            or (settings.enable_momentum_engine and state.get("momentum_engine") == "paused_database_unavailable")
        )
        normal_symbols_missing = (
            settings.enable_normal_engine
            and int(state.get("valid_symbols_normal") or 0) < settings.normal_symbol_limit
        )
        momentum_symbols_missing = (
            settings.enable_momentum_engine
            and int(state.get("valid_symbols_momentum") or 0) < settings.momentum_symbol_limit
        )
        telegram_unavailable = settings.is_production and not telegram_delivery_ready
        scan_task_failed = (
            bool(state.get("scan_task_expected"))
            and state.get("scan_task_status") in ("failed", "stopped")
        )
        scan_task_not_running = (
            bool(state.get("scan_task_expected"))
            and state.get("scan_task_status") != "running"
        )
        engine_not_running = (
            (settings.enable_normal_engine and state.get("normal_engine") != "running")
            or (settings.enable_momentum_engine and state.get("momentum_engine") != "running")
        )
        retest_task_failed = (
            database != "unavailable"
            and state.get("retest_task_status") in ("failed", "stopped")
        )
        retest_task_not_running = (
            database != "unavailable"
            and state.get("retest_task_status") != "running"
        )
        delivery_task_failed = (
            bool(state.get("telegram_delivery_task_expected"))
            and state.get("telegram_delivery_task_status") in ("failed", "stopped")
        )
        delivery_task_not_running = (
            bool(state.get("telegram_delivery_task_expected"))
            and state.get("telegram_delivery_task_status") != "running"
        )
        market_data_not_ready = not market_data["ready"]
        if (
            database == "unavailable"
            or database_paused
            or normal_symbols_missing
            or momentum_symbols_missing
            or telegram_unavailable
            or scan_task_failed
            or retest_task_failed
            or delivery_task_failed
            or scan_task_not_running
            or engine_not_running
            or retest_task_not_running
            or delivery_task_not_running
            or scan_cycle_stale
            or market_data_not_ready
        ):
            reasons = []
            if database == "unavailable" or database_paused:
                reasons.append("database_unavailable")
            if normal_symbols_missing:
                if int(state.get("valid_symbols_normal") or 0) <= 0:
                    reasons.append("normal_universe_empty")
                reasons.append("normal_universe_incomplete")
            if momentum_symbols_missing:
                if int(state.get("valid_symbols_momentum") or 0) <= 0:
                    reasons.append("momentum_universe_empty")
                reasons.append("momentum_universe_incomplete")
            if telegram_unavailable:
                reasons.append("telegram_delivery_unavailable")
            if scan_task_failed:
                reasons.append("scan_loop_failed")
            if retest_task_failed:
                reasons.append("pending_retest_monitor_failed")
            if delivery_task_failed:
                reasons.append("telegram_delivery_worker_failed")
            if scan_task_not_running:
                reasons.append("scan_loop_not_running")
            if engine_not_running:
                reasons.append("strategy_engine_not_running")
            if retest_task_not_running:
                reasons.append("pending_retest_monitor_not_running")
            if delivery_task_not_running:
                reasons.append("telegram_delivery_worker_not_running")
            if scan_cycle_stale:
                reasons.append("scan_cycle_stale")
                reasons.append("scan_cycle_not_fresh_or_not_started")
            if market_data_not_ready:
                reasons.append("per_symbol_candle_data_not_ready")
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
                    "market_data_freshness": market_data,
                    "websocket": websocket,
                    "expected_symbols_normal": settings.normal_symbol_limit,
                    "expected_symbols_momentum": settings.momentum_symbol_limit,
                    "reasons": reasons,
                },
            )
    return {
        "ok": True,
        "status": "healthy",
        "telegram_delivery_ready": telegram_delivery_ready,
        "scan_freshness": scan_freshness,
        "market_data_freshness": market_data,
        "websocket": websocket,
        "expected_symbols_normal": settings.normal_symbol_limit,
        "expected_symbols_momentum": settings.momentum_symbol_limit,
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
        "expected_symbols_normal": settings.normal_symbol_limit,
        "expected_symbols_momentum": settings.momentum_symbol_limit,
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
