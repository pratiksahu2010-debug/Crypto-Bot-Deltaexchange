"""Application lifespan — startup / shutdown with graceful recovery."""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator

from fastapi import FastAPI

from config import settings
from utils.logger import logger
from utils.memory_monitor import check_memory
from utils.time_utils import parse_interval
from exchange.delta_rest import DeltaExchangeRest
from exchange.symbol_manager import SymbolManager
from exchange.candle_builder import CandleBuilder
from exchange.delta_ws import DeltaFuturesStream
from core.engine import SignalEngine, PRIMARY_CANDLE_HISTORY
from tgbot.bot import TelegramBot
from tgbot.delivery_worker import TelegramDeliveryWorker
from database.db import init_db, dispose_db


def _watch_background_task(task: asyncio.Task, state: dict, name: str) -> None:
    """Expose unexpected background-task termination to readiness and status."""
    def _completed(done: asyncio.Task) -> None:
        if done.cancelled():
            status, error = "stopped", "task_cancelled"
        else:
            exc = done.exception()
            status = "failed" if exc is not None else "stopped"
            error = f"{type(exc).__name__}: {exc}"[:500] if exc is not None else "task_exited"
        state[f"{name}_status"] = status
        state[f"{name}_error"] = error
        if done.cancelled() and state.get("shutting_down"):
            logger.info(
                f"BACKGROUND_TASK_STOPPED task={name} reason=application_shutdown"
            )
        else:
            logger.error(
                f"BACKGROUND_TASK_EXITED task={name} status={status} error={error}"
            )

    task.add_done_callback(_completed)


def validate_runtime_config() -> None:
    """Fail fast on invalid timeframe and safety-critical runtime settings."""
    # Parse first so malformed/non-positive values retain their precise errors.
    primary = parse_interval(settings.candle_primary_tf)
    supported_candle_intervals = {
        "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h",
        "6h", "12h", "1d", "1w",
    }
    for setting_name, interval in (
        ("CANDLE_PRIMARY_TF", settings.candle_primary_tf),
    ):
        if interval not in supported_candle_intervals:
            raise ValueError(
                f"{setting_name}={interval!r} is unsupported by the Delta Exchange India Futures candle adapter; "
                f"supported intervals: {', '.join(sorted(supported_candle_intervals))}"
            )
    # Production entries and ORB use completed 5m candles. The legacy
    # CANDLE_CONFIRM_TF setting is not a signal dependency.
    if settings.is_production and settings.candle_primary_tf != "5m":
        raise ValueError(
            "Production strategy requires CANDLE_PRIMARY_TF=5m for entries and ORB."
        )

    # NaN silently bypasses ordinary < / > checks. Reject non-finite strategy
    # and execution-quality thresholds before the scanner starts.
    import math
    numeric_settings = {
        "SIGNAL_THRESHOLD": settings.signal_threshold,
        "NORMAL_RVOL_MIN": settings.normal_rvol_min,
        "NORMAL_ADX_MIN": settings.normal_adx_min,
        "MOMENTUM_ADX_MIN": settings.momentum_adx_explosion,
        "NORMAL_MAX_EXTENSION_ATR": settings.normal_max_extension_atr,
        "NORMAL_PULLBACK_TOUCH_ATR": settings.normal_pullback_touch_atr,
        "ORB_MIN_RVOL": settings.orb_min_rvol,
        "ORB_ADX_MIN": settings.orb_adx_min,
        "ORB_MAX_EXTENSION_ATR": settings.orb_max_extension_atr,
        "LIVE_MIN_MOVE_ATR": settings.live_min_move_atr,
        "LIVE_MAX_MOVE_ATR": settings.live_max_move_atr,
        "LIVE_MIN_VOLUME_PACE": settings.live_min_volume_pace,
        "LIVE_MIN_SCORE": settings.live_min_score,
        "LIVE_ORB_HOLD_SECONDS": settings.live_orb_hold_seconds,
        "LIVE_ORB_MAX_EXTENSION_ATR": settings.live_orb_max_extension_atr,
    }
    for setting_name, value in numeric_settings.items():
        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            finite = False
        if not finite:
            raise ValueError(f"{setting_name} must be a finite number")

    if settings.candle_close_grace_seconds < 0:
        raise ValueError("CANDLE_CLOSE_GRACE_SECONDS cannot be negative")
    if settings.normal_max_signal_age_seconds <= 0:
        raise ValueError("NORMAL_MAX_SIGNAL_AGE_SECONDS must be positive")
    if settings.normal_retest_ttl_seconds <= 0:
        raise ValueError("NORMAL_RETEST_TTL_SECONDS must be positive")
    if settings.normal_max_trigger_age_candles < 0:
        raise ValueError("NORMAL_MAX_TRIGGER_AGE_CANDLES cannot be negative")
    if settings.normal_pullback_lookback < 1:
        raise ValueError("NORMAL_PULLBACK_LOOKBACK must be at least 1")
    if settings.normal_max_extension_atr <= 0:
        raise ValueError("NORMAL_MAX_EXTENSION_ATR must be positive")
    if settings.normal_pullback_touch_atr <= 0:
        raise ValueError("NORMAL_PULLBACK_TOUCH_ATR must be positive")
    if settings.max_signal_age_seconds <= 0:
        raise ValueError("MAX_SIGNAL_AGE_SECONDS must be positive")
    if settings.max_candles_buffer < PRIMARY_CANDLE_HISTORY:
        raise ValueError(
            f"MAX_CANDLES_BUFFER must be at least {PRIMARY_CANDLE_HISTORY} "
            "to retain the full indicator warm-up history"
        )
    if not 0.0 <= settings.signal_threshold <= 10.0:
        raise ValueError("SIGNAL_THRESHOLD must be between 0 and 10")
    if not 0.0 < settings.normal_rvol_min:
        raise ValueError("NORMAL_RVOL_MIN must be positive")
    if not 0.0 <= settings.normal_adx_min <= 100.0:
        raise ValueError("NORMAL_ADX_MIN must be between 0 and 100")
    if not 0.0 <= settings.momentum_adx_explosion <= 100.0:
        raise ValueError("MOMENTUM_ADX_MIN must be between 0 and 100")
    if settings.orb_min_rvol <= 0:
        raise ValueError("ORB_MIN_RVOL must be positive")
    if not 0.0 <= settings.orb_adx_min <= 100.0:
        raise ValueError("ORB_ADX_MIN must be between 0 and 100")
    if settings.orb_max_extension_atr <= 0:
        raise ValueError("ORB_MAX_EXTENSION_ATR must be positive")
    if settings.orb_active_minutes <= 0:
        raise ValueError("ORB_ACTIVE_MINUTES must be positive")
    if str(settings.live_ignition_mode).lower() not in ("off", "shadow", "live"):
        raise ValueError("LIVE_IGNITION_MODE must be off, shadow or live")
    if not 0.0 < settings.live_min_move_atr < settings.live_max_move_atr:
        raise ValueError("LIVE_MIN_MOVE_ATR must be positive and below LIVE_MAX_MOVE_ATR")
    if settings.live_min_volume_pace <= 0 or settings.live_orb_max_extension_atr <= 0:
        raise ValueError("LIVE_MIN_VOLUME_PACE and LIVE_ORB_MAX_EXTENSION_ATR must be positive")
    if settings.live_orb_hold_seconds < 0 or settings.live_max_signals_per_hour < 1:
        raise ValueError("LIVE_ORB_HOLD_SECONDS cannot be negative; LIVE_MAX_SIGNALS_PER_HOUR must be >= 1")
    if settings.memory_warning_mb <= 0:
        raise ValueError("MEMORY_WARNING_MB must be positive")
    import logging
    if not isinstance(getattr(logging, settings.log_level.upper(), None), int):
        raise ValueError("LOG_LEVEL must be a valid Python logging level")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    validate_runtime_config()
    logger.info("=== Delta Exchange India Signal Bot v3.0 starting ===")
    start_time = datetime.now(timezone.utc)

    database_ready = False
    try:
        await init_db()
        database_ready = True
    except Exception as exc:
        logger.exception(f"Database initialization failed: {exc}")

    rest = DeltaExchangeRest()
    # Do not open exchange or Telegram connections when durable deduplication
    # storage is unavailable: scans are paused, so those connections would only
    # consume resources and could misleadingly look like a functioning bot.
    if database_ready:
        await rest.start()

    symbol_mgr = SymbolManager(rest)
    if database_ready:
        try:
            await symbol_mgr.load_and_validate()
        except Exception as e:
            logger.error(f"Symbol validation failed: {e}")

    candle_builder = CandleBuilder()
    engine = SignalEngine(rest, symbol_mgr, candle_builder)
    ws_stream = DeltaFuturesStream(candle_builder, rest)
    # Register before the socket task starts so the first forming candle cannot
    # arrive before the engine callback is attached.
    ws_stream.set_candle_callback(engine.on_live_candle)
    ws_started = (
        await ws_stream.start(set(symbol_mgr.normal_symbols + symbol_mgr.momentum_symbols))
        if database_ready else False
    )
    if str(settings.live_ignition_mode).lower() != "off" and not ws_started:
        logger.warning("LIVE_IGNITION_INACTIVE reason=websocket_not_started (closed-candle scan unaffected)")
    if database_ready:
        await engine.restore_active_signals()
        await engine.restore_pending_retests()

    # Telegram is also kept offline while scans are paused.
    tg = TelegramBot()
    if database_ready:
        await tg.start()
    engine.set_telegram(tg)
    delivery_worker = TelegramDeliveryWorker(tg)

    app.state.rest = rest
    app.state.symbol_mgr = symbol_mgr
    app.state.candle_builder = candle_builder
    app.state.websocket = ws_stream
    app.state.engine = engine
    app.state.telegram = tg
    app.state.telegram_delivery_worker = delivery_worker
    app.state.start_time = start_time
    app.state.bot_state = {
        "websocket_status": (
            "paused_database_unavailable" if not database_ready
            else ws_stream.status if ws_started
            else ("rest_fallback" if settings.enable_websocket else "disabled_rest_mode")
        ),
        "normal_engine": "starting" if settings.enable_normal_engine else "disabled",
        "momentum_engine": "starting" if settings.enable_momentum_engine else "disabled",
        "scan_task_expected": bool(
            database_ready and (settings.enable_normal_engine or settings.enable_momentum_engine)
        ),
        "scan_task_status": (
            "starting" if database_ready and (settings.enable_normal_engine or settings.enable_momentum_engine)
            else "paused_database_unavailable" if not database_ready else "disabled"
        ),
        "scan_task_error": None,
        "retest_task_status": "starting" if database_ready else "paused_database_unavailable",
        "retest_task_error": None,
        "telegram_delivery_task_expected": bool(database_ready and tg.bot and tg.admin_ids),
        "telegram_delivery_task_status": "starting" if database_ready and tg.bot and tg.admin_ids else "disabled",
        "telegram_delivery_task_error": None,
        "last_market_update": None,
        "last_scan": None,
        "active_signals": 0,
        "watchlist_count": 0,
        "valid_symbols_normal": len(symbol_mgr.normal_symbols),
        "valid_symbols_momentum": len(symbol_mgr.momentum_symbols),
        "expected_symbols_normal": 40,
        "expected_symbols_momentum": 100,
        "scan_count": 0,
        "ws_reconnects": 0,
        "database": ("postgres" if settings.use_postgres else "sqlite") if database_ready else "unavailable",
        "uptime_seconds": 0,
        "telegram": "polling" if tg.polling_active else ("send_only" if tg.bot else "disabled"),
        "telegram_polling": bool(tg.polling_active),
        "eligible_symbols": len(symbol_mgr.valid_pairs),
        "skipped_symbols": len(symbol_mgr.skipped_reasons),

    }

    # Never run strategy scans without durable signal-claim storage. Keep the
    # FastAPI service and lightweight health endpoints alive for Render, but
    # expose an explicit degraded state until database initialization succeeds.
    scan_task = None
    retest_task = None
    telegram_delivery_task = None
    if database_ready:
        logger.info("BACKGROUND_TASK_START task=signal-scan-loop stage=preflight")
        scan_task = asyncio.create_task(engine.run_scan_loop(app), name="signal-scan-loop")
        retest_task = asyncio.create_task(engine.pending_retest_monitor_loop(interval_seconds=2.0), name="pending-retest-loop")
        # Scanner remains "starting" until universe refresh, WebSocket sync,
        # and the first completed-candle boundary have all succeeded.
        app.state.bot_state["scan_task_status"] = "starting"
        app.state.bot_state["scan_stage"] = "task_created"
        app.state.bot_state["scan_stage_started_at"] = datetime.now(timezone.utc).isoformat()
        app.state.bot_state["retest_task_status"] = "running"
        _watch_background_task(scan_task, app.state.bot_state, "scan_task")
        _watch_background_task(retest_task, app.state.bot_state, "retest_task")
        if tg.bot and tg.admin_ids:
            telegram_delivery_task = asyncio.create_task(
                delivery_worker.run_forever(), name="telegram-delivery-loop"
            )
            app.state.bot_state["telegram_delivery_task_status"] = "running"
            _watch_background_task(
                telegram_delivery_task, app.state.bot_state, "telegram_delivery_task"
            )
    else:
        app.state.bot_state["normal_engine"] = "paused_database_unavailable"
        app.state.bot_state["momentum_engine"] = "paused_database_unavailable"
        app.state.bot_state["retest_task_status"] = "paused_database_unavailable"
        logger.critical(
            "SIGNAL_SCANNING_DISABLED database_unavailable=true; "
            "health endpoints remain available, no market scans or alerts will run"
        )

    if database_ready and symbol_mgr.valid_pairs:
        logger.info("Startup complete. Durable database and active market universe ready.")
    elif not database_ready:
        logger.warning("Startup degraded: database unavailable; exchange feeds and scans remain paused.")
    else:
        logger.error("Startup degraded: no eligible live Delta perpetual symbols; scans have no universe.")
    check_memory()

    yield

    logger.info("Shutting down...")
    # Mark expected task cancellation before cancelling supervised workers so
    # deployment restarts do not emit misleading ERROR-level crash messages.
    app.state.bot_state["shutting_down"] = True
    delivery_worker.stop()
    for task in (scan_task, retest_task, telegram_delivery_task):
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    await engine.shutdown()
    await ws_stream.stop()
    await tg.stop()
    await rest.stop()
    if database_ready:
        await dispose_db()
    logger.info("Shutdown complete.")
