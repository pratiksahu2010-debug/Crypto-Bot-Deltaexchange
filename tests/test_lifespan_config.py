"""Startup configuration validation tests."""
import pytest

from app.lifespan import validate_runtime_config
from config import settings


def test_runtime_config_accepts_current_5m_15m_defaults():
    validate_runtime_config()


def test_legacy_confirmation_timeframe_is_not_a_startup_dependency(monkeypatch):
    # The setting is retained for backward-compatible .env files, but no
    # strategy or readiness check requires a higher-timeframe feed.
    monkeypatch.setattr(settings, "candle_confirm_tf", settings.candle_primary_tf)
    validate_runtime_config()


def test_runtime_config_rejects_nonpositive_primary_interval(monkeypatch):
    monkeypatch.setattr(settings, "candle_primary_tf", "0m")
    with pytest.raises(ValueError, match="Interval must be positive"):
        validate_runtime_config()


def test_runtime_config_rejects_nonpositive_orb_rvol(monkeypatch):
    monkeypatch.setattr(settings, "orb_min_rvol", 0)
    with pytest.raises(ValueError, match="ORB_MIN_RVOL"):
        validate_runtime_config()


def test_runtime_config_rejects_invalid_log_level(monkeypatch):
    monkeypatch.setattr(settings, "log_level", "NOT_A_LEVEL")
    with pytest.raises(ValueError, match="LOG_LEVEL"):
        validate_runtime_config()


def test_runtime_config_rejects_timeframe_not_supported_by_futures_adapter(monkeypatch):
    monkeypatch.setattr(settings, "candle_primary_tf", "2w")
    with pytest.raises(ValueError, match="unsupported by the Delta Exchange India Futures candle adapter"):
        validate_runtime_config()



@pytest.mark.asyncio
async def test_expected_background_task_cancellation_during_shutdown_is_not_error(monkeypatch):
    import asyncio
    from app.lifespan import _watch_background_task

    state = {"shutting_down": True}
    events = []
    monkeypatch.setattr("app.lifespan.logger.info", lambda message: events.append(("info", message)))
    monkeypatch.setattr("app.lifespan.logger.error", lambda message: events.append(("error", message)))

    task = asyncio.create_task(asyncio.sleep(60))
    _watch_background_task(task, state, "scan_task")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    assert state["scan_task_status"] == "stopped"
    assert state["scan_task_error"] == "task_cancelled"
    assert events == [("info", "BACKGROUND_TASK_STOPPED task=scan_task reason=application_shutdown")]


def test_production_requires_strategy_timeframes_5m_and_15m(monkeypatch):
    monkeypatch.setattr(type(settings), "is_production", property(lambda self: True))
    monkeypatch.setattr(settings, "candle_primary_tf", "1m")
    monkeypatch.setattr(settings, "candle_confirm_tf", "15m")
    with pytest.raises(ValueError, match="requires CANDLE_PRIMARY_TF=5m"):
        validate_runtime_config()



def test_database_startup_does_not_delete_persistent_signal_state():
    import inspect
    from database.db import init_db

    source = inspect.getsource(init_db).upper()
    assert "DELETE FROM SIGNALS" not in source
    assert "DELETE FROM WATCHLIST" not in source
    assert "DELETE FROM SIGNAL_DEDUP" not in source
    assert "DROP TABLE" not in source


def test_signal_engine_does_not_require_orderbook_for_signal_delivery():
    import inspect
    from core.engine import SignalEngine

    source = inspect.getsource(SignalEngine)
    assert "_check_orderbook_spread" not in source
    assert "get_futures_orderbook" not in source


@pytest.mark.parametrize(
    "setting,value,error",
    [
        ("normal_max_signal_age_seconds", 0, "NORMAL_MAX_SIGNAL_AGE_SECONDS"),
        ("normal_retest_ttl_seconds", 0, "NORMAL_RETEST_TTL_SECONDS"),
        ("normal_max_trigger_age_candles", -1, "NORMAL_MAX_TRIGGER_AGE_CANDLES"),
        ("normal_pullback_lookback", 0, "NORMAL_PULLBACK_LOOKBACK"),
        ("normal_max_extension_atr", 0.0, "NORMAL_MAX_EXTENSION_ATR"),
        ("normal_pullback_touch_atr", 0.0, "NORMAL_PULLBACK_TOUCH_ATR"),
    ],
)
def test_runtime_config_rejects_invalid_normal_entry_timing_settings(
    monkeypatch, setting, value, error
):
    monkeypatch.setattr(settings, setting, value)
    with pytest.raises(ValueError, match=error):
        validate_runtime_config()


def test_signal_only_and_order_execution_are_hard_disabled():
    assert settings.signal_only is True
    assert settings.orders_enabled is False


def test_normal_adx_floor_cannot_be_lowered_below_20(monkeypatch):
    from core.engine import _normal_adx_floor

    monkeypatch.setattr(settings, "normal_adx_min", 18.0)
    assert _normal_adx_floor(settings.normal_adx_min) == 20.0
