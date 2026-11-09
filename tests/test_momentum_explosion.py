import pytest

from strategies.momentum.explosion_detector import evaluate
from strategies.momentum.strategy_manager import (
    diagnose_momentum_state, detect_momentum_state, is_pre_explosion_signal,
)


def indicators(**overrides):
    values = {
        "direction": "LONG", "rvol": 1.35, "atr_expansion": 1.08,
        "rsi": 58.0, "ema_aligned": True, "macd_direction_ok": True,
        "macd_accel": True, "rsi_accel": True, "roc_direction_ok": True,
        "squeeze_release": True, "bw_expand": True,
        "compression_active": False, "structure_break": False,
    }
    values.update(overrides)
    return values


def test_explosion_uses_squeeze_release_and_directional_indicators():
    assert evaluate(indicators())


def test_explosion_does_not_require_candle_body_or_wick_fields():
    data = indicators()
    assert "body_ratio" not in data and "upper_wick_ratio" not in data
    assert evaluate(data)


@pytest.mark.parametrize("direction,rsi,macd,roc", [
    ("LONG", 58.0, True, True),
    ("SHORT", 42.0, True, True),
])
def test_explosion_supports_both_directions(direction, rsi, macd, roc):
    assert evaluate(indicators(direction=direction, rsi=rsi,
                               macd_direction_ok=macd, roc_direction_ok=roc))


def test_explosion_rejects_missing_or_weak_indicator_evidence():
    assert not evaluate(indicators(rvol=None))
    assert not evaluate(indicators(atr_expansion=0.99))
    assert not evaluate(indicators(rsi=49.0, direction="LONG"))
    assert not evaluate(indicators(macd_direction_ok=False, macd_accel=False,
                                   roc_direction_ok=False))


def test_pre_explosion_signals_before_squeeze_release_with_indicator_alignment():
    setup = indicators(squeeze_release=False, bw_expand=False,
                       compression_active=True, atr_expansion=1.01, rvol=1.15)
    assert is_pre_explosion_signal(setup)
    assert detect_momentum_state(setup) == "PRE_EXPLOSION"


def test_compression_without_directional_acceleration_is_watch_only():
    setup = indicators(squeeze_release=False, bw_expand=False,
                       compression_active=True, roc_direction_ok=False,
                       macd_direction_ok=False, macd_accel=False,
                       rsi_accel=False, ema_aligned=False)
    assert not is_pre_explosion_signal(setup)


def test_momentum_state_diagnostics_have_stable_keys():
    checks = diagnose_momentum_state(indicators())
    assert set(checks) == {
        "EXPLOSION", "SQUEEZE_RELEASE", "PRE_EXPLOSION",
        "CONTINUATION", "VOLUME_SPIKE",
    }
    assert detect_momentum_state(indicators()) in {
        "EXPLOSION", "SQUEEZE_RELEASE", "CONTINUATION", "VOLUME_SPIKE",
    }


def test_momentum_alert_score_floor_is_eight():
    from core.engine import _score_disposition

    assert _score_disposition(7.99, 7.0) == "WATCHLIST"
    assert _score_disposition(8.0, 7.0) == "SIGNAL"
    assert _score_disposition(8.5, 8.5) == "SIGNAL"
