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
        "adx": 22.0, "plus_di": 28.0, "minus_di": 16.0,
    }
    values.update(overrides)
    if values["direction"] == "SHORT" and "plus_di" not in overrides and "minus_di" not in overrides:
        values["plus_di"], values["minus_di"] = 16.0, 28.0
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
    assert not evaluate(indicators(rsi=49.0, direction="LONG", rsi_accel=False, macd_accel=False))
    assert not evaluate(indicators(macd_direction_ok=False, macd_accel=False,
                                   roc_direction_ok=True, rsi_accel=False))
    assert not evaluate(indicators(roc_direction_ok=False, macd_accel=False,
                                   rsi_accel=False))
    assert not evaluate(indicators(adx=17.99))
    assert not evaluate(indicators(plus_di=10.0, minus_di=25.0))
    assert not evaluate(indicators(rvol=1.19))


def test_pre_explosion_signals_before_squeeze_release_with_indicator_alignment():
    setup = indicators(squeeze_release=False, bw_expand=False,
                       compression_active=True, atr_expansion=1.04, rvol=1.25)
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


def test_continuation_requires_all_directional_families_and_adx():
    from strategies.momentum.strategy_manager import diagnose_momentum_state

    base = indicators(squeeze_release=False, bw_expand=False)
    assert diagnose_momentum_state(base)["CONTINUATION"]
    assert not diagnose_momentum_state({**base, "macd_direction_ok": False})["CONTINUATION"]
    assert not diagnose_momentum_state({**base, "ema_aligned": False})["CONTINUATION"]
    assert not diagnose_momentum_state({**base, "adx": 19.99})["CONTINUATION"]
    assert not diagnose_momentum_state({**base, "rvol": 1.10})["CONTINUATION"]


def test_squeeze_release_needs_macd_roc_and_directional_di():
    from strategies.momentum.squeeze_detector import evaluate as squeeze_release

    assert squeeze_release(indicators())
    assert not squeeze_release(indicators(macd_direction_ok=False))
    assert not squeeze_release(indicators(plus_di=10.0, minus_di=30.0))


def test_volume_spike_requires_actual_atr_expansion():
    from strategies.momentum.volume_expansion import evaluate as volume_spike

    setup = indicators(rvol=1.8, atr_expansion=1.08)
    assert volume_spike(setup)
    assert not volume_spike({**setup, "atr_expansion": 1.02})
    assert not volume_spike({**setup, "rvol": 1.40})



def test_explosion_can_confirm_without_a_textbook_squeeze():
    setup = indicators(
        squeeze_release=False, compression_active=False,
        bw_expand=False, atr_expansion=1.16,
    )
    assert evaluate(setup)


def test_explosion_uses_rising_adx_early_band_and_directional_votes():
    setup = indicators(
        adx=15.5, adx_rising=True, atr_expansion=1.04, rvol=1.22,
        macd_direction_ok=True, roc_direction_ok=True,
    )
    assert evaluate(setup)


def test_explosion_rejects_opposing_di_without_rising_adx():
    assert not evaluate(indicators(plus_di=10.0, minus_di=28.0, adx_rising=False))

def test_stage_two_qualified_mex_is_not_lost_to_a_second_score_gate():
    from core.scoring_engine import ScoringEngine

    setup = indicators(
        rvol=1.20, atr_expansion=1.12, rsi=52.0,
        ema_aligned=True, structure_break=False,
        macd_direction_ok=False, macd_accel=False,
        roc_direction_ok=True, rsi_accel=True,
        squeeze_release=False, compression_active=False, bw_expand=False,
        adx=18.0, adx_rising=False,
    )
    assert evaluate(setup)
    score = ScoringEngine().score_momentum_explosion(setup, "LONG")
    assert score.total >= 8.0, score
    assert score.max_total == 10.0
