from core.scoring_engine import ScoringEngine


def orb_quality(**overrides):
    values = {
        "rvol": 1.9, "orb_session_rvol": 1.9, "close": 102.0, "vwap": 101.0,
        "ema9": 101.5, "ema21": 100.0, "rsi": 60.0, "atr": 0.5,
        "orb_breakout_extension_atr": 0.25,
        "macd_direction_ok": True, "roc_direction_ok": True,
    }
    values.update(overrides)
    return values


def test_orb_score_uses_ten_point_indicator_scale():
    result = ScoringEngine().score_orb(orb_quality(), "LONG")
    assert result.total == 10.0
    assert result.total <= 10.0
    assert all("Candle" not in component.name and "Wick" not in component.name
               for component in result.components)


def test_orb_score_changes_when_directional_indicators_disagree():
    aligned = ScoringEngine().score_orb(orb_quality(), "LONG")
    opposed = ScoringEngine().score_orb(
        orb_quality(ema9=99.0, ema21=100.0, rsi=42.0,
                    macd_direction_ok=False, roc_direction_ok=False), "LONG"
    )
    assert aligned.total > opposed.total


def test_orb_short_score_uses_mirrored_indicator_direction():
    result = ScoringEngine().score_orb(
        orb_quality(close=98.0, vwap=99.0, ema9=98.5, ema21=100.0,
                    rsi=42.0, orb_breakout_extension_atr=0.2), "SHORT"
    )
    assert result.total == 10.0
