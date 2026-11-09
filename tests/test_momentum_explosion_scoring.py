from core.scoring_engine import ScoringEngine


def early_indicator_setup(**overrides):
    values = {
        "direction": "LONG", "compression_active": True, "squeeze_release": False,
        "atr_expansion": 1.05, "rvol": 1.35, "rsi": 58.0,
        "ema_aligned": True, "macd_direction_ok": True, "macd_accel": True,
        "rsi_accel": True, "roc_direction_ok": True, "bw_expand": False,
    }
    values.update(overrides)
    return values


def test_pre_explosion_score_reaches_signal_floor_without_candle_geometry():
    result = ScoringEngine().score_momentum_explosion(early_indicator_setup(), "LONG")
    assert result.total >= 7.0
    assert result.total <= 10.0
    assert result.grade in ("SIGNAL", "STRONG", "HIGH-CONVICTION")
    assert all("Candle" not in component.name and "Close" not in component.name
               for component in result.components)


def test_short_early_explosion_uses_mirrored_rsi_and_roc():
    result = ScoringEngine().score_momentum_explosion(
        early_indicator_setup(direction="SHORT", rsi=42.0,
                              roc_direction_ok=True, ema_aligned=True), "SHORT"
    )
    assert result.total >= 7.0


def test_explosion_score_rewards_additional_indicator_confirmation():
    base = ScoringEngine().score_momentum_explosion(
        early_indicator_setup(ema_aligned=False, macd_direction_ok=False,
                              macd_accel=False, rsi_accel=False, roc_direction_ok=False), "LONG"
    )
    confirmed = ScoringEngine().score_momentum_explosion(early_indicator_setup(), "LONG")
    assert confirmed.total > base.total
