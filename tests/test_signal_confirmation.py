from core.engine import _signal_follow_through_confirmed


def base(**overrides):
    values = {
        "ema9": 101.0,
        "ema21": 100.0,
        "vwap": 100.5,
        "atr": 2.0,
        "rsi": 58.0,
        "adx": 26.0,
        "confirm_15m": True,
        "roc_direction_ok": True,
        "macd_direction_ok": True,
        "rvol": 1.5,
        "adx_rising": True,
        "plus_di": 30.0,
        "minus_di": 20.0,
        "atr_expansion": 1.08,
    }
    values.update(overrides)
    return values


def test_normal_requires_second_close_to_advance_and_hold_indicator_alignment():
    assert _signal_follow_through_confirmed(
        "normal", "LONG", "TREND_CONTINUATION", base(), 101.2, 101.5
    )


def test_normal_rejects_flat_or_reversing_confirmation_close():
    assert not _signal_follow_through_confirmed(
        "normal", "LONG", "TREND_CONTINUATION", base(), 101.2, 101.2
    )
    assert not _signal_follow_through_confirmed(
        "normal", "LONG", "TREND_CONTINUATION", base(vwap=101.6), 101.2, 101.5
    )


def test_momentum_requires_directional_acceleration_on_confirmation_bar():
    assert _signal_follow_through_confirmed(
        "momentum", "LONG", "EXPLOSION", base(), 101.2, 101.5
    )
    assert not _signal_follow_through_confirmed(
        "momentum", "LONG", "EXPLOSION",
        base(roc_direction_ok=False, macd_direction_ok=False), 101.2, 101.5
    )


def test_orb_confirmation_must_hold_beyond_opening_range_edge():
    indicators = base(orb_range_high=101.3)
    assert _signal_follow_through_confirmed(
        "momentum", "LONG", "ORB_BREAKOUT", indicators, 101.4, 101.6
    )
    assert not _signal_follow_through_confirmed(
        "momentum", "LONG", "ORB_BREAKOUT", indicators, 101.4, 101.2
    )


def test_short_confirmation_uses_inverse_direction():
    indicators = base(
        ema9=99.0, ema21=100.0, vwap=99.5, rsi=42.0,
        plus_di=20.0, minus_di=30.0,
    )
    assert _signal_follow_through_confirmed(
        "normal", "SHORT", "TREND_CONTINUATION", indicators, 98.8, 98.5
    )

def test_normal_long_confirmation_requires_rsi_above_50_not_reversal_recovery():
    assert _signal_follow_through_confirmed(
        "normal", "LONG", "TREND_CONTINUATION", base(rsi=55.0), 101.2, 101.5
    )
    assert not _signal_follow_through_confirmed(
        "normal", "LONG", "TREND_CONTINUATION", base(rsi=38.0, rsi_prev=36.0), 101.2, 101.5
    )


def test_normal_confirmation_rejects_weak_volume_or_flat_adx():
    assert not _signal_follow_through_confirmed(
        "normal", "LONG", "TREND_CONTINUATION",
        base(rvol=1.2), 101.2, 101.5,
    )
    assert not _signal_follow_through_confirmed(
        "normal", "LONG", "TREND_CONTINUATION",
        base(adx=24.0, adx_rising=False), 101.2, 101.5,
    )


def test_orb_prebreakout_requires_real_strength_and_participation():
    indicators = base(orb_range_high=101.8)
    assert _signal_follow_through_confirmed(
        "momentum", "LONG", "ORB_PREBREAKOUT", indicators, 101.2, 101.5,
    )
    assert not _signal_follow_through_confirmed(
        "momentum", "LONG", "ORB_PREBREAKOUT",
        base(orb_range_high=101.8, adx=19.0), 101.2, 101.5,
    )


def test_2r_room_filter_rejects_nearby_resistance_and_support():
    import numpy as np
    from core.engine import _has_room_for_2r_target

    highs = np.array([100.5, 101.0, 102.0, 101.0, 100.5, 100.4, 100.3,
                      100.2, 100.1, 100.0, 100.0, 100.0])
    lows = np.array([99.5, 99.0, 99.2, 99.0, 99.5, 99.6, 99.7,
                     99.8, 99.9, 99.9, 99.9, 99.9])
    assert not _has_room_for_2r_target(highs, lows, 100.0, 103.0, 1.0, "LONG")
    assert _has_room_for_2r_target(highs, lows, 100.0, 101.5, 1.0, "LONG")

    short_lows = np.array([99.5, 99.0, 98.0, 99.0, 99.5, 99.6, 99.7,
                           99.8, 99.9, 99.9, 99.9, 99.9])
    short_highs = np.array([100.5, 101.0, 100.8, 101.0, 100.5, 100.4, 100.3,
                            100.2, 100.1, 100.0, 100.0, 100.0])
    assert not _has_room_for_2r_target(short_highs, short_lows, 100.0, 97.0, 1.0, "SHORT")
    assert _has_room_for_2r_target(short_highs, short_lows, 100.0, 98.5, 1.0, "SHORT")
