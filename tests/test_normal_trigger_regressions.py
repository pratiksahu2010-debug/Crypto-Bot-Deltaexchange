"""Regression tests for Normal trigger age, VWAP data hygiene, and direction symmetry."""
import numpy as np
import pytest

from strategies.normal.entry_trigger import (
    FRESH_CROSS, detect_fresh_trigger, session_vwap_series,
    resolve_symbol_pullback_touch_atr,
)


DAY = 86_400_000


def test_session_vwap_skips_nan_bar_without_poisoning_following_values():
    times = np.array([DAY + i * 300_000 for i in range(4)], dtype=np.int64)
    high = np.array([11.0, np.nan, 13.0, 14.0])
    low = np.array([9.0, 9.0, 11.0, 12.0])
    close = np.array([10.0, 10.0, 12.0, 13.0])
    volume = np.array([2.0, 5.0, 2.0, 2.0])

    result = session_vwap_series(high, low, close, volume, times)

    assert result[0] == pytest.approx(10.0)
    assert np.isnan(result[1])
    # Invalid row did not enter the cumulative sums.
    assert result[2] == pytest.approx((10.0 * 2 + 12.0 * 2) / 4)
    assert result[3] == pytest.approx((10.0 * 2 + 12.0 * 2 + 13.0 * 2) / 6)


def test_session_vwap_resets_at_utc_day_boundary():
    times = np.array([DAY - 300_000, DAY, DAY + 300_000], dtype=np.int64)
    high = np.array([102.0, 202.0, 204.0])
    low = np.array([100.0, 200.0, 202.0])
    close = np.array([101.0, 201.0, 203.0])
    volume = np.array([10.0, 1.0, 3.0])

    result = session_vwap_series(high, low, close, volume, times)

    assert np.isnan(result[0])
    assert result[1] == pytest.approx(201.0)
    assert result[2] == pytest.approx((201.0 + 203.0 * 3) / 4)


@pytest.mark.parametrize("direction,sign", [("LONG", 1.0), ("SHORT", -1.0)])
@pytest.mark.parametrize("max_age,expected_age", [(0, 0), (1, 1), (2, 2)])
def test_fresh_cross_age_semantics_are_independent_for_long_short(
    direction, sign, max_age, expected_age
):
    n = 40
    opens = np.full(n, 100.0)
    highs = np.full(n, 101.0)
    lows = np.full(n, 99.0)
    closes = np.full(n, 100.0)
    ema9 = np.full(n, 100.0)
    ema21 = np.full(n, 100.0)
    vwap = np.full(n, np.nan)

    # A single directional EMA crossing at the exact expected age.
    crossing_idx = n - 1 - expected_age
    ema9[crossing_idx:] = 100.0 + sign
    ema21[crossing_idx:] = 100.0

    result = detect_fresh_trigger(
        opens, highs, lows, closes, ema9, ema21, vwap, 1.0, direction,
        max_trigger_age=max_age, pullback_lookback=1, pullback_touch_atr=0.25,
    )
    assert result == (FRESH_CROSS, expected_age)


@pytest.mark.parametrize("direction,sign", [("LONG", 1.0), ("SHORT", -1.0)])
def test_fresh_cross_rejects_cross_older_than_max_age(direction, sign):
    n = 40
    arrays = [np.full(n, 100.0) for _ in range(6)]
    opens, highs, lows, closes, ema9, ema21 = arrays
    vwap = np.full(n, np.nan)
    ema9[-4:] = 100.0 + sign
    ema21[-4:] = 100.0

    result = detect_fresh_trigger(
        opens, highs, lows, closes, ema9, ema21, vwap, 1.0, direction,
        max_trigger_age=1, pullback_lookback=1,
    )
    assert result is None


def test_vwap_series_rejects_mismatched_lengths_and_unsorted_timestamps():
    valid = np.array([10.0, 11.0])
    volume = np.array([1.0, 1.0])
    times = np.array([DAY, DAY + 300_000], dtype=np.int64)
    assert np.isnan(session_vwap_series(valid, valid, valid, volume, times[:1])).all()
    assert np.isnan(session_vwap_series(valid, valid, valid, volume, times[::-1])).all()



@pytest.mark.parametrize("direction,sign", [("LONG", 1.0), ("SHORT", -1.0)])
def test_vwap_reclaim_direction_logic_independently(direction, sign):
    n = 40
    opens = np.full(n, 100.0)
    highs = np.full(n, 100.5)
    lows = np.full(n, 99.5)
    closes = np.full(n, 100.0)
    ema9 = np.full(n, 100.0)
    ema21 = np.full(n, 100.0)
    vwap = np.full(n, 100.0)

    # Keep pullback-touch conditions false so this isolates VWAP reclaim.
    if direction == "LONG":
        lows[-4:] = 100.8
        vwap[-2:] = [100.2, 100.2]
        opens[-1], closes[-1], highs[-1], lows[-1] = 100.5, 101.0, 101.1, 100.0
        ema9[-1], ema21[-1] = 100.8, 100.0
    else:
        highs[-4:] = 99.2
        vwap[-2:] = [99.8, 99.8]
        opens[-1], closes[-1], highs[-1], lows[-1] = 99.5, 99.0, 100.0, 98.9
        ema9[-1], ema21[-1] = 99.2, 100.0

    result = detect_fresh_trigger(
        opens, highs, lows, closes, ema9, ema21, vwap, 1.0, direction,
        max_trigger_age=0, pullback_lookback=2,
    )
    assert result == ("VWAP_RECLAIM", 0)


@pytest.mark.parametrize("direction,sign", [("LONG", 1.0), ("SHORT", -1.0)])
def test_pullback_resume_direction_logic_independently(direction, sign):
    n = 40
    opens = np.full(n, 100.0)
    highs = np.full(n, 100.2)
    lows = np.full(n, 99.8)
    closes = np.full(n, 100.0)
    ema9 = np.full(n, 100.5)
    ema21 = np.full(n, 100.0)
    vwap = np.full(n, np.nan)

    if direction == "LONG":
        # Recent prior candle tags EMA9 zone, preserves EMA21, then the
        # trigger candle closes above previous high and EMA9 with green body.
        lows[-3] = 100.0
        closes[-3] = 100.4
        closes[-2] = 100.4
        opens[-1], closes[-1], highs[-1], lows[-1] = 100.4, 101.0, 101.1, 100.8
        ema9[-1], ema21[-1] = 100.5, 100.0
    else:
        ema9[:] = 99.5
        ema21[:] = 100.0
        highs[-3] = 100.0
        closes[-3] = 99.6
        closes[-2] = 99.6
        opens[-1], closes[-1], highs[-1], lows[-1] = 99.6, 99.0, 99.2, 98.9
        ema9[-1], ema21[-1] = 99.5, 100.0

    result = detect_fresh_trigger(
        opens, highs, lows, closes, ema9, ema21, vwap, 1.0, direction,
        max_trigger_age=0, pullback_lookback=2, pullback_touch_atr=0.25,
    )
    assert result == ("PULLBACK_RESUME", 0)


@pytest.mark.parametrize(
    "pair,overrides,expected",
    [
        ("B-BTC_USDT", {"BTCUSDT": 0.18}, 0.18),
        ("ETHUSDT", {"eth_usdt": 0.35}, 0.35),
        ("B-SOL_USDT", {"SOL-USDT": 0.22}, 0.22),
        ("B-ADA_USDT", {"ADAUSDT": float("nan")}, 0.25),
        ("B-XRP_USDT", {"XRPUSDT": 1.5}, 0.25),
        ("B-DOGE_USDT", {}, 0.25),
    ],
)
def test_symbol_pullback_touch_atr_override_normalization_and_fallback(pair, overrides, expected):
    assert resolve_symbol_pullback_touch_atr(pair, 0.25, overrides) == pytest.approx(expected)
