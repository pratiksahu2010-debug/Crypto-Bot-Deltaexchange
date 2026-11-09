"""Normal engine: signals must fire on the trigger candle, not 2-3 candles later."""
import numpy as np
import pytest

from indicators.atr import atr
from indicators.ema import ema_series
from indicators.vwap import vwap
from strategies.normal.entry_trigger import (
    FRESH_CROSS, PULLBACK_RESUME, VWAP_RECLAIM,
    detect_fresh_trigger, normal_entry_filters, session_vwap_series,
)

DAY0 = 1_700_006_400_000  # an exact UTC-midnight boundary (ms)
FIVE_MIN = 300_000


def _ohlc(closes, wick=0.05):
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate(([closes[0]], closes[:-1]))
    highs = np.maximum(opens, closes) + wick
    lows = np.minimum(opens, closes) - wick
    vol = np.full(len(closes), 100.0)
    times = DAY0 + np.arange(len(closes)) * FIVE_MIN
    return opens, highs, lows, closes, vol, times


def _trigger(closes, direction, **kw):
    o, h, l, c, v, t = _ohlc(closes)
    e9, e21 = ema_series(c, 9), ema_series(c, 21)
    vw = session_vwap_series(h, l, c, v, t)
    a = atr(h, l, c, 14)
    return detect_fresh_trigger(o, h, l, c, e9, e21, vw, a, direction, **kw)


def _uptrend_with_pullback_then_resume(extra_up_bars=0):
    base = list(100 + 0.30 * np.arange(60))            # steady uptrend
    pullback = [base[-1] - 0.45, base[-1] - 0.90]      # two bars back to EMA9
    resume = [pullback[-1] + 0.80]                     # strong close above prior high
    tail = [resume[-1] + 0.30 * (i + 1) for i in range(extra_up_bars)]
    return base + pullback + resume + tail


def test_pullback_resume_fires_on_the_resume_candle():
    result = _trigger(_uptrend_with_pullback_then_resume(), "LONG")
    assert result == (PULLBACK_RESUME, 0)


def test_same_setup_is_rejected_when_it_is_two_or_three_candles_old():
    # This is the "late entry" case: trend state is still valid, trigger is gone.
    assert _trigger(_uptrend_with_pullback_then_resume(extra_up_bars=2), "LONG") is None
    assert _trigger(_uptrend_with_pullback_then_resume(extra_up_bars=3), "LONG") is None


def test_mature_trend_without_new_trigger_is_rejected():
    closes = list(100 + 0.30 * np.arange(80))
    assert _trigger(closes, "LONG") is None


def test_short_pullback_resume_mirror():
    base = list(200 - 0.30 * np.arange(60))
    closes = base + [base[-1] + 0.45, base[-1] + 0.90, base[-1] + 0.90 - 0.80]
    assert _trigger(closes, "SHORT") == (PULLBACK_RESUME, 0)


def test_wrong_direction_never_triggers():
    assert _trigger(_uptrend_with_pullback_then_resume(), "SHORT") is None


def _downtrend_then_v_reversal(bars_after_cross):
    down = list(150 - 0.30 * np.arange(60))
    up = [down[-1] + 0.55 * (i + 1) for i in range(12)]
    return down + up[:bars_after_cross]


def test_fresh_cross_age_is_bounded_by_max_trigger_age():
    # Find the candle where EMA9 first crosses above EMA21 in the V reversal.
    full = _downtrend_then_v_reversal(12)
    c = np.asarray(full)
    e9, e21 = ema_series(c, 9), ema_series(c, 21)
    cross_idx = next(i for i in range(61, len(c)) if e9[i] > e21[i] and e9[i - 1] <= e21[i - 1])
    on_cross = full[: cross_idx + 1]
    one_late = full[: cross_idx + 2]
    three_late = full[: cross_idx + 4]

    t0 = _trigger(on_cross, "LONG")
    assert t0 is not None and t0[1] == 0
    t1 = _trigger(one_late, "LONG")
    assert t1 is not None and t1[1] <= 1
    # With the default max age of 1 candle, a cross three candles old is stale.
    t3 = _trigger(three_late, "LONG")
    assert t3 is None or t3[0] != FRESH_CROSS


def test_vwap_reclaim_trigger_and_its_age_limit():
    # Isolate the VWAP rule: EMAs are bullish and far from price (so neither the
    # pullback nor the cross rule can fire); only the VWAP cross changes.
    n = 40
    vw = np.full(n, 100.0)
    e9, e21 = np.full(n, 110.0), np.full(n, 105.0)

    def run(closes):
        c = np.asarray(closes, dtype=float)
        o = np.concatenate(([c[0]], c[:-1]))
        h, l = np.maximum(o, c) + 0.05, np.minimum(o, c) - 0.05
        return detect_fresh_trigger(o, h, l, c, e9, e21, vw, 0.5, "LONG")

    below = [99.0] * (n - 1)
    assert run(below + [100.6]) == (VWAP_RECLAIM, 0)                 # crossed this candle
    assert run(below[:-1] + [100.4, 100.7]) == (VWAP_RECLAIM, 1)     # crossed one candle ago
    assert run(below[:-2] + [100.4, 100.7, 100.8]) is None            # two candles ago: stale
    assert run([101.0] * n) is None                                   # long-standing, no cross


def test_session_vwap_series_matches_scalar_vwap():
    o, h, l, c, v, t = _ohlc(100 + np.sin(np.arange(50)))
    v = v * (1 + np.arange(50) % 5)
    series = session_vwap_series(h, l, c, v, t)
    assert series[-1] == pytest.approx(vwap(h, l, c, v, period=None))
    assert np.isfinite(series).all()  # whole series is inside one UTC day


def test_session_vwap_series_is_nan_before_session_start():
    n = 30
    times = DAY0 - 10 * FIVE_MIN + np.arange(n) * FIVE_MIN  # first 10 bars in previous day
    c = 100 + np.arange(n) * 0.1
    series = session_vwap_series(c + 0.1, c - 0.1, c, np.full(n, 10.0), times)
    assert np.isnan(series[:10]).all() and np.isfinite(series[10:]).all()


def test_trigger_rejects_bad_inputs():
    o, h, l, c, v, t = _ohlc(100 + 0.3 * np.arange(60))
    e9, e21 = ema_series(c, 9), ema_series(c, 21)
    vw = session_vwap_series(h, l, c, v, t)
    assert detect_fresh_trigger(o, h, l, c, e9, e21, vw, 0.0, "LONG") is None
    assert detect_fresh_trigger(o, h, l, c, e9, e21, vw, float("nan"), "LONG") is None
    assert detect_fresh_trigger(o, h, l, c, e9, e21, vw, 1.0, "FLAT") is None
    assert detect_fresh_trigger(o[:10], h[:10], l[:10], c[:10], e9[:10], e21[:10], vw[:10], 1.0, "LONG") is None


def test_normal_supplementary_filter_only_requires_valid_atr():
    assert normal_entry_filters({"atr": 0.5}, "LONG") == {"atr_valid": True}
    assert normal_entry_filters({"atr": 0.0}, "LONG") == {"atr_valid": False}
    assert normal_entry_filters({}, "SHORT") == {"atr_valid": False}


def test_normal_engine_uses_the_requested_six_indicator_gates():
    from core.engine import _normal_threshold_gates

    indicators = {
        "ema9": 101.0, "ema21": 100.0, "close": 101.5, "vwap": 100.5,
        "rsi": 55.0, "adx": 25.0, "atr": 0.5, "confirm_15m": True,
    }
    gates = _normal_threshold_gates(indicators, "LONG", 20.0)
    assert set(gates) == {
        "ema_9_21", "vwap_direction", "rsi_direction",
        "adx_min_25", "atr_valid", "context_15m",
    }
    assert all(gates.values())
    # The production hard floor is 25 even when a caller supplies 20.
    assert _normal_threshold_gates({**indicators, "adx": 24.99}, "LONG", 20.0)["adx_min_25"] is False
    assert _normal_threshold_gates({**indicators, "rsi": 49.0}, "LONG", 20.0)["rsi_direction"] is False
    assert _normal_threshold_gates({**indicators, "confirm_15m": False}, "LONG", 20.0)["context_15m"] is False


def test_normal_short_uses_mirrored_indicator_direction():
    from core.engine import _normal_threshold_gates

    indicators = {
        "ema9": 99.0, "ema21": 100.0, "close": 98.5, "vwap": 99.0,
        "rsi": 42.0, "adx": 25.0, "atr": 0.6, "confirm_15m": True,
    }
    assert all(_normal_threshold_gates(indicators, "SHORT", 20.0).values())
    assert not _normal_threshold_gates({**indicators, "rsi": 51.0}, "SHORT", 20.0)["rsi_direction"]


def test_normal_indicator_trigger_ignores_candle_colour_and_body_shape():
    # Same close/indicator series with radically different open/high/low values
    # must produce the same trigger; candle geometry is not a signal gate.
    n = 40
    close = np.linspace(100.0, 101.0, n)
    e9, e21 = np.full(n, 101.0), np.full(n, 100.0)
    e9[-1], e21[-1] = 101.2, 100.8
    vwap_values = np.full(n, 100.5)
    base = np.full(n, 100.0)
    one = detect_fresh_trigger(base, base + 0.1, base - 0.1, close,
                               e9, e21, vwap_values, 0.5, "LONG")
    two = detect_fresh_trigger(close + 2.0, close + 5.0, close - 5.0, close,
                               e9, e21, vwap_values, 0.5, "LONG")
    assert one == two


def test_normal_adx_config_can_only_raise_the_required_floor():
    from core.engine import _normal_threshold_gates

    base = {
        "ema9": 101.0, "ema21": 100.0, "close": 101.5, "vwap": 100.5,
        "rsi": 55.0, "adx": 24.99, "atr": 0.5, "confirm_15m": True,
    }
    assert _normal_threshold_gates(base, "LONG", 18.0)["adx_min_25"] is False
    assert _normal_threshold_gates(base, "LONG", 22.0)["adx_min_25"] is False
    assert _normal_threshold_gates({**base, "adx": 25.0}, "LONG", 22.0)["adx_min_25"] is True


def test_normal_confirmation_enforces_hard_adx_and_volume_floors(monkeypatch):
    from core.engine import _signal_follow_through_confirmed
    from config import settings

    monkeypatch.setattr(settings, "normal_adx_min", 22.0)
    indicators = {
        "ema9": 101.0, "ema21": 100.0, "vwap": 100.5, "atr": 0.5,
        "confirm_15m": True, "rsi": 56.0, "adx": 25.0,
        "adx_rising": True, "plus_di": 30.0, "minus_di": 20.0,
        "rvol": 1.4,
    }
    assert _signal_follow_through_confirmed(
        "normal", "LONG", "TREND_CONTINUATION", indicators, 101.0, 101.2
    )
    indicators["adx"] = 24.99
    assert not _signal_follow_through_confirmed(
        "normal", "LONG", "TREND_CONTINUATION", indicators, 101.0, 101.2
    )
    indicators["adx"] = 25.0
    indicators["rvol"] = 1.39
    assert not _signal_follow_through_confirmed(
        "normal", "LONG", "TREND_CONTINUATION", indicators, 101.0, 101.2
    )


def test_normal_engine_enforces_adx_floor_of_25_even_if_configured_lower():
    from core.engine import _normal_threshold_gates

    indicators = {
        "ema9": 101.0, "ema21": 100.0, "close": 101.5, "vwap": 100.5,
        "rsi": 55.0, "adx": 24.99, "atr": 0.5, "confirm_15m": True,
    }
    assert _normal_threshold_gates(indicators, "LONG", 20.0)["adx_min_25"] is False
    assert _normal_threshold_gates({**indicators, "adx": 25.0}, "LONG", 20.0)["adx_min_25"] is True
