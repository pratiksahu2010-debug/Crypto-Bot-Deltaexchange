"""Live (intrabar) Explosion + ORB ignition: deterministic, offline."""
from datetime import datetime, timezone

import numpy as np
import pytest

from strategies.momentum import orb_detector
from strategies.momentum.live_ignition import (
    LiveConfig, LiveTracker, build_context, evaluate_explosion,
)

INTERVAL = 300_000
# 2026-11-09 00:00:00 UTC == 09:00 Tokyo (Asia ORB open); Asia has no DST.
ASIA_OPEN_MS = int(datetime(2026, 11, 9, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)


def make_arrays(forming_open_ms, n=100, skip_last=False):
    """Flat, slightly oscillating market: ATR ~0.3, avg volume 1000."""
    times = forming_open_ms - INTERVAL * np.arange(n, 0, -1)
    opens = np.full(n, 100.0)
    closes = 100.0 + np.where(np.arange(n) % 2 == 0, 0.1, -0.1)
    highs = np.maximum(opens, closes) + 0.1
    lows = np.minimum(opens, closes) - 0.1
    arrays = {"time": times.astype(np.int64), "open": opens, "high": highs,
              "low": lows, "close": closes, "volume": np.full(n, 1000.0)}
    if skip_last:
        arrays = {k: np.delete(v, -1) for k, v in arrays.items()}
    return arrays


def forming(open_ms, o, h, l, c, v):
    return {"time": open_ms, "open": o, "high": h, "low": l, "close": c, "volume": v}


def ctx_for(forming_open_ms, now_ms, sessions=()):
    ctx = build_context("B-TEST_USDT", make_arrays(forming_open_ms), forming_open_ms,
                        INTERVAL, now_ms, sessions=sessions)
    # Make the completed indicator context unambiguously bullish by default.
    ctx.ema9, ctx.ema21, ctx.last_close = 100.1, 99.9, 100.0
    return ctx


def test_context_atr_and_structure_levels():
    t = 1_800_000_000_000 - (1_800_000_000_000 % INTERVAL)
    ctx = ctx_for(t, t + 90_000)
    assert ctx is not None and 0.2 < ctx.atr < 0.5
    assert ctx.resistance == pytest.approx(100.2) and ctx.support == pytest.approx(99.8)


def test_context_rejects_gap_before_forming_candle():
    t = 1_800_000_000_000 - (1_800_000_000_000 % INTERVAL)
    arrays = make_arrays(t, skip_last=True)
    assert build_context("B-TEST_USDT", arrays, t, INTERVAL, t + 90_000) is None


def test_clean_long_impulse_fires_early_in_the_candle():
    t = 1_800_000_000_000 - (1_800_000_000_000 % INTERVAL)
    now = t + 90_000                                  # 30% into the candle
    ctx = ctx_for(t, now)
    c = 100.0 + 1.5 * ctx.atr
    res = evaluate_explosion(
        ctx, forming(t, 100.0, c + 0.01, 99.98, c, 600.0), now,
        LiveConfig(impulse_pace=1.8),
    )
    assert res["state"] == "LIVE_EXPLOSION", res
    assert res["direction"] == "LONG" and res["strategy"] == "EXPLOSION"
    assert res["elapsed_seconds"] == 90.0
    assert res["swing_low"] is None and res["score_result"].total > 0


def test_clean_short_impulse_is_mirrored():
    t = 1_800_000_000_000 - (1_800_000_000_000 % INTERVAL)
    now = t + 90_000
    ctx = ctx_for(t, now)
    c = 100.0 - 1.5 * ctx.atr
    ctx.ema9, ctx.ema21 = 99.9, 100.1
    res = evaluate_explosion(
        ctx, forming(t, 100.0, 100.02, c - 0.01, c, 600.0), now,
        LiveConfig(impulse_pace=1.8),
    )
    assert res["state"] == "LIVE_EXPLOSION" and res["direction"] == "SHORT"
    assert res["swing_high"] is None


def test_rejects_move_that_already_happened():
    t = 1_800_000_000_000 - (1_800_000_000_000 % INTERVAL)
    now = t + 90_000
    ctx = ctx_for(t, now)
    c = 100.0 + 2.6 * ctx.atr
    res = evaluate_explosion(ctx, forming(t, 100.0, c + 0.01, 99.98, c, 900.0), now)
    assert res["reason"] == "already_extended"


def test_rejects_small_move():
    t = 1_800_000_000_000 - (1_800_000_000_000 % INTERVAL)
    now = t + 90_000
    ctx = ctx_for(t, now)
    c = 100.0 + 0.6 * ctx.atr
    res = evaluate_explosion(ctx, forming(t, 100.0, c + 0.01, 99.98, c, 900.0), now)
    assert res["reason"] == "move_too_small"


def test_rejects_low_time_normalised_volume():
    t = 1_800_000_000_000 - (1_800_000_000_000 % INTERVAL)
    now = t + 200_000                                  # 67% in: needs >= 1.07x avg volume
    ctx = ctx_for(t, now)
    c = 100.0 + 1.5 * ctx.atr
    res = evaluate_explosion(ctx, forming(t, 100.0, c + 0.01, 99.98, c, 500.0), now)
    assert res["reason"] == "low_volume_pace"


def test_rejects_too_early_and_too_late():
    t = 1_800_000_000_000 - (1_800_000_000_000 % INTERVAL)
    ctx = ctx_for(t, t + 5_000)
    c = 100.0 + 1.5 * ctx.atr
    candle = forming(t, 100.0, c + 0.01, 99.98, c, 600.0)
    assert evaluate_explosion(ctx, candle, t + 5_000)["reason"] == "outside_live_window"
    assert evaluate_explosion(ctx, candle, t + 270_000)["reason"] == "outside_live_window"


def test_large_opposing_wick_does_not_reject_indicator_qualified_signal():
    t = 1_800_000_000_000 - (1_800_000_000_000 % INTERVAL)
    now = t + 90_000
    ctx = ctx_for(t, now)
    c = 100.0 + 1.3 * ctx.atr
    res = evaluate_explosion(ctx, forming(t, 100.0, c + 2.0 * ctx.atr, 99.98, c, 800.0), now)
    assert res["state"] == "LIVE_EXPLOSION", res


def test_rejects_when_run_is_already_old():
    t = 1_800_000_000_000 - (1_800_000_000_000 % INTERVAL)
    now = t + 90_000
    ctx = ctx_for(t, now)
    ctx.ema21 = 100.0 - 3.0 * ctx.atr                  # candle opens 3 ATR above EMA21
    c = 100.0 + 1.5 * ctx.atr
    res = evaluate_explosion(ctx, forming(t, 100.0, c + 0.01, 99.98, c, 600.0), now)
    assert res["reason"] == "pre_extended"


def test_requires_ignition_evidence():
    t = 1_800_000_000_000 - (1_800_000_000_000 % INTERVAL)
    now = t + 90_000
    ctx = ctx_for(t, now)
    ctx.resistance = 200.0                             # no structure break possible
    ctx.compression_recent = False
    c = 100.0 + 1.25 * ctx.atr                         # pace stays below the impulse bar
    res = evaluate_explosion(ctx, forming(t, 100.0, c + 0.01, 99.98, c, 520.0), now)
    assert res["reason"] == "no_ignition_evidence"


def test_tracker_returns_best_candidate_and_ignores_stale_bucket():
    t = 1_800_000_000_000 - (1_800_000_000_000 % INTERVAL)
    now = t + 90_000
    tracker = LiveTracker(LiveConfig(sessions=(), impulse_pace=1.8))
    ctx = ctx_for(t, now)
    tracker._ctx["B-TEST_USDT"] = ctx
    c = 100.0 + 1.5 * ctx.atr
    candle = forming(t, 100.0, c + 0.01, 99.98, c, 600.0)
    found = tracker.evaluate("B-TEST_USDT", candle, INTERVAL, now, lambda: make_arrays(t))
    assert [r["state"] for r in found] == ["LIVE_EXPLOSION"]
    # An event for a bucket that is no longer forming must be ignored.
    assert tracker.evaluate("B-TEST_USDT", candle, INTERVAL, t + INTERVAL + 1,
                            lambda: make_arrays(t)) == []


def test_one_attempt_per_pair_direction_candle():
    tracker = LiveTracker(LiveConfig())
    assert tracker.first_attempt("B-X_USDT", "LONG", 1) is True
    assert tracker.first_attempt("B-X_USDT", "LONG", 1) is False
    assert tracker.first_attempt("B-X_USDT", "SHORT", 1) is True


# ---------------------------------------------------------------- ORB ------

def asia_context(range_candle_overrides=None):
    forming_open = ASIA_OPEN_MS + 30 * 60_000          # first candle after the 15-min range
    arrays = make_arrays(forming_open)
    now = forming_open + 60_000
    ctx = build_context("B-TEST_USDT", arrays, forming_open, INTERVAL, now, sessions=("asia",))
    return forming_open, now, ctx


def test_opening_range_ready_only_after_range_end():
    arrays = make_arrays(ASIA_OPEN_MS + 30 * 60_000)
    completed = {k: arrays[k] for k in ("time", "open", "high", "low", "close")}
    early = orb_detector.opening_range(completed, "asia", ASIA_OPEN_MS + 10 * 60_000)
    ready = orb_detector.opening_range(completed, "asia", ASIA_OPEN_MS + 30 * 60_000 + 1_000)
    assert early["state"] == "BUILDING_RANGE"
    assert ready["state"] == "READY" and ready["range_high"] == pytest.approx(100.2)
    late = orb_detector.opening_range(completed, "asia", ASIA_OPEN_MS + 5 * 3_600_000)
    assert late["state"] == "SESSION_CLOSED"


def test_live_orb_waits_for_hold_then_fires_once_per_direction():
    forming_open, now, ctx = asia_context()
    assert "asia" in ctx.orb
    tracker = LiveTracker(LiveConfig(sessions=("asia",)))
    rh = ctx.orb["asia"]["range_high"]
    c = rh + 0.25 * ctx.atr
    candle = forming(forming_open, 100.0, c + 0.01, 99.98, c, 400.0)
    first = tracker.evaluate("B-TEST_USDT", candle, INTERVAL, now - 20_000 + 10_000,
                             lambda: make_arrays(forming_open))
    # first sighting beyond the edge only starts the hold timer
    assert all(r["state"] != "LIVE_ORB" for r in first)
    fired = tracker.evaluate("B-TEST_USDT", candle, INTERVAL, now + 10_000,
                             lambda: make_arrays(forming_open))
    orb = [r for r in fired if r["state"] == "LIVE_ORB"]
    assert len(orb) == 1, fired
    assert orb[0]["direction"] == "LONG" and orb[0]["session"] == "asia"
    assert orb[0]["strategy"] == "ORB_BREAKOUT" and orb[0]["swing_low"] == ctx.orb["asia"]["range_low"]
    tracker.orb_fired.add(("B-TEST_USDT", "asia", orb[0]["session_date"], "LONG"))
    again = tracker.evaluate("B-TEST_USDT", candle, INTERVAL, now + 11_000,
                             lambda: make_arrays(forming_open))
    assert not [r for r in again if r["state"] == "LIVE_ORB"]


def test_live_orb_poke_that_falls_back_resets_hold_timer():
    forming_open, now, ctx = asia_context()
    tracker = LiveTracker(LiveConfig(sessions=("asia",)))
    rh = ctx.orb["asia"]["range_high"]
    above = forming(forming_open, 100.0, rh + 0.4, 99.98, rh + 0.25 * ctx.atr, 400.0)
    inside = forming(forming_open, 100.0, rh + 0.4, 99.98, rh - 0.05, 400.0)
    t0 = now
    tracker.evaluate("B-TEST_USDT", above, INTERVAL, t0, lambda: make_arrays(forming_open))
    tracker.evaluate("B-TEST_USDT", inside, INTERVAL, t0 + 5_000, lambda: make_arrays(forming_open))
    # 9 s after the first poke it is above again, but the timer restarted at +6 s.
    res = tracker.evaluate("B-TEST_USDT", above, INTERVAL, t0 + 9_000, lambda: make_arrays(forming_open))
    assert not [r for r in res if r["state"] == "LIVE_ORB"]


def test_live_orb_skips_continuation_of_an_older_break():
    forming_open, now, ctx = asia_context()
    ctx.last_close = ctx.orb["asia"]["range_high"] + 0.5   # previous candle already closed beyond
    tracker = LiveTracker(LiveConfig(sessions=("asia",)))
    rh = ctx.orb["asia"]["range_high"]
    candle = forming(forming_open, rh + 0.5, rh + 0.8, rh + 0.45, rh + 0.7, 600.0)
    tracker._ctx["B-TEST_USDT"] = ctx
    for step in (0, 10_000):
        res = tracker.evaluate("B-TEST_USDT", candle, INTERVAL, now + step, lambda: None)
    assert not [r for r in res if r["state"] == "LIVE_ORB"]


def test_live_orb_rejects_extended_break():
    forming_open, now, ctx = asia_context()
    tracker = LiveTracker(LiveConfig(sessions=("asia",)))
    tracker._ctx["B-TEST_USDT"] = ctx
    rh = ctx.orb["asia"]["range_high"]
    c = rh + 1.2 * ctx.atr                                  # > 0.6 ATR beyond the edge
    candle = forming(forming_open, 100.0, c + 0.01, 99.98, c, 900.0)
    for step in (0, 10_000):
        res = tracker.evaluate("B-TEST_USDT", candle, INTERVAL, now + step, lambda: None)
    assert not [r for r in res if r["state"] == "LIVE_ORB"]


# ------------------------------------------------ closed-candle ORB changes --

def _asia_candles(extra_closes):
    """Three range candles then post-range candles with the given closes."""
    starts = [ASIA_OPEN_MS + i * INTERVAL for i in range(3 + len(extra_closes))]
    opens, highs, lows, closes = [100.0] * 3, [101.0, 101.2, 101.1], [99.8, 100.0, 99.9], [100.5, 100.8, 100.7]
    for close in extra_closes:
        opens.append(close - 0.30)         # body-dominant, low stays clear of the range edge
        highs.append(close + 0.03)
        lows.append(close - 0.32)
        closes.append(close)
    return {"time": np.asarray(starts, dtype=np.int64), "open": np.asarray(opens),
            "high": np.asarray(highs), "low": np.asarray(lows), "close": np.asarray(closes)}


def test_closed_orb_requires_indicator_warmup_history():
    # Four bars cannot initialize EMA/RSI/MACD and must fail closed.
    result = orb_detector.evaluate(_asia_candles([101.9]), 1.5, 1.8, session="asia")
    assert result["state"] == "INVALID_DATA"


def test_closed_orb_rejects_insufficient_indicator_history():
    stale = orb_detector.evaluate(_asia_candles([101.9, 102.0, 102.1, 102.2]), 1.5, 1.8, session="asia")
    fresh = orb_detector.evaluate(_asia_candles([101.9, 102.0]), 1.5, 1.8, session="asia")
    assert stale["state"] == "INVALID_DATA"
    assert fresh["state"] == "INVALID_DATA"


def test_closed_explosion_uses_directional_indicators_not_candle_geometry():
    from strategies.momentum import explosion_detector
    indicators = {
        "direction": "LONG", "rvol": 1.65, "atr_expansion": 1.16,
        "rsi": 58.0, "ema_aligned": True, "macd_direction_ok": True,
        "macd_accel": True, "squeeze_release": True, "bw_expand": True,
    }
    assert explosion_detector.evaluate(indicators) is True
    assert explosion_detector.evaluate({
        **indicators, "ema_aligned": False, "macd_direction_ok": False,
        "macd_accel": False, "roc_direction_ok": False,
    }) is False


def test_live_explosion_requires_indicator_ignition_evidence():
    t = 1_800_000_000_000 - (1_800_000_000_000 % INTERVAL)
    now = t + 90_000
    ctx = ctx_for(t, now)
    c = 100.0 + 1.5 * ctx.atr
    candle = forming(t, 100.0, c + 0.01, 99.98, c, 520.0)

    # A swing-structure break alone is not an ignition pathway.
    ctx.resistance = 100.1
    ctx.compression_recent = False
    ctx.compression_age = None
    res = evaluate_explosion(ctx, candle, now)
    assert res["reason"] == "no_ignition_evidence"

    # A recent squeeze still needs its indicator-based volume-pace threshold.
    ctx.resistance = 200.0
    ctx.compression_recent = True
    ctx.compression_age = 0
    res = evaluate_explosion(ctx, candle, now)
    assert res["reason"] == "no_ignition_evidence"


def test_live_explosion_reports_qualified_ignition_path():
    t = 1_800_000_000_000 - (1_800_000_000_000 % INTERVAL)
    now = t + 90_000
    ctx = ctx_for(t, now)
    c = 100.0 + 1.5 * ctx.atr
    ctx.compression_age = 0
    ctx.compression_recent = True
    res = evaluate_explosion(ctx, forming(t, 100.0, c + 0.01, 99.98, c, 600.0), now)
    assert res["state"] == "LIVE_EXPLOSION"
    assert res["squeeze_ignition"] is True


def test_live_orb_does_not_reject_wick_heavy_breakout_candle():
    forming_open, now, ctx = asia_context()
    tracker = LiveTracker(LiveConfig(sessions=("asia",)))
    rh = ctx.orb["asia"]["range_high"]
    c = rh + 0.25 * ctx.atr
    # Persistent close beyond range edge but a large opposing upper wick.
    candle = forming(forming_open, 100.0, c + 1.5 * ctx.atr, 99.98, c, 700.0)
    tracker.evaluate("B-TEST_USDT", candle, INTERVAL, now, lambda: make_arrays(forming_open))
    result = tracker.evaluate("B-TEST_USDT", candle, INTERVAL, now + 10_000,
                              lambda: make_arrays(forming_open))
    assert [r for r in result if r["state"] == "LIVE_ORB"]


def test_squeeze_ignition_requires_fresh_compression():
    t = 1_800_000_000_000 - (1_800_000_000_000 % INTERVAL)
    now = t + 90_000
    ctx = ctx_for(t, now)
    ctx.resistance = 200.0
    ctx.compression_recent = True
    ctx.compression_age = 0
    c = 100.0 + 1.5 * ctx.atr
    candle = forming(t, 100.0, c + 0.01, 99.98, c, 600.0)
    fresh = evaluate_explosion(ctx, candle, now)
    assert fresh["state"] == "LIVE_EXPLOSION"
    assert fresh["squeeze_ignition"] is True

    ctx.compression_age = 4
    stale = evaluate_explosion(ctx, candle, now)
    assert stale["reason"] == "no_ignition_evidence"


def test_tracker_ignores_delayed_candle_outside_current_bucket():
    t = 1_800_000_000_000 - (1_800_000_000_000 % INTERVAL)
    tracker = LiveTracker(LiveConfig(sessions=()))
    called = {"arrays": 0}

    def arrays():
        called["arrays"] += 1
        return make_arrays(t)

    # Once the bucket has ended, delayed/replayed WebSocket payload is ignored.
    result = tracker.evaluate(
        "B-TEST_USDT", forming(t, 100.0, 101.0, 99.0, 100.5, 600.0),
        INTERVAL, t + INTERVAL, arrays,
    )
    assert result == []
    assert called["arrays"] == 0


def test_tracker_repeated_same_candle_reuses_context_and_one_shot_guard():
    from strategies.momentum import live_ignition

    t = 1_800_000_000_000 - (1_800_000_000_000 % INTERVAL)
    tracker = LiveTracker(LiveConfig(sessions=()))
    candle = forming(t, 100.0, 101.0, 99.0, 100.5, 600.0)
    called = {"arrays": 0}

    def arrays():
        called["arrays"] += 1
        return make_arrays(t)

    # Keep this focused on bucket/context caching, not candidate score thresholds.
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(live_ignition, "evaluate_orb", lambda *args, **kwargs: [])
    monkeypatch.setattr(live_ignition, "evaluate_explosion", lambda *args, **kwargs: {"state": "REJECT"})
    try:
        tracker.evaluate("B-TEST_USDT", candle, INTERVAL, t + 10_000, arrays)
        tracker.evaluate("B-TEST_USDT", candle, INTERVAL, t + 11_000, arrays)
    finally:
        monkeypatch.undo()

    assert called["arrays"] == 1
    assert tracker.first_attempt("B-TEST_USDT", "LONG", t) is True
    assert tracker.first_attempt("B-TEST_USDT", "LONG", t) is False
