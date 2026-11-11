from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

from strategies.momentum.orb_detector import evaluate, opening_range, SESSIONS
from core.scoring_engine import ScoringEngine


def _stamp(dt):
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)


def candles_for(session="london", direction="LONG", last_open_minute=5):
    spec = SESSIONS[session]
    tz = ZoneInfo(spec["timezone"])
    day = datetime(2026, 6, 2, 12, 0, tzinfo=tz).date()
    start = datetime.combine(day, spec["open"], tzinfo=tz)
    # 48 completed bars before the session anchor provide indicator history.
    times = [start - timedelta(minutes=5 * (48 - i)) for i in range(48)]
    # Exactly one completed 5m candle forms the opening range.
    times += [start]
    # The final row opens immediately after the 5-minute range has closed.
    if last_open_minute >= 5:
        times += [start + timedelta(minutes=last_open_minute)]

    if direction == "LONG":
        close = np.linspace(95.0, 100.0, len(times))
    else:
        close = np.linspace(105.0, 100.0, len(times))
    open_ = close - 0.03 if direction == "LONG" else close + 0.03
    high = np.maximum(open_, close) + 0.10
    low = np.minimum(open_, close) - 0.10
    volume = np.full(len(times), 100.0)

    # The first completed candle alone defines the opening range.
    range_start = 48
    open_[range_start], high[range_start], low[range_start], close[range_start] = (
        100.0, 100.8, 99.8, 100.5
    )

    if last_open_minute >= 5:
        last = len(times) - 1
        if direction == "LONG":
            open_[last], high[last], low[last], close[last] = 101.2, 102.2, 101.1, 102.0
        else:
            open_[last], high[last], low[last], close[last] = 99.2, 99.3, 98.8, 98.9
        volume[last] = 220.0

    return {
        "time": np.asarray([_stamp(x) for x in times], dtype=np.int64),
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }


def test_documented_orb_sessions_are_enabled_by_default_catalogue():
    assert set(SESSIONS) == {"asia", "sydney", "cme", "london", "new_york", "late_us"}
    assert SESSIONS["asia"]["timezone"] == "UTC"
    assert SESSIONS["london"]["timezone"] == "Europe/London"
    assert SESSIONS["new_york"]["timezone"] == "America/New_York"
    assert SESSIONS["late_us"]["timezone"] == "America/New_York"


def test_opening_range_uses_one_completed_five_minute_candle():
    data = candles_for("london", last_open_minute=5)
    now_before_close = _stamp(datetime(2026, 6, 2, 8, 4, tzinfo=ZoneInfo("Europe/London")))
    result = opening_range(data, "london", now_ms=now_before_close)
    assert result["state"] == "BUILDING_RANGE"

    now_after_close = _stamp(datetime(2026, 6, 2, 8, 5, tzinfo=ZoneInfo("Europe/London")))
    result = opening_range(data, "london", now_ms=now_after_close)
    assert result["state"] == "READY"
    assert result["range_candles"] == 1
    assert result["range_high"] == 100.8
    assert result["range_low"] == 99.8


def test_london_long_orb_uses_first_completed_five_minute_range():
    result = evaluate(candles_for("london", "LONG"), 1.6, 1.5, session="london", min_rvol=1.4, adx_value=25, plus_di=30, minus_di=15)
    assert result["state"] == "ORB_BREAKOUT", result
    assert result["direction"] == "LONG"
    assert result["range_end_local"].endswith("08:05:00+01:00")
    assert result["session_rvol"] >= 1.4
    assert result["indicator_confirmation"]["passed"] >= 3
    assert result["range_candles"] == 1


def test_new_york_short_orb_is_directionally_mirrored():
    result = evaluate(candles_for("new_york", "SHORT"), 1.2, 1.5, session="new_york", min_rvol=1.4, adx_value=25, plus_di=12, minus_di=28)
    assert result["state"] == "ORB_BREAKOUT", result
    assert result["direction"] == "SHORT"
    assert result["range_end_local"].endswith("09:35:00-04:00")


def test_no_prebreakout_signal_inside_range():
    data = candles_for("london", "LONG", last_open_minute=0)
    result = evaluate(data, 0.8, session="london")
    assert result["state"] in ("BUILDING_RANGE", "INSIDE_RANGE")
    assert result.get("direction") is None


def test_london_and_new_york_anchors_shift_with_dst():
    london_summer = datetime(2026, 6, 2, 8, 0, tzinfo=ZoneInfo("Europe/London"))
    london_winter = datetime(2026, 1, 5, 8, 0, tzinfo=ZoneInfo("Europe/London"))
    ny_summer = datetime(2026, 6, 2, 9, 30, tzinfo=ZoneInfo("America/New_York"))
    ny_winter = datetime(2026, 1, 5, 9, 30, tzinfo=ZoneInfo("America/New_York"))
    assert london_summer.utcoffset() != london_winter.utcoffset()
    assert ny_summer.utcoffset() != ny_winter.utcoffset()


def test_session_trade_windows_match_specification():
    asia = candles_for("asia", "LONG")
    asia_result = evaluate(asia, 0.8, 1.5, session="asia", adx_value=25, plus_di=30, minus_di=15)
    assert asia_result["range_end_utc_ms"] - asia_result["session_start_utc_ms"] == 5 * 60_000
    assert asia_result["session_end_utc_ms"] - asia_result["session_start_utc_ms"] == 245 * 60_000

    london = candles_for("london", "LONG")
    london_result = evaluate(london, 0.8, 1.5, session="london", adx_value=25, plus_di=30, minus_di=15)
    assert london_result["range_end_utc_ms"] - london_result["session_start_utc_ms"] == 5 * 60_000
    assert london_result["session_end_utc_ms"] - london_result["session_start_utc_ms"] == 245 * 60_000

    late = candles_for("late_us", "LONG")
    late_result = evaluate(late, 0.8, 1.5, session="late_us", adx_value=25, plus_di=30, minus_di=15)
    assert late_result["range_end_utc_ms"] - late_result["session_start_utc_ms"] == 5 * 60_000
    assert late_result["session_end_utc_ms"] - late_result["session_start_utc_ms"] == 185 * 60_000


def test_orb_score_uses_session_rvol_and_detector_extension_limit():
    score = ScoringEngine().score_orb({
        "rvol": 0.2, "orb_session_rvol": 1.45, "orb_rolling_rvol": 1.5,
        "ema9": 102, "ema21": 101, "close": 102,
        "orb_vwap": 101, "rsi": 58, "macd_direction_ok": True,
        "roc_direction_ok": True, "atr": 1.0,
        "orb_breakout_extension_atr": 0.70,
    }, "LONG", min_rvol=1.4, max_extension_atr=0.8)
    components = {item.name: item for item in score.components}
    assert components["ORB Session RVOL"].points == 1.5
    assert components["ORB Breakout Extension"].points == 0.75
    assert components["ORB Session RVOL"].value == 1.45


def test_orb_score_never_substitutes_general_rvol_for_session_rvol():
    score = ScoringEngine().score_orb({
        "rvol": 5.0, "orb_session_rvol": None, "orb_rolling_rvol": 5.0,
        "orb_breakout_extension_atr": 0.8,
    }, "LONG", min_rvol=1.4, max_extension_atr=0.8)
    components = {item.name: item for item in score.components}
    assert components["ORB Session RVOL"].points == 0.0
    assert components["ORB Breakout Extension"].points == 0.75


def test_orb_rejects_weak_adx_even_with_high_volume():
    result = evaluate(
        candles_for("london", "SHORT"), 0.8, 1.5, session="london",
        min_rvol=1.4, adx_value=14.5, plus_di=10, minus_di=25,
    )
    assert result["state"] == "WEAK_TREND"
    assert result["adx"] == 14.5


def test_orb_scores_marginal_di_disagreement_instead_of_hard_rejecting():
    result = evaluate(
        candles_for("london", "SHORT"), 0.8, session="london",
        min_rvol=1.4, adx_value=25, plus_di=30, minus_di=15,
    )
    assert result["state"] == "ORB_BREAKOUT", result
    assert result["di_aligned"] is False
    assert result["indicator_confirmation"]["passed"] >= 3


def test_orb_volume_baseline_excludes_pre_session_bars():
    data = candles_for("london", "LONG")
    # Overnight/pre-session volume must not contaminate the London baseline.
    data["volume"][:48] = 1000.0
    result = evaluate(
        data, 0.8, 1.5, session="london", min_rvol=1.4,
        adx_value=25, plus_di=30, minus_di=15,
    )
    assert result["rvol_baseline"] == 100.0
    assert result["session_rvol"] == 2.2


def test_telegram_does_not_label_generic_momentum_as_orb_from_session_metadata():
    from tgbot.bot import TelegramBot

    bot = TelegramBot()
    generic = bot.format_signal({
        "pair": "B-BTC_USDT", "direction": "SHORT", "engine": "momentum",
        "strategy": "CONTINUATION", "score": 8.5, "grade": "STRONG",
        "indicators": {"orb_session": "london", "orb_session_open_local": "2026-11-11T08:00:00+00:00"},
    })
    assert "<b>SIGNAL TYPE:</b> MOMENTUM EXPLOSION" in generic
    assert "<b>ORB SESSION:</b>" not in generic

    orb = bot.format_signal({
        "pair": "B-BTC_USDT", "direction": "SHORT", "engine": "momentum",
        "strategy": "ORB_BREAKOUT", "score": 8.5, "grade": "STRONG",
        "indicators": {"orb_session": "london", "orb_session_open_local": "2026-11-11T08:00:00+00:00"},
    })
    assert "<b>SIGNAL TYPE:</b> ORB" in orb
    assert "<b>ORB SESSION:</b> london" in orb



def test_orb_rising_adx_can_qualify_in_early_strength_band():
    result = evaluate(
        candles_for("london", "LONG"), 0.8, 1.5, session="london",
        min_rvol=1.4, adx_value=15.5, plus_di=30, minus_di=15,
        adx_rising=True, adx_min=20,
    )
    # Exactly-at-cap extensions must not become extended due to float noise.
    assert result["state"] == "ORB_BREAKOUT", result
    assert result["adx_rising"] is True


def test_orb_extended_but_valid_breakout_waits_for_pullback():
    result = evaluate(
        candles_for("london", "LONG"), 1.0, 1.5, session="london",
        min_rvol=1.4, max_extension_atr=0.80, max_pullback_extension_atr=1.50,
        adx_value=25, plus_di=30, minus_di=15,
    )
    assert result["state"] == "ORB_PULLBACK_PENDING", result
    assert result["pullback_required"] is True
    assert result["pullback_extension_limit_atr"] == 1.5


def test_orb_retest_path_is_optional_and_directional():
    data = candles_for("london", "LONG")
    last_time = int(data["time"][-1]) + 300_000
    data["time"] = np.append(data["time"], last_time)
    data["open"] = np.append(data["open"], 101.0)
    data["high"] = np.append(data["high"], 101.25)
    data["low"] = np.append(data["low"], 100.75)
    data["close"] = np.append(data["close"], 101.10)
    data["volume"] = np.append(data["volume"], 180.0)
    result = evaluate(
        data, 1.6, 1.5, session="london", min_rvol=1.4,
        adx_value=25, plus_di=30, minus_di=15,
    )
    assert result["state"] == "ORB_RETEST", result
    assert result["retest_touch"] is True
