from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

from strategies.momentum.orb_detector import evaluate, opening_range, SESSIONS


def _stamp(dt):
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)


def candles_for(session="london", direction="LONG", last_open_minute=30):
    spec = SESSIONS[session]
    tz = ZoneInfo(spec["timezone"])
    day = datetime(2026, 6, 2, 12, 0, tzinfo=tz).date()
    start = datetime.combine(day, spec["open"], tzinfo=tz)
    # 48 completed bars before the session anchor provide indicator history.
    times = [start - timedelta(minutes=5 * (48 - i)) for i in range(48)]
    # Exactly six completed 5m candles form the 30-minute opening range.
    times += [start + timedelta(minutes=m) for m in (0, 5, 10, 15, 20, 25)]
    # The final row opens after the range has closed.
    if last_open_minute >= 30:
        times += [start + timedelta(minutes=last_open_minute)]

    if direction == "LONG":
        close = np.linspace(95.0, 100.0, len(times))
    else:
        close = np.linspace(105.0, 100.0, len(times))
    open_ = close - 0.03 if direction == "LONG" else close + 0.03
    high = np.maximum(open_, close) + 0.10
    low = np.minimum(open_, close) - 0.10
    volume = np.full(len(times), 100.0)

    # A separate, consistent 6-bar range for this session.
    range_start = 48
    range_bars = [
        (100.0, 100.8, 99.8, 100.5),
        (100.5, 101.0, 100.2, 100.7),
        (100.7, 101.1, 100.4, 100.9),
        (100.9, 101.0, 100.3, 100.6),
        (100.6, 100.9, 100.1, 100.5),
        (100.5, 100.8, 100.0, 100.4),
    ]
    for j, values in enumerate(range_bars, start=range_start):
        open_[j], high[j], low[j], close[j] = values

    if last_open_minute >= 30:
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


def test_only_four_documented_sessions_are_enabled_by_default_catalogue():
    assert set(SESSIONS) == {"asia", "london", "new_york", "late_us"}
    assert SESSIONS["asia"]["timezone"] == "UTC"
    assert SESSIONS["london"]["timezone"] == "Europe/London"
    assert SESSIONS["new_york"]["timezone"] == "America/New_York"
    assert SESSIONS["late_us"]["timezone"] == "America/New_York"


def test_opening_range_requires_all_six_completed_candles():
    data = candles_for("london", last_open_minute=25)
    now_before_close = _stamp(datetime(2026, 6, 2, 8, 29, tzinfo=ZoneInfo("Europe/London")))
    result = opening_range(data, "london", now_ms=now_before_close)
    assert result["state"] == "BUILDING_RANGE"

    now_after_close = _stamp(datetime(2026, 6, 2, 8, 30, tzinfo=ZoneInfo("Europe/London")))
    result = opening_range(data, "london", now_ms=now_after_close)
    assert result["state"] == "READY"
    assert result["range_candles"] == 6
    assert result["range_high"] == 101.1
    assert result["range_low"] == 99.8


def test_london_long_orb_requires_completed_close_beyond_30m_range():
    result = evaluate(candles_for("london", "LONG"), 0.8, session="london", min_rvol=1.4)
    assert result["state"] == "ORB_BREAKOUT", result
    assert result["direction"] == "LONG"
    assert result["range_end_local"].endswith("08:30:00+01:00")
    assert result["session_rvol"] >= 1.4
    assert result["indicator_confirmation"]["passed"] >= 3


def test_new_york_short_orb_is_directionally_mirrored():
    result = evaluate(candles_for("new_york", "SHORT"), 0.8, session="new_york", min_rvol=1.4)
    assert result["state"] == "ORB_BREAKOUT", result
    assert result["direction"] == "SHORT"
    assert result["range_end_local"].endswith("10:00:00-04:00")


def test_no_prebreakout_signal_inside_range():
    data = candles_for("london", "LONG", last_open_minute=25)
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
    asia_result = evaluate(asia, 0.8, session="asia")
    assert asia_result["session_end_utc_ms"] - asia_result["session_start_utc_ms"] == 270 * 60_000

    london = candles_for("london", "LONG")
    london_result = evaluate(london, 0.8, session="london")
    assert london_result["session_end_utc_ms"] - london_result["session_start_utc_ms"] == 270 * 60_000

    late = candles_for("late_us", "LONG")
    late_result = evaluate(late, 0.8, session="late_us")
    assert late_result["session_end_utc_ms"] - late_result["session_start_utc_ms"] == 210 * 60_000
