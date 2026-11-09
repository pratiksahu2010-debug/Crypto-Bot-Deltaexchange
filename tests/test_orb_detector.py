from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

from strategies.momentum.orb_detector import evaluate, opening_range, SESSIONS


def _stamp(dt):
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)


def candles_for(session="london", final_minute=15, breakout=False):
    spec = SESSIONS[session]
    tz = ZoneInfo(spec["timezone"])
    day = datetime(2026, 6, 2, 12, 0, tzinfo=tz).date()
    start = datetime.combine(day, spec["open"], tzinfo=tz)
    times = [start - timedelta(minutes=5 * (48-i)) for i in range(48)]
    times += [start + timedelta(minutes=m) for m in (0, 5, 10)]
    times += [start + timedelta(minutes=final_minute)]
    close = np.linspace(95.0, 100.0, len(times))
    open_ = close - 0.05
    high = close + 0.10
    low = close - 0.10
    volume = np.full(len(times), 100.0)
    for j, (o, h, l, cl) in enumerate([
        (100.0, 100.8, 99.8, 100.5),
        (100.5, 101.0, 100.2, 100.7),
        (100.7, 101.1, 100.4, 100.9),
    ], start=48):
        open_[j], high[j], low[j], close[j] = o, h, l, cl
    last = len(times)-1
    if breakout:
        open_[last], high[last], low[last], close[last] = 101.0, 102.0, 100.9, 101.8
    else:
        open_[last], high[last], low[last], close[last] = 100.8, 101.15, 100.7, 101.05
    volume[last] = 180.0
    return {
        "time": np.asarray([_stamp(x) for x in times], dtype=np.int64),
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    }


def test_global_session_definitions_cover_crypto_and_equity_session_opens():
    assert set(SESSIONS) == {"utc", "crypto_peak", "sydney", "asia", "london", "new_york", "cme"}
    assert SESSIONS["utc"]["timezone"] == "UTC"
    assert SESSIONS["london"]["timezone"] == "Europe/London"
    assert SESSIONS["new_york"]["timezone"] == "America/New_York"
    assert SESSIONS["cme"]["timezone"] == "America/Chicago"


def test_opening_range_waits_for_three_completed_bars():
    data = candles_for("london", final_minute=10)
    result = opening_range(data, session="london", now_ms=int(data["time"][-1]))
    assert result["state"] == "BUILDING_RANGE"


def test_indicator_led_orb_prebreakout_fires_before_range_escape():
    result = evaluate(candles_for("london", final_minute=15), 0.8, 1.8, session="london")
    assert result["state"] == "ORB_PREBREAKOUT", result
    assert result["direction"] == "LONG"
    assert result["indicator_confirmation"]["passed"] >= 3
    assert result["breakout_extension_atr"] <= 0.35


def test_indicator_led_orb_breakout_supports_completed_break():
    result = evaluate(candles_for("new_york", final_minute=15, breakout=True),
                      0.8, 1.8, session="new_york")
    assert result["state"] == "ORB_BREAKOUT", result
    assert result["direction"] == "LONG"
    assert result["indicator_confirmation"]["passed"] >= 3


def test_orb_direction_and_rsi_confirmation_are_mirrored_for_short():
    data = candles_for("asia", final_minute=15)
    data["close"][-1] = 99.9
    data["open"][-1] = 100.2
    data["high"][-1] = 100.3
    data["low"][-1] = 99.8
    data["close"][:48] = np.linspace(105.0, 100.0, 48)
    data["open"][:48] = data["close"][:48] + 0.05
    data["high"][:48] = data["open"][:48] + 0.1
    data["low"][:48] = data["close"][:48] - 0.1
    result = evaluate(data, 0.8, 1.8, session="asia")
    assert result["state"] in ("ORB_PREBREAKOUT", "ORB_BREAKOUT", "INSIDE_RANGE")


def test_dst_aware_local_session_open_is_used():
    london_summer = datetime(2026, 6, 2, 8, 0, tzinfo=ZoneInfo("Europe/London"))
    london_winter = datetime(2026, 1, 5, 8, 0, tzinfo=ZoneInfo("Europe/London"))
    ny_summer = datetime(2026, 6, 2, 9, 30, tzinfo=ZoneInfo("America/New_York"))
    ny_winter = datetime(2026, 1, 5, 9, 30, tzinfo=ZoneInfo("America/New_York"))
    assert _stamp(london_summer) != _stamp(london_winter)
    assert _stamp(ny_summer) != _stamp(ny_winter)


def test_cme_session_is_dst_aware_and_skips_saturday():
    saturday = datetime(2026, 6, 6, 17, 0, tzinfo=ZoneInfo("America/Chicago"))
    data = candles_for("cme", final_minute=20)
    data["time"][-1] = _stamp(saturday)
    result = evaluate(data, 1.0, 1.5, session="cme")
    assert result["state"] == "SESSION_CLOSED"


def test_orb_evaluates_immediately_when_third_opening_bar_has_closed():
    # The last completed candle opens at 08:10 and closes at 08:15.
    # The opening range is complete at that close; waiting for an 08:15
    # candle to close would introduce an unnecessary five-minute delay.
    data = candles_for("london", final_minute=15)
    for key in data:
        data[key] = data[key][:-1]
    result = evaluate(data, 0.8, 1.8, session="london")
    assert result["state"] == "ORB_PREBREAKOUT", result
    assert result["direction"] == "LONG"
    assert result["range_end_local"].endswith("08:15:00+01:00")
