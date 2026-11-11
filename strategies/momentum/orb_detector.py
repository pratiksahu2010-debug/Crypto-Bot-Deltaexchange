"""Session-based crypto Opening Range Breakout detector for Delta perps.

All qualification uses completed 5-minute candles. Each session has its own
5-minute opening range, session-anchored VWAP, session-local RVOL baseline,
and timezone-aware trade window. London/New York use IANA zones so DST is
handled automatically. ORB alerts are emitted only after a candle CLOSES
outside that session's opening range; no pre-breakout alerts are generated.
"""
from datetime import datetime, time, timedelta, timezone
from math import isfinite
from zoneinfo import ZoneInfo

import numpy as np

from indicators.ema import ema
from indicators.rsi import rsi
from indicators.macd import macd
from indicators.vwap import vwap
from indicators.roc import roc


# Trade window starts after the first completed 5-minute opening candle.
# Asia: 00:00-00:05 UTC range, then trade for four hours.
# London/NY: 5-minute range followed by four hours of trading.
# Late US: optional 5-minute range followed by three hours of trading.
SESSIONS = {
    "asia": {
        "timezone": "UTC", "open": time(0, 0),
        "range_minutes": 5, "trade_minutes": 240,
    },
    "sydney": {
        "timezone": "Australia/Sydney", "open": time(10, 0),
        "range_minutes": 5, "trade_minutes": 240,
    },
    "cme": {
        "timezone": "America/Chicago", "open": time(17, 0),
        "range_minutes": 5, "trade_minutes": 240,
    },
    "london": {
        "timezone": "Europe/London", "open": time(8, 0),
        "range_minutes": 5, "trade_minutes": 240,
    },
    "new_york": {
        "timezone": "America/New_York", "open": time(9, 30),
        "range_minutes": 5, "trade_minutes": 240,
    },
    "late_us": {
        "timezone": "America/New_York", "open": time(16, 0),
        "range_minutes": 5, "trade_minutes": 180,
    },
}


def _local(ts, tz):
    return datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc).astimezone(tz)


def _finite(value):
    try:
        result = float(value)
        return result if isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _result(state, meta, **values):
    return {"state": state, **values, **meta}


def _session_window(now, spec, active_minutes=None):
    """Return local anchor, range end and trade-window expiry."""
    tz = ZoneInfo(spec["timezone"])
    start = datetime.combine(now.date(), spec["open"], tzinfo=tz)
    range_end = start + timedelta(minutes=int(spec["range_minutes"]))
    # active_minutes is accepted for backward compatibility but session
    # durations in the specification are authoritative.
    trade_minutes = int(spec["trade_minutes"])
    expiry = range_end + timedelta(minutes=trade_minutes)
    return start, range_end, expiry


def _session_meta(name, spec, start, range_end, expiry):
    return {
        "session": name,
        "session_timezone": spec["timezone"],
        "session_date": start.date().isoformat(),
        "session_open_local": start.isoformat(),
        "range_start_local": start.isoformat(),
        "range_end_local": range_end.isoformat(),
        "trade_window_end_local": expiry.isoformat(),
        "session_start_utc_ms": int(start.astimezone(timezone.utc).timestamp() * 1000),
        "range_end_utc_ms": int(range_end.astimezone(timezone.utc).timestamp() * 1000),
        "session_end_utc_ms": int(expiry.astimezone(timezone.utc).timestamp() * 1000),
    }


def _validate_candles(candles, require_volume=False):
    stamps = np.asarray(candles["time"], dtype=np.int64)
    arrays = {
        key: np.asarray(candles[key], dtype=np.float64)
        for key in ("open", "high", "low", "close")
    }
    volumes = (
        np.asarray(candles["volume"], dtype=np.float64)
        if "volume" in candles else None
    )
    n = len(stamps)
    if n == 0 or stamps.ndim != 1 or any(
        values.ndim != 1 or len(values) != n for values in arrays.values()
    ):
        raise ValueError("candle arrays have inconsistent lengths")
    if np.any(np.diff(stamps) <= 0):
        raise ValueError("candle timestamps must be strictly increasing")
    if any(not np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("non-finite OHLC")
    if any(np.any(arrays[key] <= 0) for key in ("open", "high", "low", "close")):
        raise ValueError("non-positive OHLC")
    if (
        np.any(arrays["high"] < arrays["low"])
        or np.any(arrays["close"] < arrays["low"])
        or np.any(arrays["close"] > arrays["high"])
    ):
        raise ValueError("invalid OHLC relationship")
    if volumes is not None and (
        volumes.ndim != 1 or len(volumes) != n
        or not np.isfinite(volumes).all() or np.any(volumes < 0)
    ):
        raise ValueError("invalid volume")
    if require_volume and volumes is None:
        raise ValueError("volume is required")
    return stamps, arrays, volumes


def _range_indexes(stamps, start, range_minutes):
    """Find the first completed 5-minute candle that defines the session range."""
    start_ms = int(start.astimezone(timezone.utc).timestamp() * 1000)
    expected = [start_ms + offset * 60_000 for offset in range(0, range_minutes, 5)]
    positions = {int(ts): i for i, ts in enumerate(stamps)}
    if not all(ts in positions for ts in expected):
        return None
    indexes = [positions[ts] for ts in expected]
    if any(stamps[indexes[i + 1]] - stamps[indexes[i]] != 300_000
           for i in range(len(indexes) - 1)):
        return None
    return indexes


def opening_range(candles, session="london", now_ms=None, active_minutes=None):
    """Return the one-bar completed 5-minute opening range for one session."""
    try:
        name = str(session).lower()
        spec = SESSIONS[name]
        if now_ms is None:
            return {"state": "INVALID_DATA", "session": name}
        tz = ZoneInfo(spec["timezone"])
        now = _local(now_ms, tz)
        start, range_end, expiry = _session_window(now, spec, active_minutes)
        meta = _session_meta(name, spec, start, range_end, expiry)
        if now < range_end:
            return _result("BUILDING_RANGE", meta)
        if now >= expiry:
            return _result("SESSION_CLOSED", meta)
        stamps, arrays, _ = _validate_candles(candles)
        indexes = _range_indexes(stamps, start, spec["range_minutes"])
        if indexes is None:
            return _result("INCOMPLETE_RANGE", meta)
        top = float(np.max(arrays["high"][indexes]))
        bottom = float(np.min(arrays["low"][indexes]))
        if not isfinite(top) or not isfinite(bottom) or top <= bottom:
            return _result("INVALID_RANGE", meta)
        return _result(
            "READY", meta, range_high=top, range_low=bottom,
            range_candles=len(indexes),
        )
    except (KeyError, TypeError, ValueError, OverflowError, IndexError):
        return {"state": "INVALID_DATA", "session": str(session).lower()}


def evaluate(
    candles, atr_value, rvol=None, session="london", min_rvol=1.10,
    max_extension_atr=1.50, active_minutes=None, adx_value=None,
    plus_di=None, minus_di=None, adx_rising=False, adx_min=20.0,
    max_breakout_age_candles=2, max_pullback_extension_atr=1.50,
):
    """Evaluate one independent session ORB using completed 5-minute candles."""
    try:
        name = str(session).lower()
        spec = SESSIONS[name]
        stamps, arrays, volumes = _validate_candles(candles, require_volume=True)
        n = len(stamps)
        if n < 35:
            return {"state": "INVALID_DATA", "session": name}

        # Delta candle timestamps are candle OPEN times. The engine supplies
        # REST-verified completed candles, so the final candle closes +5m.
        now_ms = int(stamps[-1]) + 300_000
        tz = ZoneInfo(spec["timezone"])
        now = _local(now_ms, tz)
        start, range_end, expiry = _session_window(now, spec, active_minutes)
        meta = _session_meta(name, spec, start, range_end, expiry)
        if now < range_end:
            return _result("BUILDING_RANGE", meta)
        if now >= expiry:
            return _result("SESSION_CLOSED", meta)

        indexes = _range_indexes(stamps, start, spec["range_minutes"])
        if indexes is None:
            return _result("INCOMPLETE_RANGE", meta)
        range_high = float(np.max(arrays["high"][indexes]))
        range_low = float(np.min(arrays["low"][indexes]))
        atr_v = _finite(atr_value)
        if atr_v is None or atr_v <= 0 or range_high <= range_low:
            return _result("INVALID_VOLATILITY", meta)

        width_atr = (range_high - range_low) / atr_v
        if not 0.05 <= width_atr <= 3.0:
            return _result(
                "UNTRADEABLE_RANGE", meta, range_high=range_high,
                range_low=range_low, range_width_atr=round(width_atr, 4),
            )

        close = arrays["close"]
        high = arrays["high"]
        low = arrays["low"]
        volume_values = volumes
        session_start_ms = int(start.astimezone(timezone.utc).timestamp() * 1000)
        session_idx = int(np.searchsorted(stamps, session_start_ms, side="left"))
        # If the only available session candle is the opening-range candle,
        # there is no post-range bar to evaluate yet. This is simply inside the
        # range, not a missing-volume error or a pre-breakout signal.
        if session_idx >= n - 1:
            return _result(
                "INSIDE_RANGE", meta, range_high=range_high,
                range_low=range_low, range_width_atr=round(width_atr, 4),
            )
        # Each session gets its own volume baseline. Never include pre-session
        # bars: the opening candle is the initial baseline, then the baseline
        # expands to the mean of all completed bars since this session opened.
        prior_session_volumes = volume_values[session_idx:-1]
        if len(prior_session_volumes) < 1:
            return _result(
                "SESSION_VOLUME_BASELINE_BUILDING", meta,
                range_high=range_high, range_low=range_low,
            )
        baseline = float(np.mean(prior_session_volumes))
        current_volume = float(volume_values[-1])
        session_rvol = current_volume / baseline if baseline > 0 else None
        if session_rvol is None or not isfinite(session_rvol):
            return _result(
                "INVALID_SESSION_RVOL", meta,
                range_high=range_high, range_low=range_low,
            )
        e9, e21 = ema(close, 9), ema(close, 21)
        rsi_value = rsi(close, 14)
        macd_now = macd(close)
        macd_prev = macd(close[:-1]) if n > 35 else None
        hist = macd_now[2] if macd_now else None
        hist_prev = macd_prev[2] if macd_prev else None
        session_vwap = vwap(
            high[session_idx:], low[session_idx:], close[session_idx:],
            volume_values[session_idx:], period=None,
        ) if session_idx < n else None
        roc_value = roc(close, 6)
        current_close = float(close[-1])

        direction = (
            "LONG" if current_close > range_high
            else "SHORT" if current_close < range_low
            else None
        )
        if direction is None:
            return _result(
                "INSIDE_RANGE", meta, range_high=range_high,
                range_low=range_low, range_width_atr=round(width_atr, 4),
                session_vwap=session_vwap, session_rvol=round(session_rvol, 4),
                rvol_baseline=baseline,
            )

        is_long = direction == "LONG"
        adx_v = _finite(adx_value)
        plus_di_v = _finite(plus_di)
        minus_di_v = _finite(minus_di)
        adx_floor = _finite(adx_min)
        adx_floor = 20.0 if adx_floor is None else adx_floor
        di_aligned = bool(
            plus_di_v is not None and minus_di_v is not None
            and (plus_di_v > minus_di_v if is_long else minus_di_v > plus_di_v)
        )
        # Stage 1: a closed candle must clear the range edge with directional
        # body. Stage 2 uses ADX/DI as quality evidence; rising ADX can qualify
        # in the lower band, while marginal DI disagreement is scored rather
        # than automatically vetoing an otherwise clean breakout.
        adx_ok = bool(
            adx_v is not None and (
                adx_v >= adx_floor
                or (bool(adx_rising) and adx_v >= max(14.0, adx_floor - 6.0) and di_aligned)
            )
        )
        if not adx_ok:
            return _result(
                "WEAK_TREND", meta, direction=direction,
                range_high=range_high, range_low=range_low,
                adx=adx_v, adx_min=adx_floor, plus_di=plus_di_v,
                minus_di=minus_di_v, di_aligned=di_aligned,
                adx_rising=bool(adx_rising),
                session_rvol=round(session_rvol, 4), rvol_baseline=round(baseline, 8),
            )

        macd_min_abs = max(atr_v * 0.01, 1e-12)
        flags = {
            "ema_9_21": bool(e9 is not None and e21 is not None and (e9 > e21 if is_long else e9 < e21)),
            "session_vwap": bool(session_vwap is not None and (current_close > session_vwap if is_long else current_close < session_vwap)),
            "rsi": bool(rsi_value is not None and (rsi_value >= 53 if is_long else rsi_value <= 47)),
            "macd_histogram": bool(hist is not None and abs(float(hist)) >= macd_min_abs and (hist > 0 if is_long else hist < 0)),
            "macd_acceleration": bool(hist is not None and hist_prev is not None and (hist > hist_prev if is_long else hist < hist_prev)),
            "roc": bool(roc_value is not None and (roc_value > 0 if is_long else roc_value < 0)),
            "di_alignment": di_aligned,
            "adx_rising": bool(adx_rising),
        }
        passed = sum(flags.values())
        trend_votes = int(flags["ema_9_21"]) + int(flags["session_vwap"])
        momentum_votes = int(flags["macd_histogram"]) + int(flags["roc"])
        # Require some independent trend and momentum evidence, but do not
        # demand RSI, MACD acceleration and DI all agree on the same candle.
        if passed < 3 or trend_votes < 1 or momentum_votes < 1:
            return _result(
                "INDICATOR_UNCONFIRMED", meta, direction=direction,
                range_high=range_high, range_low=range_low, adx=adx_v,
                plus_di=plus_di_v, minus_di=minus_di_v, di_aligned=di_aligned,
                indicator_confirmation={
                    "checks": flags, "passed": passed,
                    "minimum_passed": 3, "trend_votes": trend_votes,
                    "momentum_votes": momentum_votes,
                },
                session_rvol=round(session_rvol, 4), rvol_baseline=round(baseline, 8),
            )

        # Use one participation decision, not two independent hard RVOL gates.
        # Either the session baseline or the rolling baseline may confirm it.
        rolling_rvol = _finite(rvol)
        try:
            requested_rvol = float(min_rvol)
        except (TypeError, ValueError, OverflowError):
            requested_rvol = 1.10
        if not isfinite(requested_rvol) or requested_rvol <= 0:
            requested_rvol = 1.10
        participation_floor = max(1.0, requested_rvol * 0.70)
        session_volume_ok = session_rvol >= participation_floor
        rolling_volume_ok = rolling_rvol is not None and rolling_rvol >= participation_floor
        if not (session_volume_ok or rolling_volume_ok):
            return _result(
                "LOW_PARTICIPATION", meta, direction=direction,
                range_high=range_high, range_low=range_low,
                session_rvol=round(session_rvol, 4),
                rolling_rvol=rolling_rvol,
                participation_min=round(participation_floor, 4),
                rvol_baseline=round(baseline, 8),
            )

        # Stage 1 breakout quality: close beyond a small ATR buffer and body
        # in the breakout direction. Retests may close nearer to the edge.
        edge = range_high if is_long else range_low
        previous_beyond = (
            close[-2] > edge if is_long else close[-2] < edge
        ) if n >= 2 else False
        retest_tolerance = atr_v * 0.15
        retest_touch = bool(
            previous_beyond
            and (
                (low[-1] <= edge + retest_tolerance and current_close > edge and arrays["close"][-1] > arrays["open"][-1])
                if is_long else
                (high[-1] >= edge - retest_tolerance and current_close < edge and arrays["close"][-1] < arrays["open"][-1])
            )
        )
        breakout_buffer = atr_v * 0.03
        body_aligned = arrays["close"][-1] > arrays["open"][-1] if is_long else arrays["close"][-1] < arrays["open"][-1]
        beyond_buffer = (
            current_close >= edge + breakout_buffer if is_long
            else current_close <= edge - breakout_buffer
        )
        if not body_aligned or (not beyond_buffer and not retest_touch):
            return _result(
                "WEAK_BREAKOUT_CLOSE", meta, direction=direction,
                range_high=range_high, range_low=range_low,
                breakout_buffer=breakout_buffer, body_aligned=body_aligned,
                retest_touch=retest_touch, adx=adx_v, plus_di=plus_di_v,
                minus_di=minus_di_v, di_aligned=di_aligned,
                session_rvol=round(session_rvol, 4), rolling_rvol=rolling_rvol,
                indicator_confirmation={"checks": flags, "passed": passed},
            )

        extension = abs(current_close - edge) / atr_v
        breakout_age = 0
        for i in range(n - 1, -1, -1):
            candle_close_ms = int(stamps[i]) + 300_000
            if candle_close_ms <= int(meta["range_end_utc_ms"]):
                break
            beyond = close[i] > range_high if is_long else close[i] < range_low
            if not beyond:
                break
            breakout_age += 1

        # Stage 3: immediate entries are limited to max_extension_atr.
        # A valid but later breakout (up to max_pullback_extension_atr) is
        # retained as a pullback candidate; the engine must wait for price to
        # return to its planned entry zone before alerting. This catches strong
        # moves without chasing an already-extended close.
        try:
            immediate_extension_limit = float(max_extension_atr)
        except (TypeError, ValueError, OverflowError):
            immediate_extension_limit = 0.80
        if not isfinite(immediate_extension_limit) or immediate_extension_limit <= 0:
            immediate_extension_limit = 0.80
        try:
            pullback_extension_limit = float(max_pullback_extension_atr)
        except (TypeError, ValueError, OverflowError):
            pullback_extension_limit = 1.50
        if not isfinite(pullback_extension_limit):
            pullback_extension_limit = 1.50
        pullback_extension_limit = max(immediate_extension_limit, pullback_extension_limit)

        if breakout_age > int(max_breakout_age_candles):
            state = "STALE_BREAKOUT"
        elif extension > pullback_extension_limit + 1e-9:
            state = "EXTENDED_BREAKOUT"
        elif retest_touch:
            state = "ORB_RETEST"
        elif extension > immediate_extension_limit + 1e-9:
            state = "ORB_PULLBACK_PENDING"
        else:
            state = "ORB_BREAKOUT"

        return _result(
            state, meta, direction=direction,
            range_high=range_high, range_low=range_low,
            breakout_level=edge,
            breakout_extension_atr=round(extension, 4),
            immediate_extension_limit_atr=round(immediate_extension_limit, 4),
            pullback_extension_limit_atr=round(pullback_extension_limit, 4),
            pullback_required=state == "ORB_PULLBACK_PENDING",
            range_width_atr=round(width_atr, 4),
            breakout_age_candles=breakout_age,
            range_candles=len(indexes),
            session_rvol=round(session_rvol, 4),
            rolling_rvol=round(rolling_rvol, 4) if rolling_rvol is not None else None,
            rolling_rvol_min=round(participation_floor, 4),
            participation_floor=round(participation_floor, 4),
            participation_confirmed=True,
            rvol_baseline=round(baseline, 8),
            session_vwap=session_vwap,
            indicator_confirmation={
                "checks": flags, "passed": passed,
                "minimum_passed": 3, "trend_votes": trend_votes,
                "momentum_votes": momentum_votes,
                "ema9": e9, "ema21": e21, "vwap": session_vwap,
                "rsi": rsi_value, "macd_histogram": hist, "roc": roc_value,
            },
            adx=adx_v, adx_min=adx_floor,
            plus_di=plus_di_v, minus_di=minus_di_v,
            trend_confirmed=bool(adx_ok), di_aligned=di_aligned,
            adx_rising=bool(adx_rising), wick_quality=bool(body_aligned),
            retest_touch=bool(retest_touch), failed_breakout=False,
        )
    except (KeyError, TypeError, ValueError, OverflowError, IndexError, FloatingPointError):
        return {"state": "INVALID_DATA", "session": str(session).lower()}
