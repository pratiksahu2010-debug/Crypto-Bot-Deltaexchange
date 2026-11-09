"""Global crypto Opening Range Breakout detector.

Opening ranges are anchored to local market clocks using IANA time zones:
UTC daily reset, Sydney, Tokyo, London, New York and CME Globex. DST is
handled by zoneinfo. The first three completed 5-minute bars define a
15-minute range. The detector can alert on an indicator-led pre-breakout
approach to the range edge, not only after price has already escaped.

Signals use EMA(9/21), session VWAP, RSI(14), MACD(12/26/9), ROC and RVOL.
ATR(14) is used to normalize volatility and the distance to the range edge.
Candle colour, body ratio, wick ratio, order-book fields and risk data are
not used to determine ORB direction.
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

SESSIONS = {
    "utc": {"timezone": "UTC", "open": time(0, 0)},
    "crypto_peak": {"timezone": "UTC", "open": time(16, 0)},
    "sydney": {"timezone": "Australia/Sydney", "open": time(10, 0)},
    "asia": {"timezone": "Asia/Tokyo", "open": time(9, 0)},
    "london": {"timezone": "Europe/London", "open": time(8, 0)},
    "new_york": {"timezone": "America/New_York", "open": time(9, 30)},
    "cme": {"timezone": "America/Chicago", "open": time(17, 0)},
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


def _cme_closed(now):
    return (
        now.weekday() == 5
        or (now.weekday() == 4 and now.time() >= time(17, 0))
        or (now.weekday() == 6 and now.time() < time(17, 0))
    )


def _session_window(now, spec, active_minutes):
    tz = ZoneInfo(spec["timezone"])
    start = datetime.combine(now.date(), spec["open"], tzinfo=tz)
    end = start + timedelta(minutes=15)
    expiry = end + timedelta(minutes=max(15, int(active_minutes)))
    return start, end, expiry


def opening_range(candles, session="london", now_ms=None, active_minutes=180):
    """Return the exact three-bar, completed 15-minute opening range."""
    try:
        name = str(session).lower()
        spec = SESSIONS[name]
        tz = ZoneInfo(spec["timezone"])
        if now_ms is None:
            return {"state": "INVALID_DATA", "session": name}
        now = _local(now_ms, tz)
        if name == "cme" and _cme_closed(now):
            return {"state": "SESSION_CLOSED", "session": name}
        start, end, expiry = _session_window(now, spec, active_minutes)
        meta = {
            "session": name, "session_timezone": spec["timezone"],
            "session_date": now.date().isoformat(),
            "session_open_local": start.isoformat(),
            "range_start_local": start.isoformat(),
            "range_end_local": end.isoformat(),
            "session_start_utc_ms": int(start.astimezone(timezone.utc).timestamp() * 1000),
            "session_end_utc_ms": int(expiry.astimezone(timezone.utc).timestamp() * 1000),
        }
        if now < end:
            return _result("BUILDING_RANGE", meta)
        if now >= expiry:
            return _result("SESSION_CLOSED", meta)
        stamps = np.asarray(candles["time"], dtype=np.int64)
        highs = np.asarray(candles["high"], dtype=np.float64)
        lows = np.asarray(candles["low"], dtype=np.float64)
        if (stamps.ndim != 1 or highs.ndim != 1 or lows.ndim != 1 or len(stamps) == 0
                or not (len(stamps) == len(highs) == len(lows))
                or np.any(np.diff(stamps) <= 0) or not np.isfinite(highs).all()
                or not np.isfinite(lows).all() or np.any(highs <= 0)
                or np.any(lows <= 0) or np.any(highs < lows)):
            return _result("INVALID_DATA", meta)
        expected = [
            int((start + timedelta(minutes=m)).astimezone(timezone.utc).timestamp() * 1000)
            for m in (0, 5, 10)
        ]
        index = {int(ts): i for i, ts in enumerate(stamps)}
        if not all(ts in index for ts in expected):
            return _result("INCOMPLETE_RANGE", meta)
        positions = [index[ts] for ts in expected]
        if any(stamps[positions[i+1]] - stamps[positions[i]] != 300_000 for i in range(2)):
            return _result("INCOMPLETE_RANGE", meta)
        top = float(np.max(highs[positions]))
        bottom = float(np.min(lows[positions]))
        if not isfinite(top) or not isfinite(bottom) or top <= bottom:
            return _result("INVALID_RANGE", meta)
        return _result("READY", meta, range_high=top, range_low=bottom)
    except (KeyError, TypeError, ValueError, OverflowError, IndexError):
        return {"state": "INVALID_DATA", "session": str(session).lower()}


def evaluate(
    candles, atr_value, rvol, session="london", min_rvol=1.10,
    max_extension_atr=1.50, active_minutes=180, adx_value=None,
    plus_di=None, minus_di=None, adx_rising=False, adx_min=18.0,
    max_breakout_age_candles=2,
):
    """Evaluate pre-breakout pressure and completed ORB breaks on 5m candles."""
    try:
        name = str(session).lower()
        spec = SESSIONS[name]
        tz = ZoneInfo(spec["timezone"])
        stamps = np.asarray(candles["time"], dtype=np.int64)
        arrays = {key: np.asarray(candles[key], dtype=np.float64)
                  for key in ("open", "high", "low", "close")}
        if "volume" not in candles:
            volumes = None
        else:
            volumes = np.asarray(candles["volume"], dtype=np.float64)
        n = len(stamps)
        if n < 30 or any(x.ndim != 1 or len(x) != n for x in arrays.values()):
            return {"state": "INVALID_DATA", "session": name}
        if volumes is not None and (volumes.ndim != 1 or len(volumes) != n):
            return {"state": "INVALID_DATA", "session": name}
        if (np.any(np.diff(stamps) <= 0)
                or any(not np.isfinite(x).all() for x in arrays.values())
                or any(np.any(arrays[k] <= 0) for k in ("open", "high", "low", "close"))
                or np.any(arrays["high"] < arrays["low"])
                or np.any(arrays["close"] < arrays["low"])
                or np.any(arrays["close"] > arrays["high"])):
            return {"state": "INVALID_DATA", "session": name}
        if volumes is not None and (not np.isfinite(volumes).all() or np.any(volumes < 0)):
            return {"state": "INVALID_DATA", "session": name}
        # Delta candle timestamps are normalized to candle OPEN time. The last row is
        # already closed by the engine, so session eligibility must use its
        # close timestamp; otherwise ORB starts one full 5m bar late.
        now_ms = int(stamps[-1]) + 300_000
        now = _local(now_ms, tz)
        if name == "cme" and _cme_closed(now):
            return {"state": "SESSION_CLOSED", "session": name}
        start, end, expiry = _session_window(now, spec, active_minutes)
        meta = {
            "session": name, "session_timezone": spec["timezone"],
            "session_open_local": start.isoformat(), "range_start_local": start.isoformat(),
            "range_end_local": end.isoformat(), "session_date": now.date().isoformat(),
            "session_start_utc_ms": int(start.astimezone(timezone.utc).timestamp() * 1000),
            "session_end_utc_ms": int(expiry.astimezone(timezone.utc).timestamp() * 1000),
        }
        if now < end:
            return _result("BUILDING_RANGE", meta)
        if now >= expiry:
            return _result("SESSION_CLOSED", meta)
        expected = [start + timedelta(minutes=m) for m in (0, 5, 10)]
        bars = {_local(ts, tz): i for i, ts in enumerate(stamps)
                if _local(ts, tz).date() == now.date()}
        if not all(x in bars for x in expected):
            return _result("INCOMPLETE_RANGE", meta)
        indexes = [bars[x] for x in expected]
        if any(stamps[indexes[i+1]] - stamps[indexes[i]] != 300_000 for i in range(2)):
            return _result("INCOMPLETE_RANGE", meta)
        range_high = float(np.max(arrays["high"][indexes]))
        range_low = float(np.min(arrays["low"][indexes]))
        atr_v = _finite(atr_value)
        if atr_v is None or atr_v <= 0 or range_high <= range_low:
            return _result("INVALID_VOLATILITY", meta)
        width_atr = (range_high - range_low) / atr_v
        if not 0.05 <= width_atr <= 3.0:
            return _result("UNTRADEABLE_RANGE", meta, range_high=range_high,
                           range_low=range_low, range_width_atr=width_atr)

        close = arrays["close"]
        high = arrays["high"]
        low = arrays["low"]
        volume = _finite(rvol)
        if volume is None and volumes is not None:
            baseline = float(np.mean(volumes[-21:-1])) if n >= 22 else 0.0
            volume = float(volumes[-1] / baseline) if baseline > 0 else None
        e9, e21 = ema(close, 9), ema(close, 21)
        rsi_value = rsi(close, 14)
        macd_now = macd(close)
        macd_prev = macd(close[:-1]) if n > 35 else None
        hist = macd_now[2] if macd_now else None
        hist_prev = macd_prev[2] if macd_prev else None
        # Session VWAP starts at this market's local opening timestamp,
        # which also works when Sydney's DST places its open on the prior UTC day.
        session_start_ms = int(start.astimezone(timezone.utc).timestamp() * 1000)
        session_idx = int(np.searchsorted(stamps, session_start_ms, side="left"))
        vwap_value = None
        if volumes is not None and session_idx < n:
            vwap_value = vwap(high[session_idx:], low[session_idx:], close[session_idx:],
                              volumes[session_idx:], period=None)
        roc_value = roc(close, 6)
        c = float(close[-1])
        previous_close = float(close[-2])
        edge_distance_high = (range_high - c) / atr_v
        edge_distance_low = (c - range_low) / atr_v

        def flags_for(direction):
            is_long = direction == "LONG"
            return {
                "ema_9_21": bool(e9 is not None and e21 is not None
                                 and (e9 > e21 if is_long else e9 < e21)),
                "vwap": bool(vwap_value is not None
                             and (c > vwap_value if is_long else c < vwap_value)),
                "rsi": bool(rsi_value is not None
                            and (rsi_value >= 52 if is_long else rsi_value <= 48)),
                "macd_histogram": bool(hist is not None and
                                       (hist > 0 if is_long else hist < 0)),
                "macd_acceleration": bool(hist is not None and hist_prev is not None
                                          and (hist > hist_prev if is_long else hist < hist_prev)),
                "roc": bool(roc_value is not None and
                            (roc_value > 0 if is_long else roc_value < 0)),
            }

        # Pre-breakout: the market is pressing the edge while trend and
        # momentum indicators already point in the breakout direction.
        pre_candidates = []
        if 0 <= edge_distance_high <= 0.35:
            pre_candidates.append("LONG")
        if 0 <= edge_distance_low <= 0.35:
            pre_candidates.append("SHORT")
        for direction in pre_candidates:
            flags = flags_for(direction)
            passed = sum(flags.values())
            directional_pair = flags["ema_9_21"] and (
                flags["macd_histogram"] or flags["roc"]
            )
            if (volume is not None and volume >= min_rvol and directional_pair
                    and passed >= 3):
                return _result(
                    "ORB_PREBREAKOUT", meta, direction=direction,
                    range_high=range_high, range_low=range_low,
                    breakout_level=range_high if direction == "LONG" else range_low,
                    breakout_extension_atr=round(
                        edge_distance_high if direction == "LONG" else edge_distance_low, 4
                    ),
                    range_width_atr=round(width_atr, 4), rvol=round(volume, 4),
                    indicator_confirmation={"checks": flags, "passed": passed,
                        "ema9": e9, "ema21": e21, "vwap": vwap_value,
                        "rsi": rsi_value, "macd_histogram": hist, "roc": roc_value},
                    trend_confirmed=bool(_finite(adx_value) is not None
                                          and _finite(adx_value) >= adx_min),
                )

        direction = "LONG" if c > range_high else "SHORT" if c < range_low else None
        if direction is None:
            return _result("INSIDE_RANGE", meta, range_high=range_high,
                           range_low=range_low, range_width_atr=round(width_atr, 4))
        flags = flags_for(direction)
        passed = sum(flags.values())
        # Three independent indicators, with EMA or VWAP mandatory, keeps
        # breakouts directional without waiting for candle geometry.
        if passed < 3 or not (flags["ema_9_21"] or flags["vwap"]):
            return _result("INDICATOR_UNCONFIRMED", meta, direction=direction,
                range_high=range_high, range_low=range_low,
                indicator_confirmation={"checks": flags, "passed": passed})
        if volume is None or volume < min_rvol:
            return _result("LOW_VOLUME_BREAKOUT", meta, direction=direction,
                           range_high=range_high, range_low=range_low, rvol=volume)
        edge = range_high if direction == "LONG" else range_low
        extension = abs(c - edge) / atr_v
        age = 0
        for i in range(n-2, -1, -1):
            if _local(stamps[i], tz) < end:
                break
            beyond = close[i] > range_high if direction == "LONG" else close[i] < range_low
            if not beyond:
                break
            age += 1
        if age > int(max_breakout_age_candles):
            state = "STALE_BREAKOUT"
        elif extension > float(max_extension_atr):
            state = "EXTENDED_BREAKOUT"
        else:
            state = "ORB_BREAKOUT"
        return _result(
            state, meta, direction=direction, range_high=range_high, range_low=range_low,
            breakout_level=edge, breakout_extension_atr=round(extension, 4),
            range_width_atr=round(width_atr, 4), breakout_age_candles=age,
            rvol=round(volume, 4), indicator_confirmation={
                "checks": flags, "passed": passed, "ema9": e9, "ema21": e21,
                "vwap": vwap_value, "rsi": rsi_value, "macd_histogram": hist, "roc": roc_value,
            }, trend_confirmed=bool(_finite(adx_value) is not None and _finite(adx_value) >= adx_min),
            di_aligned=None, adx_rising=bool(adx_rising), failed_breakout=False,
        )
    except (KeyError, TypeError, ValueError, OverflowError, IndexError, FloatingPointError):
        return {"state": "INVALID_DATA", "session": str(session).lower()}
