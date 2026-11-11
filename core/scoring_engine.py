"""Indicator scoring for independent signal engines; component maxima are explicit."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import math


_NUMERIC_BOUNDS = {
    "ema9": (0.0, None), "ema21": (0.0, None),
    "close": (0.0, None), "vwap": (0.0, None), "orb_vwap": (0.0, None), "atr": (0.0, None),
    "adx": (0.0, 100.0), "rsi": (0.0, 100.0), "rsi_prev": (0.0, 100.0),
    "rvol": (0.0, None), "orb_session_rvol": (0.0, None),
    "orb_rolling_rvol": (0.0, None), "atr_pct": (0.0, None),
    "body_ratio": (0.0, 1.0), "atr_expansion": (0.0, None),
    "close_location": (0.0, 1.0),
    "orb_breakout_extension_atr": (0.0, None),
    "orb_range_width_atr": (0.0, None), "roc": (None, None),
    "plus_di": (0.0, 100.0), "minus_di": (0.0, 100.0),
    "upper_wick_ratio": (0.0, 1.0), "lower_wick_ratio": (0.0, 1.0),
}
_BOOLEAN_KEYS = {
    "di_direction_ok", "adx_rising", "structure_direction_ok",
    "squeeze_release", "compression_active", "bw_expand", "ema_aligned", "macd_accel",
    "macd_direction_ok", "rsi_accel", "structure_break",
    "roc_direction_ok", "orb_trend_confirmed", "orb_di_aligned",
    "orb_adx_rising", "orb_wick_quality",
}


def _validated_indicators(indicators: Dict[str, Any]) -> Dict[str, Any]:
    """Copy indicator data and fail closed on malformed/non-finite values."""
    clean = dict(indicators) if isinstance(indicators, dict) else {}
    for key, (minimum, maximum) in _NUMERIC_BOUNDS.items():
        raw = clean.get(key)
        if raw is None:
            continue
        try:
            if isinstance(raw, bool):
                raise ValueError("boolean is not a numeric indicator")
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            clean[key] = None
            continue
        if (
            not math.isfinite(value)
            or (minimum is not None and value < minimum)
            or (maximum is not None and value > maximum)
        ):
            clean[key] = None
        else:
            clean[key] = value

    for key in _BOOLEAN_KEYS:
        value = clean.get(key)
        if value is None and key in ("di_direction_ok", "orb_di_aligned"):
            # Preserve the tri-state meaning: unavailable DI is not opposition.
            clean[key] = None
            continue
        # Accept native and NumPy booleans, but never truthy strings or numbers.
        clean[key] = value is True or type(value).__name__ == "bool_" and bool(value)
    return clean


def _validate_direction(direction: str) -> None:
    if direction not in ("LONG", "SHORT"):
        raise ValueError("direction must be LONG or SHORT")
from utils.logger import logger
from config import settings

@dataclass
class ScoreComponent:
    name: str
    points: float
    max_points: float
    value: Any
    direction: str  # bullish / bearish / neutral
    explanation: str
    valid: bool = True


@dataclass
class ScoreResult:
    total: float
    max_total: float = 10.0
    components: List[ScoreComponent] = field(default_factory=list)
    grade: str = "NONE"
    direction: str = "NEUTRAL"

    def __post_init__(self):
        # Grade against this result's actual maximum. Normal is an 8.5-point
        # model; Momentum remains on its existing 10-point scale.
        try:
            maximum = float(self.max_total)
            score = float(self.total)
            ratio = score / maximum if math.isfinite(maximum) and maximum > 0 else 0.0
        except (TypeError, ValueError, OverflowError):
            ratio = 0.0
        if ratio >= 0.90:
            self.grade = "HIGH-CONVICTION"
        elif ratio >= 0.80:
            self.grade = "STRONG"
        elif ratio >= 0.70:
            self.grade = "SIGNAL"
        elif ratio >= 0.60:
            self.grade = "WATCHLIST"
        else:
            self.grade = "NONE"


class ScoringEngine:
    """Normal engine component weights; the five indicators sum to 8.5."""

    NORMAL_WEIGHTS = {
        # Normal's five scored indicators total 8.5 points.
        "ema_trend": 2.0,
        "vwap": 2.0,
        "rsi": 1.5,
        "adx": 2.0,
        "atr": 1.0,
    }

    MOMENTUM_WEIGHTS = {
        "squeeze_release": 1.5,
        "atr_expansion": 1.2,
        "rvol_surge": 1.2,
        "ema_alignment": 1.0,
        "macd_accel": 1.0,
        "rsi_accel": 0.8,
        "candle_body": 1.0,
        "structure_break": 1.0,
        "bandwidth_expand": 0.8,
        "roc": 0.5,
    }

    def score_normal(self, indicators: Dict[str, Any], direction: str, adx_min: float = 25.0) -> ScoreResult:
        """Score the five Normal indicators; maximum score is 8.5 points."""
        _validate_direction(direction)
        values = _validated_indicators(indicators)
        is_long = direction == "LONG"
        components: List[ScoreComponent] = []

        def add(name, points, maximum, value, explanation):
            components.append(ScoreComponent(
                name, float(points), float(maximum), value, direction,
                explanation, float(points) > 0,
            ))

        ema9, ema21 = values.get("ema9"), values.get("ema21")
        ema_ok = ema9 is not None and ema21 is not None and (ema9 > ema21 if is_long else ema9 < ema21)
        add("EMA 9/21", 2.0 if ema_ok else 0.0, 2.0, (ema9, ema21),
            "EMA9/EMA21 bullish alignment" if ema_ok and is_long else
            "EMA9/EMA21 bearish alignment" if ema_ok else "EMA alignment absent")

        close, vwap_value = values.get("close"), values.get("vwap")
        vwap_ok = close is not None and vwap_value is not None and (close > vwap_value if is_long else close < vwap_value)
        add("VWAP", 2.0 if vwap_ok else 0.0, 2.0, vwap_value,
            "Price above session VWAP" if vwap_ok and is_long else
            "Price below session VWAP" if vwap_ok else "VWAP alignment absent")

        rsi_value = values.get("rsi")
        rsi_ok = rsi_value is not None and (rsi_value > 50.0 if is_long else rsi_value < 50.0)
        add("RSI 14", 1.5 if rsi_ok else 0.0, 1.5, rsi_value,
            f"RSI {rsi_value:.1f} supports direction" if rsi_ok else "RSI not directionally aligned")

        adx_value = values.get("adx")
        adx_rising = bool(values.get("adx_rising"))
        plus_di, minus_di = values.get("plus_di"), values.get("minus_di")
        di_ok = (
            plus_di is not None and minus_di is not None
            and (plus_di > minus_di if is_long else minus_di > plus_di)
        )
        try:
            configured_adx_floor = float(adx_min)
        except (TypeError, ValueError, OverflowError):
            configured_adx_floor = 25.0
        if not math.isfinite(configured_adx_floor):
            configured_adx_floor = 25.0
        full_adx_floor = max(25.0, configured_adx_floor)
        early_adx_floor = max(20.0, full_adx_floor - 5.0)
        adx_full = adx_value is not None and adx_value >= full_adx_floor
        adx_early = (
            adx_value is not None and early_adx_floor <= adx_value < full_adx_floor
            and adx_rising and di_ok
        )
        adx_points = 2.0 if adx_full else 1.0 if adx_early else 0.0
        add("ADX 14", adx_points, 2.0, adx_value,
            f"ADX {adx_value:.1f}; full-strength band" if adx_full else
            f"ADX {adx_value:.1f}; rising early band with DI alignment" if adx_early else
            "ADX below strength requirement or early confirmation absent")

        atr_value = values.get("atr")
        atr_ok = atr_value is not None and atr_value > 0
        add("ATR 14", 1.0 if atr_ok else 0.0, 1.0, atr_value,
            "ATR valid" if atr_ok else "ATR unavailable")

        total = round(sum(item.points for item in components), 2)
        return ScoreResult(total=total, max_total=8.5, components=components, direction=direction)

    def score_momentum_explosion(self, indicators: Dict[str, Any], direction: str) -> ScoreResult:
        """Score confirmed early momentum without double-penalising lagging indicators."""
        _validate_direction(direction)
        values = _validated_indicators(indicators)
        is_long = direction == "LONG"
        components: List[ScoreComponent] = []

        def add(name, points, maximum, value, explanation):
            components.append(ScoreComponent(name, float(points), float(maximum),
                value, direction, explanation, float(points) > 0))

        compression = bool(values.get("compression_active") or values.get("squeeze_release"))
        add("Volatility Compression/Release", 1.0 if compression else 0.0, 1.0,
            compression, "Recent compression or release" if compression else "No recent squeeze context")

        atr_exp = values.get("atr_expansion")
        atr_floor = 1.03 if values.get("compression_active") is True else 1.05 if values.get("squeeze_release") is True else 1.08
        atr_ok = atr_exp is not None and atr_exp >= atr_floor
        add("ATR Expansion", 1.25 if atr_ok else 0.0, 1.25, atr_exp,
            f"ATR expansion meets {atr_floor:.2f}x state threshold" if atr_ok else f"ATR expansion below {atr_floor:.2f}x")

        rvol = values.get("rvol")
        rvol_floor = 1.20 if values.get("compression_active") is True else 1.25
        rvol_ok = rvol is not None and rvol >= rvol_floor
        add("Relative Volume", 1.25 if rvol_ok else 0.0, 1.25, rvol,
            f"RVOL meets {rvol_floor:.2f}x state threshold" if rvol_ok else f"RVOL below {rvol_floor:.2f}x")

        ema_ok = bool(values.get("ema_aligned"))
        structure_ok = bool(values.get("structure_break"))
        momentum_votes = sum(bool(values.get(key)) for key in (
            "macd_direction_ok", "roc_direction_ok", "macd_accel", "rsi_accel"
        ))
        trend_ok = ema_ok or structure_ok or momentum_votes >= 3
        add("Trend/Structure Evidence", 1.5 if trend_ok else 0.0, 1.5,
            {"ema_aligned": ema_ok, "structure_break": structure_ok, "momentum_votes": momentum_votes},
            "EMA/structure or strong independent momentum evidence" if trend_ok else "Trend/structure evidence absent")

        macd_ok = bool(values.get("macd_direction_ok") or values.get("macd_accel"))
        add("MACD Direction/Acceleration", 1.25 if macd_ok else 0.0, 1.25, macd_ok,
            "MACD direction or acceleration supports the move" if macd_ok else "MACD evidence absent")

        rsi_value = values.get("rsi")
        rsi_floor = 52.0 if values.get("compression_active") is True else 54.0
        rsi_ceiling = 48.0 if values.get("compression_active") is True else 46.0
        rsi_direction_ok = rsi_value is not None and (
            rsi_value >= rsi_floor if is_long else rsi_value <= rsi_ceiling
        )
        rsi_ok = rsi_direction_ok or bool(values.get("rsi_accel")) or momentum_votes >= 3
        add("RSI Momentum", 1.0 if rsi_ok else 0.0, 1.0, rsi_value,
            "RSI direction/acceleration supports early momentum" if rsi_ok else "RSI momentum absent")

        roc_ok = bool(values.get("roc_direction_ok"))
        add("ROC Direction", 1.25 if roc_ok else 0.0, 1.25, values.get("roc"),
            "ROC supports direction" if roc_ok else "ROC not aligned")

        bandwidth = bool(values.get("bw_expand"))
        add("Bandwidth Expansion", 0.5 if bandwidth else 0.0, 0.5, bandwidth,
            "Bollinger bandwidth expanding" if bandwidth else "Bandwidth not expanding")

        adx = values.get("adx")
        adx_ok = adx is not None and (
            adx >= 18.0 or (adx >= 14.0 and bool(values.get("adx_rising")))
        )
        add("ADX Trend Development", 1.0 if adx_ok else 0.0, 1.0, adx,
            "ADX strong or rising from early band" if adx_ok else "ADX trend development absent")

        total = round(sum(item.points for item in components), 2)
        return ScoreResult(total=total, max_total=10.0, components=components, direction=direction)

    def score_orb(
        self, indicators: Dict[str, Any], direction: str,
        min_rvol: float = 1.10, max_extension_atr: float = 0.80,
    ) -> ScoreResult:
        """Score an ORB using the same session-specific limits as its detector."""
        _validate_direction(direction)
        values = _validated_indicators(indicators)
        is_long = direction == "LONG"
        components: List[ScoreComponent] = []

        def add(name, points, maximum, value, explanation):
            components.append(ScoreComponent(name, float(points), float(maximum),
                value, direction, explanation, float(points) > 0))

        # General rolling RVOL is not an ORB qualification input.
        rvol = values.get("orb_session_rvol")
        try:
            rvol_floor = float(min_rvol)
        except (TypeError, ValueError, OverflowError):
            rvol_floor = 1.10
        volume_ok = rvol is not None and math.isfinite(rvol_floor) and rvol >= rvol_floor
        add("ORB Session RVOL", 1.5 if volume_ok else 0.0, 1.5, rvol,
            f"Session RVOL meets {rvol_floor:.2f} minimum" if volume_ok
            else f"Session RVOL below {rvol_floor:.2f}")

        rolling_rvol = values.get("orb_rolling_rvol")
        rolling_floor = max(1.20, rvol_floor * 0.85)
        rolling_ok = rolling_rvol is not None and rolling_rvol >= rolling_floor
        add("ORB Rolling RVOL", 0.5 if rolling_ok else 0.0, 0.5, rolling_rvol,
            f"Rolling RVOL meets {rolling_floor:.2f} minimum" if rolling_ok
            else f"Rolling RVOL below {rolling_floor:.2f}")

        e9, e21 = values.get("ema9"), values.get("ema21")
        ema_ok = e9 is not None and e21 is not None and (e9 > e21 if is_long else e9 < e21)
        add("ORB EMA 9/21", 1.5 if ema_ok else 0.0, 1.5, (e9, e21),
            "EMA direction aligned" if ema_ok else "EMA direction not aligned")

        close = values.get("close")
        vwap_value = values.get("orb_vwap") if values.get("orb_vwap") is not None else values.get("vwap")
        vwap_ok = close is not None and vwap_value is not None and (close > vwap_value if is_long else close < vwap_value)
        add("ORB VWAP", 1.25 if vwap_ok else 0.0, 1.25, vwap_value,
            "VWAP supports direction" if vwap_ok else "VWAP not aligned")

        rsi_value = values.get("rsi")
        rsi_ok = rsi_value is not None and (rsi_value >= 55 if is_long else rsi_value <= 45)
        add("ORB RSI", 1.25 if rsi_ok else 0.0, 1.25, rsi_value,
            "RSI meets ORB directional threshold" if rsi_ok else "RSI below ORB directional threshold")

        macd_hist = values.get("macd_hist")
        atr_for_macd = values.get("atr")
        try:
            macd_floor = max(float(atr_for_macd) * 0.01, 1e-12)
            macd_ok = (
                macd_hist is not None and math.isfinite(float(macd_hist))
                and abs(float(macd_hist)) >= macd_floor
                and (float(macd_hist) > 0 if is_long else float(macd_hist) < 0)
            )
        except (TypeError, ValueError, OverflowError):
            macd_ok = False
        add("ORB MACD", 1.5 if macd_ok else 0.0, 1.5, macd_hist,
            "MACD histogram has meaningful directional magnitude" if macd_ok
            else "MACD histogram is weak or not directionally aligned")

        roc_ok = bool(values.get("roc_direction_ok"))
        add("ORB ROC", 1.25 if roc_ok else 0.0, 1.25, values.get("roc"),
            "ROC confirms direction" if roc_ok else "ROC not aligned")

        atr_value = values.get("atr")
        atr_ok = atr_value is not None and atr_value > 0
        add("ORB ATR", 0.5 if atr_ok else 0.0, 0.5, atr_value,
            "ATR available" if atr_ok else "ATR unavailable")

        distance = values.get("orb_breakout_extension_atr")
        try:
            extension_limit = float(max_extension_atr)
        except (TypeError, ValueError, OverflowError):
            extension_limit = 0.80
        distance_ok = distance is not None and math.isfinite(extension_limit) and distance <= extension_limit
        add("ORB Breakout Extension", 0.75 if distance_ok else 0.0, 0.75, distance,
            f"Extension within {extension_limit:.2f} ATR qualification limit"
            if distance_ok else f"Extension exceeds {extension_limit:.2f} ATR")

        total = round(sum(item.points for item in components), 2)
        return ScoreResult(total=total, max_total=10.0, components=components, direction=direction)

    def score_momentum(self, indicators: Dict[str, Any], direction: str) -> ScoreResult:
        """Score trend continuation from independent indicator families."""
        _validate_direction(direction)
        values = _validated_indicators(indicators)
        is_long = direction == "LONG"
        components: List[ScoreComponent] = []

        def add(name, points, maximum, value, explanation):
            components.append(ScoreComponent(name, float(points), float(maximum),
                value, direction, explanation, float(points) > 0))

        squeeze = bool(values.get("squeeze_release") or values.get("compression_active"))
        add("Squeeze State", 1.0 if squeeze else 0.0, 1.0, squeeze,
            "Volatility compression/release present" if squeeze else "No squeeze state")

        atr_exp = values.get("atr_expansion")
        atr_ok = atr_exp is not None and atr_exp >= 1.08
        add("ATR Expansion", 1.5 if atr_ok else 0.0, 1.5, atr_exp,
            "Volatility expanding" if atr_ok else "ATR expansion absent")

        rvol = values.get("rvol")
        volume_ok = rvol is not None and rvol >= 1.25
        add("Relative Volume", 1.5 if volume_ok else 0.0, 1.5, rvol,
            "Participation expanding" if volume_ok else "Relative volume low")

        ema_ok = bool(values.get("ema_aligned"))
        add("EMA Alignment", 1.5 if ema_ok else 0.0, 1.5, ema_ok,
            "EMA trend aligned" if ema_ok else "EMA trend not aligned")

        macd_ok = bool(values.get("macd_direction_ok"))
        add("MACD", 1.5 if macd_ok else 0.0, 1.5, macd_ok,
            "MACD direction/acceleration aligned" if macd_ok else "MACD not aligned")

        rsi_value = values.get("rsi")
        rsi_ok = rsi_value is not None and (rsi_value >= 55 if is_long else rsi_value <= 45)
        add("RSI", 1.0 if rsi_ok else 0.0, 1.0,
            rsi_value, "RSI momentum aligned" if rsi_ok else "RSI not aligned")

        roc_ok = bool(values.get("roc_direction_ok"))
        add("ROC", 1.5 if roc_ok else 0.0, 1.5, values.get("roc"),
            "ROC aligned" if roc_ok else "ROC not aligned")

        bandwidth = bool(values.get("bw_expand"))
        add("Bollinger Bandwidth", 0.5 if bandwidth else 0.0, 0.5, bandwidth,
            "Bandwidth expanding" if bandwidth else "Bandwidth flat")

        total = round(sum(item.points for item in components), 2)
        return ScoreResult(total=total, components=components, direction=direction)
