"""Three-stage Momentum Explosion qualification.

Stage 1 discovers early directional candidates; Stage 2 confirms momentum,
participation and volatility using independent evidence groups. Indicator
families contribute evidence instead of all being mandatory on one candle.
"""
import math


def _num(data, key):
    try:
        value = float(data.get(key))
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _flag(data, key):
    value = data.get(key)
    return value is True or (type(value).__name__ == "bool_" and bool(value))


def _directional_di(data, direction):
    plus_di, minus_di = _num(data, "plus_di"), _num(data, "minus_di")
    if plus_di is None or minus_di is None:
        return None
    return plus_di > minus_di if direction == "LONG" else minus_di > plus_di


def diagnose(indicators: dict) -> dict:
    """Return auditable stage-two evidence without mutating the input."""
    if not isinstance(indicators, dict):
        return {"qualified": False, "reason": "invalid_indicators"}
    direction = str(indicators.get("direction", "")).upper()
    if direction not in ("LONG", "SHORT"):
        return {"qualified": False, "reason": "invalid_direction"}

    rvol = _num(indicators, "rvol")
    expansion = _num(indicators, "atr_expansion")
    adx = _num(indicators, "adx")
    if rvol is None or expansion is None or adx is None:
        return {"qualified": False, "reason": "missing_core_data"}
    if rvol < 1.20:
        return {"qualified": False, "reason": "participation_below_floor"}
    if expansion < 1.03:
        return {"qualified": False, "reason": "volatility_not_expanding"}

    adx_rising = _flag(indicators, "adx_rising")
    adx_ok = adx >= 17.0
    if not adx_ok:
        return {"qualified": False, "reason": "trend_strength_not_confirmed"}

    # Stage 2: two independent directional momentum votes, with at least one
    # actual direction signal (MACD or ROC). Acceleration may supply the other.
    macd_direction = _flag(indicators, "macd_direction_ok")
    roc_direction = _flag(indicators, "roc_direction_ok")
    macd_accel = _flag(indicators, "macd_accel")
    rsi_accel = _flag(indicators, "rsi_accel")
    momentum_votes = sum((macd_direction, roc_direction, macd_accel, rsi_accel))
    momentum_ok = momentum_votes >= 2 and (macd_direction or roc_direction)

    ema_aligned = _flag(indicators, "ema_aligned")
    structure_break = _flag(indicators, "structure_break")
    trend_ok = ema_aligned or structure_break

    # Compression alone is Stage 1 watchlist context, not a confirmed explosion.
    # Stage 2 needs an actual release, bandwidth expansion, or strong ATR expansion.
    squeeze_release = _flag(indicators, "squeeze_release")
    bandwidth_expanding = _flag(indicators, "bw_expand")
    volatility_ok = squeeze_release or bandwidth_expanding or expansion >= 1.12

    rsi = _num(indicators, "rsi")
    rsi_ok = (
        rsi is not None
        and (rsi >= 52.0 if direction == "LONG" else rsi <= 48.0)
    ) or rsi_accel

    di_aligned = _directional_di(indicators, direction)
    # Missing DI is diagnostic rather than an automatic rejection. A clearly
    # opposing DI is tolerated only when ADX is rising and momentum is strong.
    di_ok = (
        di_aligned is not False
        or (adx_rising and momentum_votes >= 3)
    )

    # The release may qualify without a textbook squeeze if volatility is
    # expanding and directional trend/structure evidence is present.
    qualified = bool(
        momentum_ok and volatility_ok and di_ok
        and (trend_ok or momentum_votes >= 3)
        and (rsi_ok or momentum_votes >= 3)
    )
    return {
        "qualified": qualified,
        "reason": "confirmed" if qualified else "insufficient_independent_evidence",
        "direction": direction,
        "rvol": rvol,
        "atr_expansion": expansion,
        "adx": adx,
        "adx_rising": adx_rising,
        "momentum_votes": momentum_votes,
        "trend_ok": trend_ok,
        "volatility_ok": volatility_ok,
        "di_aligned": di_aligned,
        "rsi_ok": bool(rsi_ok),
    }


def evaluate(indicators: dict) -> bool:
    """True only for a confirmed directional release; see diagnose() for why."""
    return bool(diagnose(indicators).get("qualified"))
