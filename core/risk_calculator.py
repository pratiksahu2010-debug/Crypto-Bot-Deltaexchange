"""Structure-aware ATR risk levels for signal-only alerts."""
from typing import Optional
from dataclasses import dataclass
import math
@dataclass
class RiskLevels:
    entry: float; entry_zone_low: float; entry_zone_high: float
    stop_loss: float; tp1: float; tp2: float; tp3: float
    risk_pct: float; rr1: float; rr2: float; rr3: float; invalidation: float

def calculate_risk(price: float, atr: float, direction: str, atr_mult_sl: float=1.5,
                   atr_mult_tp1: float=2.0, atr_mult_tp2: float=3.0, atr_mult_tp3: float=4.0,
                   swing_low: Optional[float]=None, swing_high: Optional[float]=None,
                   cost_bps: Optional[float]=None, min_net_rr: float=0.0) -> Optional[RiskLevels]:
    try:
        price = float(price)
        atr = float(atr)
        atr_mult_sl = float(atr_mult_sl)
        reward_multipliers = tuple(float(x) for x in (atr_mult_tp1, atr_mult_tp2, atr_mult_tp3))
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        direction not in ("LONG", "SHORT")
        or not all(math.isfinite(x) for x in (price, atr, atr_mult_sl, *reward_multipliers))
        or price <= 0 or atr <= 0 or atr / price >= 0.25 or atr_mult_sl <= 0
        or any(x <= 0 for x in reward_multipliers)
        or not (reward_multipliers[0] < reward_multipliers[1] < reward_multipliers[2])
    ):
        return None
    is_long=direction=="LONG"
    # Structural stop with an ATR safety buffer; reject pathological distant structure.
    if is_long:
        structural=(swing_low-0.15*atr) if swing_low and swing_low<price else price-atr*atr_mult_sl
        sl=max(price-atr*2.5, min(price-atr*0.8, structural))
        if sl>=price: sl=price-atr*atr_mult_sl
        risk=price-sl
        # Legacy parameter names are retained for compatibility; targets are R-multiples.
        targets=[price+multiplier*risk for multiplier in reward_multipliers]
        zone=(price-0.15*atr,price+0.10*atr)
    else:
        structural=(swing_high+0.15*atr) if swing_high and swing_high>price else price+atr*atr_mult_sl
        sl=min(price+atr*2.5, max(price+atr*0.8, structural))
        if sl<=price: sl=price+atr*atr_mult_sl
        risk=sl-price
        # Legacy parameter names are retained for compatibility; targets are R-multiples.
        targets=[price-multiplier*risk for multiplier in reward_multipliers]
        zone=(price-0.10*atr,price+0.15*atr)
    if risk <= 0:
        return None
    # High-volatility short setups can otherwise produce zero/negative targets,
    # and extreme long setups can place the stop or entry band below zero.
    # Such levels are not actionable prices and must never reach an alert.
    if (
        not all(math.isfinite(value) and value > 0 for value in (sl, *targets, zone[0], zone[1]))
        or zone[0] >= zone[1]
    ):
        return None
    # Legacy cost_bps/min_net_rr arguments remain accepted for API compatibility,
    # but are intentionally ignored. Signal qualification and targets are gross
    # structural ATR calculations; fees, spread and slippage are not gates.
    rrs=[abs(t-price)/risk for t in targets]
    return RiskLevels(round(price,8),round(zone[0],8),round(zone[1],8),round(sl,8),
        round(targets[0],8),round(targets[1],8),round(targets[2],8),round(risk/price*100,4),
        round(rrs[0],2),round(rrs[1],2),round(rrs[2],2),round(sl,8))


def classify_entry_state(live_price: float, entry: float, zone_low: float,
                          zone_high: float, direction: str) -> str:
    """Classify entry without widening the actionable zone into a chase."""
    try:
        import math
        price, anchor = float(live_price), float(entry)
        low, high = float(zone_low), float(zone_high)
    except (TypeError, ValueError, OverflowError):
        return "EXPIRED"
    if not all(math.isfinite(x) for x in (price, anchor, low, high)) or min(price, anchor, low) <= 0 or high <= low:
        return "EXPIRED"
    if low <= price <= high:
        return "IMMEDIATE"
    # Outside the entry band but still close enough to monitor for a pullback.
    # It is never an alert until price returns to the original zone.
    width = high - low
    if direction == "LONG" and high < price <= high + 1.5 * width:
        return "RETEST"
    if direction == "SHORT" and low - 1.5 * width <= price < low:
        return "RETEST"
    return "EXPIRED"


def is_entry_price_actionable(live_price: float, zone_low: float, zone_high: float) -> bool:
    """Backward-compatible strict check for immediate entries only."""
    return classify_entry_state(live_price, (zone_low + zone_high) / 2.0,
                                zone_low, zone_high, "LONG") == "IMMEDIATE"
