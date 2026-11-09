"""Average Directional Index using correctly seeded Wilder smoothing."""
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple

def _wilder(arr: np.ndarray, period: int, start: int = 0) -> np.ndarray:
    out = np.full(len(arr), np.nan, dtype=np.float64)
    if period < 1 or start + period > len(arr):
        return out
    seed_idx = start + period - 1
    seed = arr[start:seed_idx + 1]
    if not np.all(np.isfinite(seed)):
        return out
    out[seed_idx] = float(np.mean(seed))
    for i in range(seed_idx + 1, len(arr)):
        if np.isfinite(arr[i]) and np.isfinite(out[i-1]):
            out[i] = (out[i-1] * (period - 1) + arr[i]) / period
    return out

def _calculate(high, low, close, period):
    try:
        high = np.asarray(high, dtype=np.float64)
        low = np.asarray(low, dtype=np.float64)
        close = np.asarray(close, dtype=np.float64)
        p = int(period)
    except (TypeError, ValueError, OverflowError):
        return None
    n = len(close)
    if (
        high.ndim != 1 or low.ndim != 1 or close.ndim != 1
        or p != period or p < 1 or not (len(high) == len(low) == n)
        or n < 2 * p
        or not all(np.all(np.isfinite(x)) for x in (high, low, close))
        or np.any(high <= 0) or np.any(low <= 0) or np.any(close <= 0)
        or np.any(high < low) or np.any(close > high) or np.any(close < low)
    ):
        return None
    period = p
    tr=np.zeros(n); plus=np.zeros(n); minus=np.zeros(n)
    tr[0]=high[0]-low[0]
    for i in range(1,n):
        up=high[i]-high[i-1]; down=low[i-1]-low[i]
        plus[i]=up if up>down and up>0 else 0.
        minus[i]=down if down>up and down>0 else 0.
        tr[i]=max(high[i]-low[i],abs(high[i]-close[i-1]),abs(low[i]-close[i-1]))
    # Seed Wilder TR/DM using bars 1..period (first bar has no directional movement)
    tr_s=_wilder(tr,period,start=1)
    plus_s=_wilder(plus,period,start=1)
    minus_s=_wilder(minus,period,start=1)
    pdi=np.full(n,np.nan); mdi=np.full(n,np.nan); dx=np.full(n,np.nan)
    for i in range(period,n):
        if not np.isfinite(tr_s[i]) or tr_s[i] <= 0: continue
        pdi[i]=100*plus_s[i]/tr_s[i]; mdi[i]=100*minus_s[i]/tr_s[i]
        total=pdi[i]+mdi[i]
        dx[i]=100*abs(pdi[i]-mdi[i])/total if total>0 else 0.
    # ADX seed must use `period` valid DX observations, not zero-filled warmup.
    adx_s=_wilder(dx,period,start=period)
    return adx_s,pdi,mdi

def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int=14) -> Optional[Tuple[float,float,float]]:
    result=_calculate(high,low,close,period)
    if result is None: return None
    a,p,m=(x[-1] for x in result)
    if not (np.isfinite(a) and np.isfinite(p) and np.isfinite(m)): return None
    return float(a),float(p),float(m)

def adx_series(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int=14):
    """Full ADX, +DI, -DI series; NaN during the true warm-up."""
    return _calculate(high,low,close,period)
