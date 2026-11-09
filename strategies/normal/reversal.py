"""Exhaustion/reversal rejection with directional confirmation."""
def evaluate(indicators: dict, direction: str) -> bool:
    rsi=indicators.get("rsi"); close=indicators.get("close"); e9=indicators.get("ema9")
    vwap=indicators.get("vwap"); body=indicators.get("body_ratio")
    if any(v is None for v in (rsi,close,e9,vwap,body)): return False
    prev_rsi=indicators.get("rsi_prev")
    if direction=="LONG":
        return rsi>=prev_rsi and rsi<40 and close>e9 and close>vwap and body>=0.25 if prev_rsi is not None else False
    if direction=="SHORT":
        return rsi<=prev_rsi and rsi>60 and close<e9 and close<vwap and body>=0.25 if prev_rsi is not None else False
    return False
