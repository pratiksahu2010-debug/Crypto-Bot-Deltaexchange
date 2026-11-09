"""EMA pullback/recovery strategy."""
def evaluate(indicators: dict, direction: str) -> bool:
    close=indicators.get("close"); e9=indicators.get("ema9"); e21=indicators.get("ema21")
    rsi=indicators.get("rsi"); vwap=indicators.get("vwap"); adx=indicators.get("adx")
    if any(v is None for v in (close,e9,e21,rsi,vwap,adx)): return False
    if adx<20: return False
    # Pullback near fast EMA, then close back on trend side with momentum.
    near = abs(close-e9)/close <= 0.004 if close else False
    if direction=="LONG":
        return e9>e21 and close>=e9 and near and 48<=rsi<=68 and close>vwap
    if direction=="SHORT":
        return e9<e21 and close<=e9 and near and 32<=rsi<=52 and close<vwap
    return False
