"""Volume-confirmed directional breakout."""
def evaluate(indicators: dict, direction: str) -> bool:
    close=indicators.get("close"); high=indicators.get("high"); low=indicators.get("low")
    rvol=indicators.get("rvol"); body=indicators.get("body_ratio"); adx=indicators.get("adx")
    if any(v is None for v in (close,high,low,rvol,body,adx)): return False
    resistance=indicators.get("resistance"); support=indicators.get("support")
    if direction=="LONG":
        return resistance is not None and close > resistance and rvol>=1.5 and body>=0.5 and adx>=17
    if direction=="SHORT":
        return support is not None and close < support and rvol>=1.5 and body>=0.5 and adx>=20
    return False
