"""Lightweight backtest scaffold (no look-ahead)."""
from typing import List, Dict, Any

def run_simple_backtest(candles: List[dict], signals: List[dict]) -> Dict[str, Any]:
    """Placeholder that returns structure only. Extend with real simulation."""
    return {
        "trades": len(signals),
        "note": "Scaffold only — implement full candle-close simulation for production use",
    }
