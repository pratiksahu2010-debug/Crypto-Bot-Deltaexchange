import numpy as np

from core.engine import _closed_15m_context_from_5m


def _bars(count=78):
    start = 1_800_000_000_000
    start -= start % 900_000
    close = 100.0 + np.arange(count, dtype=float) * 0.08
    open_ = close - 0.02
    return {
        "time": start + np.arange(count, dtype=np.int64) * 300_000,
        "open": open_,
        "high": close + 0.05,
        "low": open_ - 0.05,
        "close": close,
        "volume": np.full(count, 100.0),
    }


def test_15m_context_uses_only_complete_three_bar_buckets():
    bars = _bars(77)  # 25 complete 15m bars plus two bars in a forming bucket
    result = _closed_15m_context_from_5m(bars)
    assert result["available"] is True
    assert result["bars"] == 25
    assert result["bias"] == "BULLISH"
    assert result["close"] == bars["close"][74]


def test_15m_context_rejects_missing_constituent_bar():
    bars = _bars(77)
    # Remove the middle 5m constituent from one 15m bucket.
    for key in bars:
        bars[key] = np.delete(bars[key], 30)
    result = _closed_15m_context_from_5m(bars)
    assert result["available"] is True
    assert result["bars"] == 24
    assert result["close"] == bars["close"][73]


def test_15m_context_fails_closed_when_history_is_insufficient():
    result = _closed_15m_context_from_5m(_bars(72))
    assert result["available"] is False
    assert result["bias"] == "NEUTRAL"
