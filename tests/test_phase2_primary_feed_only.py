import pytest

from config import settings
from exchange.delta_ws import DeltaFuturesStream
from exchange.symbol_manager import SymbolManager


def test_delta_websocket_subscribes_only_to_primary_signal_timeframe(monkeypatch):
    monkeypatch.setattr(settings, "candle_primary_tf", "5m")
    monkeypatch.setattr(settings, "candle_confirm_tf", "15m")

    class Rest:
        def delta_symbol_for_pair(self, pair):
            return "BTCUSD"

    stream = DeltaFuturesStream(object(), Rest())
    pairs = ["B-BTC_USDT"]
    symbols = [stream.rest.delta_symbol_for_pair(pair) for pair in pairs]
    intervals = (settings.candle_primary_tf,)
    stream._channels = {f"candlestick_{interval}": symbols for interval in intervals}

    assert stream._channels == {"candlestick_5m": ["BTCUSD"]}


def test_symbol_quarantine_uses_only_primary_timeframe(monkeypatch):
    monkeypatch.setattr(settings, "candle_primary_tf", "5m")
    monkeypatch.setattr(settings, "candle_confirm_tf", "15m")

    class Rest:
        def get_stale_market_timeframes(self):
            return {
                "B-BTC_USDT": {"15m": 3},
                "B-ETH_USDT": {"5m": 2},
            }

    manager = SymbolManager(Rest())
    manager._quarantined = {}
    manager._refresh_quarantine_from_rest()

    assert "B-BTC_USDT" not in manager._quarantined
    assert "B-ETH_USDT" in manager._quarantined


@pytest.mark.asyncio
async def test_quarantine_recovery_probes_only_primary_timeframe(monkeypatch):
    monkeypatch.setattr(settings, "candle_primary_tf", "5m")
    monkeypatch.setattr(settings, "candle_confirm_tf", "15m")
    calls = []

    class Rest:
        async def get_candles(self, pair, interval, limit=100):
            calls.append((pair, interval, limit))
            return [{"time": i} for i in range(300)]

        def get_stale_market_timeframes(self):
            return {}

    manager = SymbolManager(Rest())
    manager._last_liquidity_refresh = 10**12
    manager._quarantine_retry_seconds = 900
    manager._quarantined = {
        "B-BTC_USDT": {"since": 0, "retry_at": 0, "reason": "stale"}
    }
    await manager.refresh_universes()

    assert calls == [("B-BTC_USDT", "5m", 300)]
    assert "B-BTC_USDT" not in manager._quarantined
