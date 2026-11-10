import asyncio
import time

import pytest

from exchange.delta_rest import DeltaExchangeRest


@pytest.mark.asyncio
async def test_concurrent_identical_candle_requests_are_coalesced(monkeypatch):
    now = 1_800_000_000 + 1200
    monkeypatch.setattr("exchange.delta_rest.time.time", lambda: now)
    client = DeltaExchangeRest()
    client._canonical_to_delta = {"B-BTC_USDT": "BTCUSD"}
    calls = 0

    async def fake_get(url, params=None):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.02)
        base = (now // 300) * 300 - 300
        return {"success": True, "result": [{
            "time": base, "open": 100, "high": 102,
            "low": 99, "close": 101, "volume": 10,
        }]}

    client._get = fake_get
    results = await asyncio.gather(*[
        client.get_candles("B-BTC_USDT", "5m", limit=10)
        for _ in range(8)
    ])

    assert calls == 1
    assert all(result == results[0] for result in results)
    assert client.candle_cache_hits >= 7
