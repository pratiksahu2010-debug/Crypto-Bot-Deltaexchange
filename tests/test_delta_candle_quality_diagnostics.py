import pytest

from exchange.delta_rest import DeltaExchangeRest


@pytest.mark.asyncio
async def test_candle_feed_reports_gaps_duplicates_future_and_invalid_rows(monkeypatch):
    now = 1_800_000_000 + 1830  # allow the newest completed 5m candle's grace period
    monkeypatch.setattr("exchange.delta_rest.time.time", lambda: now)
    client = DeltaExchangeRest()
    client._canonical_to_delta = {"B-BTC_USDT": "BTCUSD"}
    base = (now // 300) * 300 - 900

    def row(ts, close=100):
        return {
            "time": ts, "open": close, "high": close + 2,
            "low": close - 1, "close": close + 1, "volume": 10,
        }

    async def fake_get(url, params=None):
        return {"success": True, "result": [
            row(base), row(base), row(base + 600),
            row(now + 600), {"time": base + 900, "open": "bad"},
        ]}

    client._get = fake_get
    candles = await client.get_candles("B-BTC_USDT", "5m", limit=10)
    assert [c["time"] for c in candles] == [base * 1000, (base + 600) * 1000]
    status = client.get_candle_feed_status()["B-BTC_USDT"]["5m"]
    assert status["duplicate_row_count"] == 1
    assert status["missing_candle_count"] == 1
    assert status["future_row_count"] == 1
    assert status["invalid_row_count"] == 1
