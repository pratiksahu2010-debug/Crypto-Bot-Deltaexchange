from exchange.candle_builder import CandleBuilder


def _candle(ts, close):
    return {
        "time": ts, "open": close, "high": close + 1,
        "low": close - 1, "close": close, "volume": 10,
    }


def test_completed_arrays_excludes_forming_websocket_candle():
    builder = CandleBuilder()
    builder.seed("B-BTC_USDT", "5m", [
        _candle(0, 100), _candle(300_000, 101), _candle(600_000, 102),
    ])

    completed = builder.get_completed_arrays("B-BTC_USDT", "5m", now_ms=900_000)

    assert completed["time"].tolist() == [0, 300_000]
    assert completed["close"].tolist() == [100.0, 101.0]


def test_rest_seed_preserves_newer_websocket_forming_candle():
    builder = CandleBuilder()
    builder.seed("B-BTC_USDT", "5m", [_candle(0, 100), _candle(300_000, 101)])
    builder.on_candle("B-BTC_USDT", "5m", _candle(600_000, 103))

    # A REST backfill may finish after the newer WS candle has arrived.
    builder.seed("B-BTC_USDT", "5m", [_candle(0, 100), _candle(300_000, 101)])

    arrays = builder.get_arrays("B-BTC_USDT", "5m")
    assert arrays["time"].tolist() == [0, 300_000, 600_000]
    assert arrays["close"][-1] == 103.0


def test_completed_arrays_waits_for_exchange_close_grace():
    builder = CandleBuilder()
    builder.seed("B-BTC_USDT", "5m", [_candle(300_000, 101)])

    # Candle ends at 600,000 ms; configured 3-second grace has not elapsed.
    assert builder.get_completed_arrays(
        "B-BTC_USDT", "5m", now_ms=602_999
    ) == {}

    completed = builder.get_completed_arrays(
        "B-BTC_USDT", "5m", now_ms=603_000
    )
    assert completed["time"].tolist() == [300_000]
