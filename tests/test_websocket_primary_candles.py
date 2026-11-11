from exchange.candle_builder import CandleBuilder


def _candle(ts, close):
    return {
        "time": ts, "open": close, "high": close + 1,
        "low": close - 1, "close": close, "volume": 10,
    }


def test_completed_arrays_excludes_forming_websocket_candle(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "candle_close_grace_seconds", 3)
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


def test_completed_arrays_waits_for_exchange_close_grace(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "candle_close_grace_seconds", 3)
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



def test_300_completed_candles_survive_a_forming_websocket_candle(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "max_candles_buffer", 300)
    monkeypatch.setattr(settings, "candle_close_grace_seconds", 3)
    builder = CandleBuilder()
    duration = 300_000
    history = [_candle(i * duration, 100 + i / 1000) for i in range(300)]
    builder.seed("B-BTC_USDT", "5m", history)
    # The active candle must coexist with all 300 completed history bars.
    builder.on_candle("B-BTC_USDT", "5m", _candle(300 * duration, 101))

    completed = builder.get_completed_arrays(
        "B-BTC_USDT", "5m", now_ms=300 * duration + 3_000
    )
    assert len(builder.get_buffer("B-BTC_USDT", "5m")) == 301
    assert len(completed["close"]) == 300


def test_websocket_ignores_older_event_for_same_forming_candle(monkeypatch):
    import time
    from exchange.delta_rest import DeltaExchangeRest
    from exchange.delta_ws import DeltaFuturesStream

    now_ms = int(time.time() * 1000)
    monkeypatch.setattr("exchange.delta_ws.time.time", lambda: now_ms / 1000)

    class Builder:
        def __init__(self):
            self.rows = []

        def on_candle(self, pair, interval, candle):
            self.rows.append(candle)

        def get_buffer(self, pair, interval):
            return type("Buffer", (), {"candles": []})()

    rest = DeltaExchangeRest()
    rest._delta_to_canonical = {"BTCUSD": "B-BTC_USDT"}
    rest._canonical_to_delta = {"B-BTC_USDT": "BTCUSD"}
    builder = Builder()
    stream = DeltaFuturesStream(builder, rest)
    base = {
        "type": "candlestick_5m", "res": "5m", "sy": "BTCUSD",
        "o": 100, "h": 103, "l": 99, "v": 10,
    }
    stream._handle_candlestick({**base, "ts": (now_ms + 1_000) * 1000, "c": 102})
    stream._handle_candlestick({**base, "ts": (now_ms + 500) * 1000, "c": 101})

    assert len(builder.rows) == 1
    assert builder.rows[0]["close"] == 102
