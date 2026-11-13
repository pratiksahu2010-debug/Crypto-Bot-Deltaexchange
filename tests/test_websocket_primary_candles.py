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
    stream._handle_candlestick({**base, "ts": (now_ms + 1_000) * 1000, "c": 102})

    assert len(builder.rows) == 1
    assert builder.rows[0]["close"] == 102
    assert stream.candle_event_quality["B-BTC_USDT"] == {
        "out_of_order": 1, "duplicate": 1,
    }



def test_websocket_requires_ack_for_every_subscribed_symbol():
    from exchange.delta_ws import DeltaFuturesStream

    stream = DeltaFuturesStream(object(), object())
    stream._channels = {"candlestick_5m": ["BTCUSD", "ETHUSD"]}

    stream._handle_message({
        "type": "subscriptions",
        "channels": [{
            "name": "candlestick_5m",
            "symbols": ["BTCUSD"],
        }],
    })
    assert not stream.connected
    assert "ETHUSD" in stream._subscription_error

    stream._subscription_error = None
    stream._handle_message({
        "type": "subscriptions",
        "channels": [{
            "name": "candlestick_5m",
            "symbols": ["BTCUSD", "ETHUSD"],
        }],
    })
    assert stream.connected
    assert stream._subscribed_channels == {"candlestick_5m"}



def test_market_data_warmup_counts_completed_bars_not_forming_bar(monkeypatch):
    import asyncio
    import time
    from config import settings
    from exchange.market_data import MarketData

    monkeypatch.setattr(settings, "candle_close_grace_seconds", 0)
    duration = 300_000
    current_open = int(time.time() * 1000) // duration * duration
    builder = CandleBuilder()
    initial = [
        _candle(current_open - i * duration, 100 + i / 1000)
        for i in range(299, 0, -1)
    ]
    builder.seed("B-BTC_USDT", "5m", initial)
    builder.on_candle("B-BTC_USDT", "5m", _candle(current_open, 101))

    class Rest:
        async def get_candles(self, pair, interval, limit):
            assert limit == 320
            return [
                _candle(current_open - i * duration, 100 + i / 1000)
                for i in range(300, 0, -1)
            ]

    market_data = MarketData(Rest(), builder)
    assert asyncio.run(market_data.ensure_buffer("B-BTC_USDT", "5m"))
    completed = builder.get_completed_arrays("B-BTC_USDT", "5m")
    assert len(completed["close"]) == 300


def test_partial_rest_refresh_replaces_ws_close_and_preserves_history_and_newer_forming_bar():
    builder = CandleBuilder()
    duration = 300_000
    builder.seed("B-BTC_USDT", "5m", [
        _candle(i * duration, 100 + i) for i in range(4)
    ])
    # The current candle is newer than the short REST refresh window.
    builder.on_candle("B-BTC_USDT", "5m", _candle(4 * duration, 200))

    # REST finalizes the two most recent completed candles. Its overlapping
    # values must replace the earlier WS snapshots without dropping old bars.
    builder.seed("B-BTC_USDT", "5m", [
        _candle(2 * duration, 102.25),
        _candle(3 * duration, 103.25),
    ], preserve_history=True)

    arrays = builder.get_arrays("B-BTC_USDT", "5m")
    assert arrays["time"].tolist() == [0, duration, 2 * duration, 3 * duration, 4 * duration]
    assert arrays["close"].tolist() == [100.0, 101.0, 102.25, 103.25, 200.0]
