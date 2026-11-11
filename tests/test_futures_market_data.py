import pytest

from exchange.coindcx_rest import CoinDCXRest


@pytest.mark.asyncio
async def test_futures_15m_candles_are_aggregated_from_three_completed_5m_bars(monkeypatch):
    base = 1_800_000_000
    monkeypatch.setattr("exchange.coindcx_rest.time.time", lambda: base + 1200)
    client = CoinDCXRest()

    async def fake_get(url, params=None):
        assert url.endswith("/market_data/candlesticks")
        assert params["resolution"] == "5"
        assert params["pcode"] == "f"
        return [
            {"time": base, "open": 100, "high": 102, "low": 99, "close": 101, "volume": 10},
            {"time": base + 300, "open": 101, "high": 103, "low": 100, "close": 102, "volume": 20},
            {"time": base + 600, "open": 102, "high": 104, "low": 101, "close": 103, "volume": 30},
        ]

    client._get = fake_get
    candles = await client.get_candles("B-BTC_USDT", "15m", limit=10)

    assert len(candles) == 1
    assert candles[0] == {
        "time": base * 1000, "open": 100.0, "high": 104.0,
        "low": 99.0, "close": 103.0, "volume": 60.0,
    }



@pytest.mark.asyncio
async def test_candle_cache_expires_when_a_new_candle_becomes_completed(monkeypatch):
    base = 1_800_000_000
    now = [base + 302]
    monkeypatch.setattr("exchange.coindcx_rest.time.time", lambda: now[0])
    client = CoinDCXRest()
    calls = 0

    async def fake_get(url, params=None):
        nonlocal calls
        calls += 1
        return [
            {"time": base - 300, "open": 99, "high": 100, "low": 98, "close": 99, "volume": 1},
            {"time": base, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 2},
        ]

    client._get = fake_get
    before_close = await client.get_candles("B-BTC_USDT", "5m", limit=10)
    assert [candle["time"] for candle in before_close] == [(base - 300) * 1000]

    # Cross the 5m close plus the configured 3s grace while remaining
    # inside the ordinary five-second cache TTL.
    now[0] = base + 303
    after_close = await client.get_candles("B-BTC_USDT", "5m", limit=10)

    assert [candle["time"] for candle in after_close] == [
        (base - 300) * 1000, base * 1000
    ]
    assert calls == 2


@pytest.mark.asyncio
async def test_futures_15m_candles_must_fit_entire_requested_window(monkeypatch):
    base = 1_800_000_000
    now = base + 1200
    start_ms = base * 1000
    # The three 5m bars are present, but the requested end cuts off the
    # final 5m bar before the synthesized 15m candle has closed.
    end_ms = (base + 899) * 1000
    monkeypatch.setattr("exchange.coindcx_rest.time.time", lambda: now)
    client = CoinDCXRest()

    async def fake_get(url, params=None):
        return [
            {"time": base, "open": 100, "high": 102, "low": 99, "close": 101, "volume": 10},
            {"time": base + 300, "open": 101, "high": 103, "low": 100, "close": 102, "volume": 20},
            {"time": base + 600, "open": 102, "high": 104, "low": 101, "close": 103, "volume": 30},
        ]

    client._get = fake_get
    candles = await client.get_candles(
        "B-BTC_USDT", "15m", limit=10,
        start_time=start_ms, end_time=end_ms,
    )

    assert candles == []


@pytest.mark.asyncio
async def test_futures_candles_exclude_rows_outside_requested_time_window(monkeypatch):
    base = 1_800_000_000
    now = base + 920
    start_ms = base * 1000
    end_ms = (base + 900) * 1000
    monkeypatch.setattr("exchange.coindcx_rest.time.time", lambda: now)
    client = CoinDCXRest()

    async def fake_get(url, params=None):
        assert params["from"] == base
        assert params["to"] == base + 900
        return [
            {"time": base - 300, "open": 99, "high": 100, "low": 98, "close": 99, "volume": 1},
            {"time": base, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 2},
            {"time": base + 300, "open": 100, "high": 102, "low": 99, "close": 101, "volume": 3},
            {"time": base + 600, "open": 101, "high": 103, "low": 100, "close": 102, "volume": 4},
            {"time": base + 900, "open": 102, "high": 104, "low": 101, "close": 103, "volume": 5},
        ]

    client._get = fake_get
    candles = await client.get_candles(
        "B-BTC_USDT", "5m", limit=10,
        start_time=start_ms, end_time=end_ms,
    )

    assert [candle["time"] for candle in candles] == [
        start_ms, start_ms + 300_000, start_ms + 600_000,
    ]


def test_futures_ws_prices_ignore_out_of_order_market_updates(monkeypatch):
    client = CoinDCXRest()
    now_ms = 1_800_000_000_000
    monkeypatch.setattr("exchange.coindcx_rest.time.time", lambda: now_ms / 1000)

    assert client.set_ws_prices(
        {"B-BTC_USDT": {"ls": "65000", "ts": now_ms - 1_000}}, now_ms
    ) == 1
    # A delayed event is still within the freshness window, but older than
    # the cached exchange timestamp and must not replace the newer quote.
    assert client.set_ws_prices(
        {"B-BTC_USDT": {"ls": "64900", "ts": now_ms - 2_000}}, now_ms
    ) == 0

    assert client._ws_prices["BTCUSDT"] == (65000.0, now_ms - 1_000)


def test_futures_ws_price_cache_expires_stale_entries_and_enforces_size_bound(monkeypatch):
    client = CoinDCXRest()
    now_ms = 1_800_000_000_000
    monkeypatch.setattr("exchange.coindcx_rest.time.time", lambda: now_ms / 1000)

    client._ws_prices["STALEUSDT"] = (1.0, now_ms - 6_000)
    for index in range(2_002):
        client._ws_prices[f"M{index}USDT"] = (
            float(index + 1), now_ms - (2_002 - index)
        )

    client._prune_ws_price_cache(now_ms, force=True)

    assert "STALEUSDT" not in client._ws_prices
    assert len(client._ws_prices) == client.MAX_WS_PRICE_CACHE_ENTRIES
    # The two oldest still-fresh entries are evicted; newest quotes survive.
    assert "M0USDT" not in client._ws_prices
    assert "M1USDT" not in client._ws_prices
    assert "M2USDT" in client._ws_prices
    assert "M2001USDT" in client._ws_prices


def test_futures_ws_price_parser_uses_last_trade_not_mark_price(monkeypatch):
    client = CoinDCXRest()
    now_ms = 1_800_000_000_000
    monkeypatch.setattr("exchange.coindcx_rest.time.time", lambda: now_ms / 1000)

    client.set_ws_prices({
        "B-BTC_USDT": {"ls": "65000.5", "mp": "65100.0", "ts": now_ms},
        "B-ETH_USDT": {"mp": "3000.0", "ts": now_ms},
    }, now_ms)

    assert client._ws_prices["BTCUSDT"][0] == 65000.5
    assert "ETHUSDT" not in client._ws_prices


@pytest.mark.asyncio
async def test_futures_price_endpoint_normalizes_nested_payload_and_ltp(monkeypatch):
    client = CoinDCXRest()
    now_ms = 1_800_000_000_000
    monkeypatch.setattr("exchange.coindcx_rest.time.time", lambda: now_ms / 1000)

    async def fake_get(url, params=None):
        assert url.endswith("/market_data/v3/current_prices/futures/rt")
        return {"ts": now_ms, "prices": {"B-BTC_USDT": {"ls": "65000", "mp": "65100"}}}

    client._get = fake_get
    rows = await client.get_ticker()
    assert rows[0]["market"] == "B-BTC_USDT"
    assert rows[0]["last_price"] == "65000"
    assert await client.get_last_price("B-BTC_USDT") == 65000.0


@pytest.mark.asyncio
async def test_active_futures_instruments_string_list_is_normalized(monkeypatch):
    client = CoinDCXRest()

    async def fake_get(url, params=None):
        assert url.endswith("/exchange/v1/derivatives/futures/data/active_instruments")
        assert params == {"margin_currency_short_name[]": "USDT"}
        return ["B-BTC_USDT", "B-ETH_USDT", "B-BTC_USDT", "B-USDC_USDT", "B-INVALID_INR"]

    client._get = fake_get
    markets = await client.get_market_details()

    assert [row["pair"] for row in markets] == ["B-BTC_USDT", "B-ETH_USDT", "B-USDC_USDT"]
    assert markets[0]["target_currency_short_name"] == "BTC"
    assert markets[0]["base_currency_short_name"] == "USDT"
    assert all(row["status"] == "active" for row in markets)


@pytest.mark.asyncio
async def test_active_futures_instruments_accept_wrapped_and_metadata_shapes():
    client = CoinDCXRest()
    payload = {
        "data": [
            {"pair": "B-SOL_USDT", "status": "active", "margin_currency_short_name": "USDT",
             "target_currency_short_name": "SOL"},
            {"pair": "B-XRP_USDT", "status": "inactive", "margin_currency_short_name": "USDT"},
            {"pair": "B-ADA_INR", "status": "active", "margin_currency_short_name": "INR"},
        ]
    }

    async def fake_get(url, params=None):
        return payload

    client._get = fake_get
    markets = await client.get_market_details()
    assert [row["pair"] for row in markets] == ["B-SOL_USDT", "B-XRP_USDT"]
    assert markets[1]["status"] == "inactive"


@pytest.mark.asyncio
async def test_futures_price_endpoint_normalizes_list_payload_and_volume(monkeypatch):
    client = CoinDCXRest()
    now_ms = 1_800_000_000_000
    monkeypatch.setattr("exchange.coindcx_rest.time.time", lambda: now_ms / 1000)

    async def fake_get(url, params=None):
        return [
            {"market": "B-BTC_USDT", "ls": "65000", "mp": "65100", "ts": now_ms, "v": "12.5"},
            {"pair": "B-ETH_USDT", "lastPrice": "3000", "timestamp": now_ms, "volume": "20"},
        ]

    client._get = fake_get
    rows = await client.get_ticker()
    assert len(rows) == 2
    assert rows[0]["last_price"] == "65000"
    assert rows[0]["base_volume"] == "12.5"
    assert await client.get_last_price("B-BTC_USDT") == 65000.0
    assert await client.get_last_price("B-ETH_USDT") == 3000.0


@pytest.mark.asyncio
async def test_futures_price_endpoint_uses_root_timestamp_and_v_volume(monkeypatch):
    client = CoinDCXRest()
    now_ms = 1_800_000_000_000
    monkeypatch.setattr("exchange.coindcx_rest.time.time", lambda: now_ms / 1000)

    async def fake_get(url, params=None):
        return {"ts": now_ms, "prices": {"B-SOL_USDT": {"ls": "150", "v": "1000"}}}

    client._get = fake_get
    rows = await client.get_ticker()
    assert rows[0]["timestamp"] == now_ms
    assert rows[0]["base_volume"] == "1000"
    assert await client.get_last_price("B-SOL_USDT") == 150.0


@pytest.mark.asyncio
async def test_futures_price_missing_timestamp_fails_closed(monkeypatch):
    client = CoinDCXRest()
    now_ms = 1_800_000_000_000
    monkeypatch.setattr("exchange.coindcx_rest.time.time", lambda: now_ms / 1000)

    async def fake_get(url, params=None):
        return {"prices": {"B-BTC_USDT": {"ls": "65000"}}}

    client._get = fake_get
    assert await client.get_last_price("B-BTC_USDT") is None


def test_futures_ws_price_handler_accepts_documented_event_payload(monkeypatch):
    from exchange.coindcx_ws import CoinDCXFuturesStream

    class CandleStub:
        def on_candle(self, *args, **kwargs):
            pass

    rest = CoinDCXRest()
    now_ms = 1_800_000_000_000
    monkeypatch.setattr("exchange.coindcx_rest.time.time", lambda: now_ms / 1000)
    stream = CoinDCXFuturesStream(CandleStub(), rest)
    stream._handle_prices({"data": {
        "ts": now_ms,
        "prices": {"B-BTC_USDT": {"ls": "65000", "mp": "65100", "ts": now_ms}}
    }})
    assert rest._ws_prices["BTCUSDT"][0] == 65000.0



@pytest.mark.parametrize("row_ts,expected_source", [
    (1_800_000_000_000, "row"),
    (1_800_000_000, "row"),
])
@pytest.mark.asyncio
async def test_futures_ticker_row_timestamp_is_normalized_and_used(monkeypatch, row_ts, expected_source):
    client = CoinDCXRest()
    now_ms = 1_800_000_000_000
    monkeypatch.setattr("exchange.coindcx_rest.time.time", lambda: now_ms / 1000)

    async def fake_get(url, params=None):
        return {
            "ts": now_ms - 60_000,
            "prices": {"B-BTC_USDT": {"ls": "65000", "ts": row_ts}},
        }

    client._get = fake_get
    rows = await client.get_ticker()
    assert rows[0]["timestamp_source"] == expected_source
    assert await client.get_last_price("B-BTC_USDT") == 65000.0


@pytest.mark.parametrize("root_ts", [1_800_000_000_000, 1_800_000_000])
@pytest.mark.asyncio
async def test_futures_ticker_uses_root_timestamp_when_row_timestamp_absent(monkeypatch, root_ts):
    client = CoinDCXRest()
    now_ms = 1_800_000_000_000
    monkeypatch.setattr("exchange.coindcx_rest.time.time", lambda: now_ms / 1000)

    async def fake_get(url, params=None):
        return {"ts": root_ts, "prices": {"B-ETH_USDT": {"ls": "3000"}}}

    client._get = fake_get
    rows = await client.get_ticker()
    assert rows[0]["timestamp_source"] == "root"
    assert await client.get_last_price("B-ETH_USDT") == 3000.0


@pytest.mark.asyncio
async def test_futures_ticker_rejects_stale_row_even_when_root_timestamp_is_fresh(monkeypatch):
    client = CoinDCXRest()
    now_ms = 1_800_000_000_000
    monkeypatch.setattr("exchange.coindcx_rest.time.time", lambda: now_ms / 1000)

    async def fake_get(url, params=None):
        return {
            "ts": now_ms,
            "prices": {"B-ETH_USDT": {"ls": "3000", "ts": now_ms - 30_000}},
        }

    client._get = fake_get
    rows = await client.get_ticker()
    assert await client.get_last_price("B-ETH_USDT") is None
    assert rows[0]["timestamp_source"] == "row"


@pytest.mark.asyncio
async def test_futures_ticker_snapshot_and_symbol_lookup_share_freshness_rules(monkeypatch):
    client = CoinDCXRest()
    now_ms = 1_800_000_000_000
    monkeypatch.setattr("exchange.coindcx_rest.time.time", lambda: now_ms / 1000)

    async def fake_get(url, params=None):
        return {
            "ts": now_ms,
            "prices": {
                "B-BTC_USDT": {"ls": "65000"},
                "B-ETH_USDT": {"ls": "3000", "ts": now_ms - 30_000},
            },
        }

    client._get = fake_get
    rows = await client.get_ticker()
    ages = [client._ticker_age_seconds(row["timestamp"], now_ms) for row in rows]
    assert sum(age is not None and -5 <= age <= 10 for age in ages) == 1
    assert await client.get_last_price("B-BTC_USDT") == 65000.0
    assert await client.get_last_price("B-ETH_USDT") is None


@pytest.mark.asyncio
async def test_futures_ticker_accepts_bounded_six_second_future_clock_skew(monkeypatch):
    client = CoinDCXRest()
    now_ms = 1_800_000_000_000
    monkeypatch.setattr("exchange.coindcx_rest.time.time", lambda: now_ms / 1000)

    async def fake_get(url, params=None):
        return {"ts": now_ms + 6_000, "prices": {"B-ETH_USDT": {"ls": "3000"}}}

    client._get = fake_get
    rows = await client.get_ticker()
    age = client._ticker_age_seconds(rows[0]["timestamp"], now_ms)
    assert age == -6.0
    assert client._is_fresh_ticker_age(age, client.MAX_REST_TICKER_AGE_SECONDS)
    assert await client.get_last_price("B-ETH_USDT") == 3000.0


@pytest.mark.asyncio
async def test_futures_ticker_rejects_timestamp_beyond_future_skew_limit(monkeypatch):
    client = CoinDCXRest()
    now_ms = 1_800_000_000_000
    monkeypatch.setattr("exchange.coindcx_rest.time.time", lambda: now_ms / 1000)

    async def fake_get(url, params=None):
        return {"ts": now_ms + 16_000, "prices": {"B-ETH_USDT": {"ls": "3000"}}}

    client._get = fake_get
    assert await client.get_last_price("B-ETH_USDT") is None


def test_futures_ws_accepts_bounded_future_timestamp_skew(monkeypatch):
    client = CoinDCXRest()
    now_ms = 1_800_000_000_000
    monkeypatch.setattr("exchange.coindcx_rest.time.time", lambda: now_ms / 1000)
    accepted = client.set_ws_prices(
        {"B-BTC_USDT": {"ls": "65000", "ts": now_ms + 6_000}},
        now_ms + 6_000,
    )
    assert accepted == 1


@pytest.mark.asyncio
async def test_futures_ws_resubscribes_all_channels_after_reconnect(monkeypatch):
    from exchange.coindcx_ws import CoinDCXFuturesStream

    class CandleStub:
        def on_candle(self, *args, **kwargs):
            pass

    class SocketStub:
        def __init__(self):
            self.connected = False
            self.emitted = []
            self.handlers = {}

        def event(self, handler):
            self.handlers[handler.__name__] = handler
            return handler

        def on(self, event_name):
            def register(handler):
                self.handlers[event_name] = handler
                return handler
            return register

        async def connect(self, url, transports=None):
            self.connected = True
            self.connect_args = {"url": url, "transports": transports}
            await self.handlers["connect"]()

        async def emit(self, event, payload):
            self.emitted.append((event, payload))

        async def disconnect(self):
            self.connected = False
            await self.handlers["disconnect"]()

    socket = SocketStub()
    monkeypatch.setattr("exchange.coindcx_ws.socketio.AsyncClient", lambda **kwargs: socket)
    monkeypatch.setattr("exchange.coindcx_ws.settings.enable_websocket", True)
    stream = CoinDCXFuturesStream(CandleStub(), CoinDCXRest())

    started = await stream.start(["B-BTC_USDT", "B-ETH_USDT"])

    assert started is True
    assert socket.connect_args["transports"] == ["websocket"]
    expected = {
        "currentPrices@futures@rt",
        "B-BTC_USDT_5m-futures", "B-BTC_USDT_15m-futures",
        "B-ETH_USDT_5m-futures", "B-ETH_USDT_15m-futures",
    }
    first_joins = [payload["channelName"] for event, payload in socket.emitted if event == "join"]
    assert set(first_joins) == expected
    assert len(first_joins) == len(expected)
    assert stream._subscribed_channels == expected

    # Simulate an unexpected disconnect and the automatic Socket.IO reconnect.
    await socket.handlers["disconnect"]()
    assert stream._subscribed_channels == set()
    assert stream.reconnect_count == 1
    socket.connected = True
    await socket.handlers["connect"]()

    all_joins = [payload["channelName"] for event, payload in socket.emitted if event == "join"]
    assert all_joins.count("currentPrices@futures@rt") == 2
    assert set(all_joins) == expected
    assert len(all_joins) == len(expected) * 2
    assert stream._subscribed_channels == expected


@pytest.mark.asyncio
async def test_futures_ticker_handles_nested_data_prices_envelope(monkeypatch):
    client = CoinDCXRest()
    now_ms = 1_800_000_000_000
    monkeypatch.setattr("exchange.coindcx_rest.time.time", lambda: now_ms / 1000)

    async def fake_get(url, params=None):
        return {"data": {"ts": now_ms, "prices": {
            "B-BTC_USDT": {"ls": "65000"},
        }}}

    client._get = fake_get
    rows = await client.get_ticker()
    assert rows[0]["market"] == "B-BTC_USDT"
    assert rows[0]["timestamp"] == now_ms
    assert await client.get_last_price("B-BTC_USDT") == 65000.0


@pytest.mark.asyncio
async def test_futures_candles_handle_nested_data_candles_envelope(monkeypatch):
    base = 1_800_000_000
    monkeypatch.setattr("exchange.coindcx_rest.time.time", lambda: base + 600)
    client = CoinDCXRest()

    async def fake_get(url, params=None):
        return {"data": {"candles": [
            {"time": base, "open": 100, "high": 102, "low": 99, "close": 101, "volume": 10},
        ]}}

    client._get = fake_get
    candles = await client.get_candles("B-BTC_USDT", "5m", limit=10)
    assert len(candles) == 1
    assert candles[0]["close"] == 101.0


def test_futures_ws_candle_parser_recovers_full_pair_from_channel_and_normalizes_event_time(monkeypatch):
    from exchange.coindcx_ws import CoinDCXFuturesStream

    class CandleStub:
        def __init__(self):
            self.rows = []

        def on_candle(self, pair, interval, candle):
            self.rows.append((pair, interval, candle))

    builder = CandleStub()
    stream = CoinDCXFuturesStream(builder, CoinDCXRest())
    base = 1_800_000_000
    monkeypatch.setattr("exchange.coindcx_ws.time.time", lambda: base + 120)
    stream._handle_candlestick({"data": {
        "channel": "B-BTC_USDT_5m-futures",
        "open_time": base,
        "open": 100, "high": 102, "low": 99, "close": 101, "volume": 10,
        "Ets": base + 300,
    }})
    assert len(builder.rows) == 1
    pair, interval, candle = builder.rows[0]
    assert pair == "B-BTC_USDT"
    assert interval == "5m"
    assert candle["time"] == base * 1000
    assert stream._last_event_ms == (base + 120) * 1000


def test_futures_ws_candle_parser_reads_channel_from_outer_event_envelope(monkeypatch):
    from exchange.coindcx_ws import CoinDCXFuturesStream

    class CandleStub:
        def __init__(self):
            self.rows = []

        def on_candle(self, pair, interval, candle):
            self.rows.append((pair, interval, candle))

    builder = CandleStub()
    monkeypatch.setattr("exchange.coindcx_ws.time.time", lambda: 1_800_000_120)
    stream = CoinDCXFuturesStream(builder, CoinDCXRest())
    stream._handle_candlestick({
        "channel": "candlestick@B-ETH_USDT_5m-futures",
        "data": {
            "open_time": 1_800_000_000,
            "open": 100, "high": 102, "low": 99, "close": 101, "volume": 10,
        },
    })
    assert len(builder.rows) == 1
    assert builder.rows[0][0:2] == ("B-ETH_USDT", "5m")


@pytest.mark.parametrize("limit", [0, -1, 1.5, 1001, "invalid"])
@pytest.mark.asyncio
async def test_futures_candle_request_rejects_invalid_limit_before_network(monkeypatch, limit):
    client = CoinDCXRest()

    async def unexpected_get(*args, **kwargs):
        raise AssertionError("invalid candle request must not reach the exchange")

    client._get = unexpected_get
    with pytest.raises(ValueError, match="Candle limit"):
        await client.get_candles("B-BTC_USDT", "15m", limit=limit)


@pytest.mark.parametrize(
    "start_time,end_time",
    [(1_800_000_000_000, 1_799_999_999_000), (0, 1_800_000_000_000)],
)
@pytest.mark.asyncio
async def test_futures_candle_request_rejects_invalid_time_range(monkeypatch, start_time, end_time):
    client = CoinDCXRest()

    async def unexpected_get(*args, **kwargs):
        raise AssertionError("invalid candle request must not reach the exchange")

    client._get = unexpected_get
    with pytest.raises(ValueError, match="start_time"):
        await client.get_candles(
            "B-BTC_USDT", "5m", limit=100,
            start_time=start_time, end_time=end_time,
        )


@pytest.mark.asyncio
async def test_futures_ws_subscribes_to_configured_timeframes(monkeypatch):
    from exchange.coindcx_ws import CoinDCXFuturesStream

    class CandleStub:
        def on_candle(self, *args, **kwargs):
            pass

    class SocketStub:
        def __init__(self):
            self.connected = False
            self.emitted = []
            self.handlers = {}
        def event(self, handler):
            self.handlers[handler.__name__] = handler
            return handler
        def on(self, event_name):
            def register(handler):
                self.handlers[event_name] = handler
                return handler
            return register
        async def connect(self, url, transports=None):
            self.connected = True
            await self.handlers["connect"]()
        async def emit(self, event, payload):
            self.emitted.append((event, payload))
        async def disconnect(self):
            self.connected = False
            await self.handlers["disconnect"]()

    socket = SocketStub()
    monkeypatch.setattr("exchange.coindcx_ws.socketio.AsyncClient", lambda **kwargs: socket)
    monkeypatch.setattr("exchange.coindcx_ws.settings.enable_websocket", True)
    monkeypatch.setattr("exchange.coindcx_ws.settings.candle_primary_tf", "1m")
    monkeypatch.setattr("exchange.coindcx_ws.settings.candle_confirm_tf", "1h")
    stream = CoinDCXFuturesStream(CandleStub(), CoinDCXRest())
    assert await stream.start(["B-BTC_USDT"])
    joined = {payload["channelName"] for event, payload in socket.emitted if event == "join"}
    assert joined == {
        "currentPrices@futures@rt",
        "B-BTC_USDT_1m-futures",
        "B-BTC_USDT_1h-futures",
    }


def test_futures_ws_candle_parser_infers_configured_interval_from_channel(monkeypatch):
    from exchange.coindcx_ws import CoinDCXFuturesStream

    class CandleStub:
        def __init__(self):
            self.rows = []
        def on_candle(self, pair, interval, candle):
            self.rows.append((pair, interval, candle))

    builder = CandleStub()
    monkeypatch.setattr("exchange.coindcx_ws.time.time", lambda: 1_800_000_120)
    stream = CoinDCXFuturesStream(builder, CoinDCXRest())
    stream._handle_candlestick({
        "channel": "candlestick@B-BTC_USDT_1h-futures",
        "data": {
            "open_time": 1_800_000_000,
            "open": 100, "high": 102, "low": 99, "close": 101, "volume": 10,
        },
    })
    assert len(builder.rows) == 1
    assert builder.rows[0][0:2] == ("B-BTC_USDT", "1h")



@pytest.mark.asyncio
async def test_futures_orderbook_normalizes_top_of_book_and_spread(monkeypatch):
    client = CoinDCXRest()
    now_ms = 1_800_000_000_000
    monkeypatch.setattr("exchange.coindcx_rest.time.time", lambda: now_ms / 1000)

    async def fake_get(url, params=None):
        assert url.endswith("/market_data/v3/orderbook/B-BTC_USDT-futures/10")
        return {
            "ts": now_ms,
            "bids": {"100.0": "2.0", "99.5": "4.0"},
            "asks": {"100.2": "3.0", "101.0": "5.0"},
        }

    client._get = fake_get
    book = await client.get_futures_orderbook("B-BTC_USDT")
    assert book["best_bid"] == 100.0
    assert book["best_ask"] == 100.2
    assert book["spread_bps"] == pytest.approx(19.98001998)
    assert book["age_seconds"] == 0.0
    assert book["bid_depth_usdt"] == pytest.approx(598.0)
    assert book["ask_depth_usdt"] == pytest.approx(805.6)
    assert book["min_side_depth_usdt"] == pytest.approx(598.0)
    assert book["depth_levels"] == 5


@pytest.mark.asyncio
async def test_futures_orderbook_rejects_stale_or_crossed_book(monkeypatch):
    client = CoinDCXRest()
    now_ms = 1_800_000_000_000
    monkeypatch.setattr("exchange.coindcx_rest.time.time", lambda: now_ms / 1000)

    async def stale_get(url, params=None):
        return {
            "ts": now_ms - 10_000,
            "bids": {"100": "1"},
            "asks": {"101": "1"},
        }

    client._get = stale_get
    with pytest.raises(ValueError, match="stale"):
        await client.get_futures_orderbook("B-BTC_USDT")

    async def crossed_get(url, params=None):
        return {
            "ts": now_ms,
            "bids": {"101": "1"},
            "asks": {"100": "1"},
        }

    client._get = crossed_get
    with pytest.raises(ValueError, match="crossed"):
        await client.get_futures_orderbook("B-BTC_USDT")


@pytest.mark.asyncio
async def test_futures_orderbook_rejects_missing_two_sided_liquidity(monkeypatch):
    client = CoinDCXRest()
    now_ms = 1_800_000_000_000
    monkeypatch.setattr("exchange.coindcx_rest.time.time", lambda: now_ms / 1000)

    async def fake_get(url, params=None):
        return {"ts": now_ms, "bids": {"100": "1"}, "asks": {}}

    client._get = fake_get
    with pytest.raises(ValueError, match="two-sided"):
        await client.get_futures_orderbook("B-BTC_USDT")


def test_orderbook_spread_gate_is_fail_closed_and_respects_configured_limit():
    from core.market_quality import evaluate_orderbook_spread

    assert evaluate_orderbook_spread({"spread_bps": 12.5}, 20.0) == (True, 12.5, "ok")
    assert evaluate_orderbook_spread({"spread_bps": 25.0}, 20.0) == (False, 25.0, "spread_too_wide")
    assert evaluate_orderbook_spread({}, 20.0) == (False, None, "orderbook_unavailable")
    assert evaluate_orderbook_spread({"spread_bps": float("nan")}, 20.0)[0] is False
    assert evaluate_orderbook_spread({"spread_bps": 5.0}, 0.0)[0] is False



def test_orderbook_depth_gate_is_optional_and_enforces_both_sides():
    from core.market_quality import evaluate_orderbook_depth

    snapshot = {"bid_depth_usdt": 2500.0, "ask_depth_usdt": 1800.0}
    assert evaluate_orderbook_depth(snapshot, 1000.0) == (True, 1800.0, "ok")
    assert evaluate_orderbook_depth(snapshot, 1800.0) == (True, 1800.0, "ok")
    assert evaluate_orderbook_depth(snapshot, 1800.01) == (False, 1800.0, "insufficient_depth")
    assert evaluate_orderbook_depth(snapshot, 0.0) == (True, 1800.0, "disabled")
    assert evaluate_orderbook_depth({}, 0.0) == (True, None, "disabled")
    assert evaluate_orderbook_depth({}, 1000.0) == (False, None, "orderbook_depth_unavailable")
    assert evaluate_orderbook_depth(
        {"bid_depth_usdt": float("nan"), "ask_depth_usdt": 1000}, 0
    )[0] is False


@pytest.mark.parametrize("open_time_offset", [-600, 300])
def test_futures_ws_rejects_stale_and_future_candles(monkeypatch, open_time_offset):
    from exchange.coindcx_ws import CoinDCXFuturesStream

    class CandleStub:
        def __init__(self):
            self.rows = []
        def on_candle(self, pair, interval, candle):
            self.rows.append((pair, interval, candle))

    now_s = 1_800_000_120
    monkeypatch.setattr("exchange.coindcx_ws.time.time", lambda: now_s)
    builder = CandleStub()
    stream = CoinDCXFuturesStream(builder, CoinDCXRest())
    open_s = 1_800_000_000 + open_time_offset
    stream._handle_candlestick({
        "channel": "B-BTC_USDT_5m-futures",
        "data": {
            "open_time": open_s,
            "open": 100, "high": 102, "low": 99, "close": 101, "volume": 10,
        },
    })
    assert builder.rows == []
    assert stream._last_event_ms is None


def test_futures_ws_rejects_malformed_ohlcv_before_buffer_update(monkeypatch):
    from exchange.coindcx_ws import CoinDCXFuturesStream

    class CandleStub:
        def __init__(self):
            self.rows = []
        def on_candle(self, pair, interval, candle):
            self.rows.append((pair, interval, candle))

    base = 1_800_000_000
    monkeypatch.setattr("exchange.coindcx_ws.time.time", lambda: base + 120)
    builder = CandleStub()
    stream = CoinDCXFuturesStream(builder, CoinDCXRest())
    stream._handle_candlestick({
        "channel": "B-BTC_USDT_5m-futures",
        "data": {
            "open_time": base,
            "open": 100, "high": 98, "low": 99, "close": 101, "volume": 10,
        },
    })
    assert builder.rows == []


@pytest.mark.parametrize("interval", ["30m", "4h", "8h", "3d", "1w"])
def test_futures_ws_candle_parser_infers_supported_interval_from_channel(monkeypatch, interval):
    from exchange.coindcx_ws import CoinDCXFuturesStream
    from utils.time_utils import parse_interval

    class CandleStub:
        def __init__(self):
            self.rows = []

        def on_candle(self, pair, timeframe, candle):
            self.rows.append((pair, timeframe, candle))

    now_s = 1_800_000_000
    duration_s = int(parse_interval(interval).total_seconds())
    open_s = now_s - (now_s % duration_s)
    monkeypatch.setattr("exchange.coindcx_ws.time.time", lambda: now_s)
    builder = CandleStub()
    stream = CoinDCXFuturesStream(builder, CoinDCXRest())
    stream._handle_candlestick({
        "channel": f"candlestick@B-BTC_USDT_{interval}-futures",
        "data": {
            "open_time": open_s,
            "open": 100, "high": 102, "low": 99, "close": 101, "volume": 10,
        },
    })

    assert len(builder.rows) == 1
    assert builder.rows[0][0:2] == ("B-BTC_USDT", interval)


def test_futures_ws_rejects_delayed_candle_after_interval_close(monkeypatch):
    from exchange.coindcx_ws import CoinDCXFuturesStream

    class CandleStub:
        def __init__(self):
            self.rows = []

        def on_candle(self, pair, interval, candle):
            self.rows.append((pair, interval, candle))

    base = 1_800_000_000
    # The 5m candle opened at base and is now exactly closed. Even though
    # it is inside the previous broad freshness allowance, it is no longer
    # forming-candle context and must not mutate the websocket buffer.
    monkeypatch.setattr("exchange.coindcx_ws.time.time", lambda: base + 300)
    builder = CandleStub()
    stream = CoinDCXFuturesStream(builder, CoinDCXRest())
    stream._handle_candlestick({
        "channel": "B-BTC_USDT_5m-futures",
        "data": {
            "open_time": base,
            "open": 100, "high": 102, "low": 99, "close": 101, "volume": 10,
        },
    })

    assert builder.rows == []
    assert stream._last_event_ms is None

@pytest.mark.parametrize(
    "duration,channel_interval",
    [("1M", "1M"), ("", "1M")],
)
def test_futures_ws_rejects_monthly_candles_instead_of_treating_them_as_minutes(
    monkeypatch, duration, channel_interval
):
    from exchange.coindcx_ws import CoinDCXFuturesStream

    class CandleStub:
        def __init__(self):
            self.rows = []

        def on_candle(self, pair, interval, candle):
            self.rows.append((pair, interval, candle))

    base = 1_800_000_000
    monkeypatch.setattr("exchange.coindcx_ws.time.time", lambda: base + 120)
    builder = CandleStub()
    stream = CoinDCXFuturesStream(builder, CoinDCXRest())
    row = {
        "open_time": base,
        "open": 100, "high": 102, "low": 99, "close": 101, "volume": 10,
    }
    if duration:
        row["duration"] = duration
    stream._handle_candlestick({
        "channel": f"candlestick@B-BTC_USDT_{channel_interval}-futures",
        "data": row,
    })

    assert builder.rows == []
    assert stream._last_event_ms is None




def test_futures_ws_out_of_order_candle_does_not_reach_intrabar_callback(monkeypatch):
    from exchange.candle_builder import CandleBuilder
    from exchange.coindcx_ws import CoinDCXFuturesStream

    base = 1_800_000_000_000
    # The next bucket is accepted just inside the 6-second future tolerance.
    # A delayed update for the still-forming previous bucket must not reach
    # the callback after the buffer has advanced.
    monkeypatch.setattr("exchange.coindcx_ws.time.time", lambda: (base + 299_000) / 1000)
    builder = CandleBuilder()
    builder.seed("B-BTC_USDT", "5m", [{
        "time": base + 300_000,
        "open": 101, "high": 103, "low": 100, "close": 102, "volume": 20,
    }])
    callbacks = []
    stream = CoinDCXFuturesStream(builder, CoinDCXRest())
    stream.set_candle_callback(lambda pair, interval, candle: callbacks.append(candle))

    stream._handle_candlestick({
        "channel": "candlestick@B-BTC_USDT_5m-futures",
        "data": {
            "open_time": base,
            "open": 100, "high": 102, "low": 99, "close": 101, "volume": 10,
        },
    })

    buffered = builder.get_buffer("B-BTC_USDT", "5m").candles
    assert len(buffered) == 1
    assert buffered[0]["time"] == base + 300_000
    assert callbacks == []


@pytest.mark.asyncio
async def test_empty_live_candle_response_is_tracked_and_fresh_response_clears_it(monkeypatch):
    base = 1_800_000_000
    monkeypatch.setattr("exchange.coindcx_rest.time.time", lambda: base + 600)
    client = CoinDCXRest()

    async def empty_get(url, params=None):
        return []

    client._get = empty_get
    assert await client.get_candles("B-BTC_USDT", "5m", limit=10) == []
    assert client.get_stale_market_timeframes() == {"B-BTC_USDT": {"5m": 1}}

    async def fresh_get(url, params=None):
        return [{
            "time": base, "open": 100, "high": 102,
            "low": 99, "close": 101, "volume": 10,
        }]

    client._get = fresh_get
    client._candle_cache.clear()
    candles = await client.get_candles("B-BTC_USDT", "5m", limit=10)
    assert len(candles) == 1
    assert client.get_stale_market_timeframes() == {"B-BTC_USDT": {}}


@pytest.mark.asyncio
async def test_incomplete_synthesized_15m_feed_is_tracked_as_unavailable(monkeypatch):
    base = 1_800_000_000
    monkeypatch.setattr("exchange.coindcx_rest.time.time", lambda: base + 1200)
    client = CoinDCXRest()

    async def incomplete_get(url, params=None):
        return [
            {"time": base, "open": 100, "high": 102, "low": 99, "close": 101, "volume": 10},
            {"time": base + 300, "open": 101, "high": 103, "low": 100, "close": 102, "volume": 20},
        ]

    client._get = incomplete_get
    assert await client.get_candles("B-BTC_USDT", "15m", limit=10) == []
    assert client.get_stale_market_timeframes() == {"B-BTC_USDT": {"15m": 1}}


@pytest.mark.asyncio
async def test_malformed_candle_envelope_is_cached_to_avoid_repeated_rest_calls(monkeypatch):
    base = 1_800_000_000
    monkeypatch.setattr("exchange.coindcx_rest.time.time", lambda: base + 600)
    client = CoinDCXRest()
    calls = 0

    async def malformed_get(url, params=None):
        nonlocal calls
        calls += 1
        return {"error": "temporarily unavailable"}

    client._get = malformed_get
    assert await client.get_candles("B-BTC_USDT", "5m", limit=10) == []
    assert await client.get_candles("B-BTC_USDT", "5m", limit=10) == []
    assert calls == 1
    assert client.candle_cache_hits == 1
    assert client.get_stale_market_timeframes() == {"B-BTC_USDT": {"5m": 1}}


@pytest.mark.asyncio
async def test_malformed_candle_response_keeps_cache_within_size_bound(monkeypatch):
    import time

    base = 1_800_000_000
    monkeypatch.setattr("exchange.coindcx_rest.time.time", lambda: base + 600)
    client = CoinDCXRest()
    now = time.monotonic()
    client._candle_cache = {
        ("seed", index): (now, [])
        for index in range(1001)
    }

    async def malformed_get(url, params=None):
        return {"error": "temporarily unavailable"}

    client._get = malformed_get
    assert await client.get_candles("B-BTC_USDT", "5m", limit=10) == []
    assert len(client._candle_cache) <= 1000


@pytest.mark.asyncio
async def test_futures_orderbook_cache_prunes_expired_and_bounds_unique_markets(monkeypatch):
    client = CoinDCXRest()
    now_ms = 1_800_000_000_000
    now_mono = 500.0
    monkeypatch.setattr("exchange.coindcx_rest.time.time", lambda: now_ms / 1000)
    monkeypatch.setattr("exchange.coindcx_rest.time.monotonic", lambda: now_mono)

    # Simulate many symbols whose snapshots expired and were never looked up again.
    client._orderbook_cache = {
        f"B-OLD{index}_USDT:10": (now_mono - 10, {"timestamp": now_ms})
        for index in range(1200)
    }

    async def fake_get(url, params=None):
        return {
            "ts": now_ms,
            "bids": {"100": "2", "99": "3"},
            "asks": {"101": "2", "102": "3"},
        }

    client._get = fake_get
    book = await client.get_futures_orderbook("B-BTC_USDT")
    assert book["pair"] == "B-BTC_USDT"
    assert len(client._orderbook_cache) == 1

    # A large set of still-fresh snapshots is also capped deterministically.
    client._orderbook_cache = {
        f"B-FRESH{index}_USDT:10": (now_mono - 0.5, {"timestamp": now_ms})
        for index in range(1200)
    }
    client._prune_orderbook_cache(now_mono)
    assert len(client._orderbook_cache) == 1000



@pytest.mark.parametrize(
    "pair",
    [
        "B-XAU_USDT", "B-PAXG_USDT", "B-XAG_USDT",
        "B-CL_USDT", "B-BZ_USDT", "B-NATGAS_USDT", "B-COPPER_USDT",
        "B-WTI_USDT", "B-PLATINUM_USDT", "B-WHEAT_USDT",
    ],
)
def test_crypto_universe_rejects_commodity_futures(pair):
    from exchange.symbol_manager import SymbolManager

    assert SymbolManager._is_non_crypto_pair(pair) is True


@pytest.mark.parametrize("pair", ["B-BTC_USDT", "B-ETH_USDT", "B-SOL_USDT", "B-LINK_USDT"])
def test_crypto_universe_keeps_crypto_assets(pair):
    from exchange.symbol_manager import SymbolManager

    assert SymbolManager._is_non_crypto_pair(pair) is False


@pytest.mark.asyncio
async def test_fixed_allowlist_excludes_commodities_and_never_adds_unlisted_crypto():
    from pathlib import Path
    import yaml
    from exchange.symbol_manager import SymbolManager

    root = Path(__file__).resolve().parents[1] / "config"
    normal = yaml.safe_load((root / "normal_symbols.yaml").read_text())["symbols"]
    momentum = yaml.safe_load((root / "momentum_symbols.yaml").read_text())["symbols"]
    extras = [
        "B-XAU_USDT", "B-PAXG_USDT", "B-XAG_USDT", "B-CL_USDT",
        "B-NATGAS_USDT", "B-COPPER_USDT", "B-UNLISTEDXYZ_USDT",
    ]
    configured_assets = [
        pair.removeprefix("B-").removesuffix("_USDT")
        for pair in momentum + extras
    ]

    class RestStub:
        async def get_market_details(self):
            return [
                {
                    "pair": f"B-{asset}_USDT",
                    "symbol": f"{asset}USD",
                    "base_currency_short_name": "USD",
                    "target_currency_short_name": asset,
                    "quote_currency": "USD",
                    "status": "active",
                    "contract_type": "perpetual_futures",
                }
                for asset in configured_assets
            ]

        async def get_ticker(self):
            return []

    manager = SymbolManager(RestStub())
    await manager.load_and_validate()

    assert manager.normal_symbols == normal
    assert manager.momentum_symbols == momentum
    assert "B-UNLISTEDXYZ_USDT" not in manager.momentum_symbols
    for pair in ("B-XAU_USDT", "B-PAXG_USDT", "B-XAG_USDT", "B-CL_USDT", "B-NATGAS_USDT", "B-COPPER_USDT"):
        assert pair not in manager.valid_pairs
        assert manager.skipped_reasons[pair] == (
            "non-crypto commodity contract excluded from crypto-only scanning"
        )
