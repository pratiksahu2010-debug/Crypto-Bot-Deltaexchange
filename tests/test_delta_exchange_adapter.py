import time
import pytest

from exchange.delta_rest import DeltaExchangeRest


def _product(symbol="BTCUSD", asset="BTC", contract_value="0.001", unit="BTC"):
    return {
        "id": 27, "symbol": symbol, "contract_type": "perpetual_futures",
        "state": "live", "trading_status": "operational",
        "contract_value": contract_value, "contract_unit_currency": unit,
        "underlying_asset": {"symbol": asset},
        "quoting_asset": {"symbol": "USD"},
    }


@pytest.mark.asyncio
async def test_delta_products_are_paginated_and_normalized_to_stable_strategy_keys():
    client = DeltaExchangeRest()
    calls = []

    async def fake_get(url, params=None):
        calls.append(dict(params or {}))
        if not params.get("after"):
            return {
                "success": True, "result": [_product()],
                "meta": {"after": "cursor-1"},
            }
        return {
            "success": True,
            "result": [_product("ETHUSD", "ETH", "0.01", "ETH")],
            "meta": {"after": None},
        }

    client._get = fake_get
    markets = await client.get_market_details()

    assert [m["pair"] for m in markets] == ["B-BTC_USDT", "B-ETH_USDT"]
    assert [m["delta_symbol"] for m in markets] == ["BTCUSD", "ETHUSD"]
    assert client.delta_symbol_for_pair("B-BTC_USDT") == "BTCUSD"
    assert client.canonical_pair_for_symbol("ETHUSD") == "B-ETH_USDT"
    assert calls[0]["contract_types"] == "perpetual_futures"
    assert calls[0]["states"] == "live"
    assert calls[1]["after"] == "cursor-1"


@pytest.mark.asyncio
async def test_delta_ticker_uses_close_and_explicit_usd_turnover():
    client = DeltaExchangeRest()
    client._canonical_to_delta = {"B-BTC_USDT": "BTCUSD"}
    client._delta_to_canonical = {"BTCUSD": "B-BTC_USDT"}

    async def fake_get(url, params=None):
        assert url.endswith("/v2/tickers")
        assert params == {"contract_types": "perpetual_futures"}
        return {"success": True, "result": [{
            "symbol": "BTCUSD", "close": "65000.5", "mark_price": "65100",
            "timestamp": int(time.time()), "turnover_usd": 1234567,
            "turnover": 1200000, "turnover_symbol": "INR", "volume": 10,
            "contract_type": "perpetual_futures",
        }]}

    client._get = fake_get
    rows = await client.get_ticker()
    assert len(rows) == 1
    assert rows[0]["pair"] == "B-BTC_USDT"
    assert rows[0]["last_price"] == "65000.5"
    assert rows[0]["quote_volume"] == 1234567
    assert rows[0]["mark_price"] == "65100"


@pytest.mark.asyncio
async def test_delta_native_15m_candles_are_requested_without_synthetic_aggregation(monkeypatch):
    base = 1_800_000_000
    monkeypatch.setattr("exchange.delta_rest.time.time", lambda: base + 1803)
    client = DeltaExchangeRest()
    client._canonical_to_delta = {"B-BTC_USDT": "BTCUSD"}
    seen = {}

    async def fake_get(url, params=None):
        seen.update(params)
        assert url.endswith("/v2/history/candles")
        return {"success": True, "result": [
            {"time": base, "open": 100, "high": 102, "low": 99, "close": 101, "volume": 10},
            {"time": base + 900, "open": 101, "high": 103, "low": 100, "close": 102, "volume": 20},
        ]}

    client._get = fake_get
    candles = await client.get_candles("B-BTC_USDT", "15m", limit=10)

    assert seen["symbol"] == "BTCUSD"
    assert seen["resolution"] == "15m"
    assert [c["time"] for c in candles] == [base * 1000, (base + 900) * 1000]


@pytest.mark.asyncio
async def test_delta_candles_use_seconds_api_window_and_reject_outside_rows(monkeypatch):
    base = 1_800_000_000
    now = base + 1200
    monkeypatch.setattr("exchange.delta_rest.time.time", lambda: now)
    client = DeltaExchangeRest()
    client._canonical_to_delta = {"B-BTC_USDT": "BTCUSD"}
    start_ms, end_ms = base * 1000, (base + 900) * 1000

    async def fake_get(url, params=None):
        assert params["start"] == base
        assert params["end"] == base + 900
        return {"success": True, "result": [
            {"time": base - 300, "open": 99, "high": 100, "low": 98, "close": 99, "volume": 1},
            {"time": base, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 2},
            {"time": base + 300, "open": 100, "high": 102, "low": 99, "close": 101, "volume": 3},
            {"time": base + 600, "open": 101, "high": 103, "low": 100, "close": 102, "volume": 4},
            {"time": base + 900, "open": 102, "high": 104, "low": 101, "close": 103, "volume": 5},
        ]}

    client._get = fake_get
    candles = await client.get_candles(
        "B-BTC_USDT", "5m", limit=10, start_time=start_ms, end_time=end_ms
    )
    assert [c["time"] for c in candles] == [start_ms, start_ms + 300_000, start_ms + 600_000]


@pytest.mark.asyncio
async def test_delta_orderbook_depth_uses_contract_value_and_microsecond_timestamp(monkeypatch):
    now_ms = int(time.time() * 1000)
    client = DeltaExchangeRest()
    client._canonical_to_delta = {"B-BTC_USDT": "BTCUSD"}
    client._product_metadata = {
        "B-BTC_USDT": {
            "contract_value": "0.001", "contract_unit_currency": "BTC",
            "underlying_asset": {"symbol": "BTC"}, "quoting_asset": {"symbol": "USD"},
        }
    }

    async def fake_get(url, params=None):
        assert url.endswith("/v2/l2orderbook/BTCUSD")
        return {"success": True, "result": {
            "last_updated_at": now_ms * 1000,
            "buy": [["65000", "10"], ["64999", "10"]],
            "sell": [["65001", "10"], ["65002", "10"]],
        }}

    client._get = fake_get
    book = await client.get_futures_orderbook("B-BTC_USDT")
    assert book["best_bid"] == 65000
    assert book["best_ask"] == 65001
    assert book["bid_depth_usdt"] == pytest.approx((65000 * 10 + 64999 * 10) * 0.001)


def test_delta_ws_candlestick_maps_product_to_canonical_pair_and_updates_ltp(monkeypatch):
    from exchange.delta_ws import DeltaFuturesStream

    class Builder:
        def __init__(self):
            self.rows = []

        def on_candle(self, pair, interval, candle):
            self.rows.append((pair, interval, candle))

        def get_buffer(self, pair, interval):
            return type("Buffer", (), {"candles": []})()

    now_ms = int(time.time() * 1000)
    client = DeltaExchangeRest()
    client._delta_to_canonical = {"BTCUSD": "B-BTC_USDT"}
    client._canonical_to_delta = {"B-BTC_USDT": "BTCUSD"}
    builder = Builder()
    stream = DeltaFuturesStream(builder, client)
    interval_ms = 300_000
    candle_open = now_ms // interval_ms * interval_ms

    stream._handle_candlestick({
        "type": "candlestick_5m", "res": "5m", "sy": "BTCUSD",
        "ts": now_ms * 1000, "o": 100, "h": 102, "l": 99, "c": 101, "v": 10,
    })

    assert builder.rows
    assert builder.rows[0][0:2] == ("B-BTC_USDT", "5m")
    assert builder.rows[0][2]["time"] == candle_open
    assert client._ws_prices["B-BTC_USDT"][0] == 101



def test_delta_liquidity_uses_explicit_usd_turnover_not_contract_volume():
    from exchange.symbol_manager import SymbolManager

    now = int(time.time())
    rows = [{
        "market": "B-BTC_USDT", "pair": "B-BTC_USDT", "symbol": "BTCUSD",
        "timestamp": now, "turnover_usd": 1_250_000,
        "quote_volume": 1_250_000, "base_volume": 0.001, "last_price": 65000,
    }]
    manager = SymbolManager(type("RestStub", (), {})())
    values = manager._ticker_liquidity(rows)
    assert values["BTCUSDT"] == 1_250_000
    assert values["BTCUSD"] == 1_250_000

def test_delta_liquidity_does_not_infer_usd_turnover_from_contract_volume():
    from exchange.symbol_manager import SymbolManager

    rows = [{
        "market": "B-BTC_USDT", "pair": "B-BTC_USDT",
        "symbol": "BTCUSD", "delta_symbol": "BTCUSD",
        "contract_type": "perpetual_futures",
        "timestamp": int(time.time()), "turnover_usd": 0,
        "quote_volume": 0, "base_volume": 100,
        "last_price": 65000,
    }]
    manager = SymbolManager(type("RestStub", (), {})())
    assert manager._ticker_liquidity(rows) == {}



@pytest.mark.asyncio
async def test_symbol_manager_accepts_delta_usd_perpetual_under_internal_strategy_key(monkeypatch):
    from exchange.symbol_manager import SymbolManager

    class RestStub:
        MAX_REST_TICKER_AGE_SECONDS = 15.0
        _ticker_age_seconds = staticmethod(DeltaExchangeRest._ticker_age_seconds)
        _is_fresh_ticker_age = staticmethod(DeltaExchangeRest._is_fresh_ticker_age)

        async def get_market_details(self):
            return [{
                "pair": "B-BTC_USDT", "symbol": "BTCUSD",
                "base_currency_short_name": "USD",
                "target_currency_short_name": "BTC",
                "status": "active", "quote_currency": "USD",
            }]

        async def get_ticker(self):
            # Missing turnover must not rotate a fixed, preselected allowlist.
            return [{
                "market": "B-BTC_USDT", "pair": "B-BTC_USDT",
                "symbol": "BTCUSD", "timestamp": int(time.time()),
                "quote_volume": 0, "turnover_usd": 0,
            }]

    manager = SymbolManager(RestStub())
    monkeypatch.setattr(manager, "_load_yaml", lambda path: ["B-BTC_USDT"])
    await manager.load_and_validate()

    assert manager.valid_pairs == {"B-BTC_USDT"}
    assert manager.normal_symbols == ["B-BTC_USDT"]
    assert manager.momentum_symbols == ["B-BTC_USDT"]
    assert manager.display_name("B-BTC_USDT") == "BTC/USD"
