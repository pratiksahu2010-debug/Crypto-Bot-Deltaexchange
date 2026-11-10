import asyncio

from exchange.symbol_manager import SymbolManager


class FakeRest:
    def __init__(self, details):
        self.details = details

    async def get_market_details(self):
        return self.details

    async def get_ticker(self):
        return [{"market": "B-BTC_USDT", "base_volume": "10000", "last_price": "1000"}]


def test_momentum_whitelist_reports_not_listed_and_inactive_symbols(monkeypatch):
    async def run():
        rest = FakeRest([
            {
                "pair": "B-EOS_USDT",
                "coindcx_name": "B-EOS_USDT",
                "target_currency_short_name": "EOS",
                "base_currency_short_name": "USDT",
                "status": "inactive",
            },
            {
                "pair": "B-BTC_USDT",
                "coindcx_name": "B-BTC_USDT",
                "target_currency_short_name": "BTC",
                "base_currency_short_name": "USDT",
                "status": "active",
            },
        ])
        manager = SymbolManager(rest)

        def load_yaml(path):
            if path.endswith("momentum_symbols.yaml"):
                return ["B-SHIB_USDT", "B-EOS_USDT", "B-BTC_USDT"]
            return []

        monkeypatch.setattr(manager, "_load_yaml", load_yaml)
        await manager.load_and_validate()

        assert manager.whitelist_diagnostics["SHIBUSDT"]["status"] == "NOT_LISTED"
        assert manager.whitelist_diagnostics["EOSUSDT"]["status"] == "INACTIVE"
        assert manager.momentum_symbols == ["B-BTC_USDT"]

    asyncio.run(run())



def test_unlisted_static_symbols_are_not_replaced_by_dynamic_futures(monkeypatch):
    async def run():
        class MultiMarketRest(FakeRest):
            async def get_ticker(self):
                return [
                    {"market": "B-BTC_USDT", "base_volume": "10000", "last_price": "1000"},
                    {"market": "B-ETH_USDT", "base_volume": "10000", "last_price": "100"},
                    {"market": "B-SOL_USDT", "base_volume": "10000", "last_price": "50"},
                ]

        rest = MultiMarketRest([
            {
                "pair": f"B-{symbol}_USDT",
                "coindcx_name": f"B-{symbol}_USDT",
                "target_currency_short_name": symbol,
                "base_currency_short_name": "USDT",
                "status": "active",
            }
            for symbol in ("BTC", "ETH", "SOL")
        ])
        manager = SymbolManager(rest)

        def load_yaml(path):
            if path.endswith("normal_symbols.yaml"):
                return ["B-TON_USDT", "B-BTC_USDT"]
            if path.endswith("momentum_symbols.yaml"):
                return ["B-TON_USDT", "B-BTC_USDT"]
            return []

        monkeypatch.setattr(manager, "_load_yaml", load_yaml)
        monkeypatch.setattr("exchange.symbol_manager.settings.normal_symbol_limit", 2)
        monkeypatch.setattr("exchange.symbol_manager.settings.momentum_symbol_limit", 2)
        monkeypatch.setattr("exchange.symbol_manager.settings.normal_min_24h_turnover_usdt", 1_000_000)
        monkeypatch.setattr("exchange.symbol_manager.settings.momentum_min_24h_turnover_usdt", 500_000)
        await manager.load_and_validate()

        assert manager.normal_symbols == ["B-BTC_USDT"]
        assert manager.momentum_symbols == ["B-BTC_USDT"]
        assert manager.whitelist_diagnostics["TONUSDT"]["status"] == "NOT_LISTED"

    asyncio.run(run())


def test_fixed_universe_never_adds_dynamic_symbols_or_stablecoins(monkeypatch):
    async def run():
        class MultiMarketRest(FakeRest):
            async def get_ticker(self):
                return [
                    {"market": "B-BTC_USDT", "base_volume": "10000", "last_price": "1000"},
                    {"market": "B-ETH_USDT", "base_volume": "1", "last_price": "100"},
                    {"market": "B-USDC_USDT", "base_volume": "10000000", "last_price": "1"},
                ]

        rest = MultiMarketRest([
            {
                "pair": f"B-{symbol}_USDT",
                "coindcx_name": f"B-{symbol}_USDT",
                "target_currency_short_name": symbol,
                "base_currency_short_name": "USDT",
                "status": "active",
            }
            for symbol in ("BTC", "ETH", "USDC")
        ])
        manager = SymbolManager(rest)

        def load_yaml(path):
            if path.endswith("normal_symbols.yaml"):
                return ["B-TON_USDT"]
            if path.endswith("momentum_symbols.yaml"):
                return ["B-TON_USDT"]
            return []

        monkeypatch.setattr(manager, "_load_yaml", load_yaml)
        monkeypatch.setattr("exchange.symbol_manager.settings.normal_symbol_limit", 2)
        monkeypatch.setattr("exchange.symbol_manager.settings.momentum_symbol_limit", 2)
        monkeypatch.setattr("exchange.symbol_manager.settings.normal_min_24h_turnover_usdt", 1_000_000)
        monkeypatch.setattr("exchange.symbol_manager.settings.momentum_min_24h_turnover_usdt", 500_000)
        await manager.load_and_validate()

        assert manager.normal_symbols == []
        assert manager.momentum_symbols == []
        assert "B-USDC_USDT" not in manager.normal_symbols + manager.momentum_symbols
        assert "B-ETH_USDT" not in manager.normal_symbols + manager.momentum_symbols

    asyncio.run(run())


def test_normal_40_and_momentum_100_fixed_allowlists():
    from pathlib import Path
    import yaml

    root = Path(__file__).resolve().parent.parent
    normal = yaml.safe_load((root / "config/normal_symbols.yaml").read_text())["symbols"]
    momentum = yaml.safe_load((root / "config/momentum_symbols.yaml").read_text())["symbols"]

    assert len(normal) == 40
    assert len(set(normal)) == 40
    assert len(momentum) == 100
    assert len(set(momentum)) == 100
    assert set(normal).issubset(set(momentum))
    assert all(symbol.startswith("B-") and symbol.endswith("_USDT") for symbol in momentum)



def test_turnover_never_excludes_static_symbols(monkeypatch):
    async def run():
        class TurnoverRest(FakeRest):
            async def get_ticker(self):
                return [
                    {"market": "B-BTC_USDT", "turnover_usd": "2000000"},
                    {"market": "B-ETH_USDT", "turnover_usd": "100000"},
                ]

        rest = TurnoverRest([
            {
                "pair": f"B-{symbol}_USDT",
                "target_currency_short_name": symbol,
                "base_currency_short_name": "USDT",
                "status": "active",
            }
            for symbol in ("BTC", "ETH")
        ])
        manager = SymbolManager(rest)
        monkeypatch.setattr(manager, "_load_yaml", lambda path: ["B-BTC_USDT", "B-ETH_USDT"])
        monkeypatch.setattr("exchange.symbol_manager.settings.normal_symbol_limit", 2)
        monkeypatch.setattr("exchange.symbol_manager.settings.momentum_symbol_limit", 2)
        monkeypatch.setattr("exchange.symbol_manager.settings.normal_min_24h_turnover_usdt", 1_000_000)
        monkeypatch.setattr("exchange.symbol_manager.settings.momentum_min_24h_turnover_usdt", 500_000)
        await manager.load_and_validate()

        assert manager.normal_symbols == ["B-BTC_USDT", "B-ETH_USDT"]
        assert manager.momentum_symbols == ["B-BTC_USDT", "B-ETH_USDT"]

    asyncio.run(run())


def test_unknown_turnover_does_not_empty_static_universe(monkeypatch):
    async def run():
        class UnknownTurnoverRest(FakeRest):
            async def get_ticker(self):
                return []

        rest = UnknownTurnoverRest([{
            "pair": "B-BTC_USDT",
            "target_currency_short_name": "BTC",
            "base_currency_short_name": "USDT",
            "status": "active",
        }])
        manager = SymbolManager(rest)
        monkeypatch.setattr(manager, "_load_yaml", lambda path: ["B-BTC_USDT"])
        await manager.load_and_validate()

        assert manager.normal_symbols == ["B-BTC_USDT"]
        assert manager.momentum_symbols == ["B-BTC_USDT"]
        assert manager.whitelist_diagnostics["BTCUSDT"]["status"] == "LIQUIDITY_UNKNOWN"

    asyncio.run(run())
