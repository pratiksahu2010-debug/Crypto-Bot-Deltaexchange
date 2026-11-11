"""Fixed Delta allowlist and per-symbol exchange diagnostics tests."""
import asyncio
import time
from pathlib import Path

import yaml

from exchange.symbol_manager import SymbolManager
from exchange.delta_rest import DeltaExchangeRest


def _fixed_lists():
    root = Path(__file__).resolve().parents[1] / "config"
    normal = yaml.safe_load((root / "normal_symbols.yaml").read_text())["symbols"]
    momentum = yaml.safe_load((root / "momentum_symbols.yaml").read_text())["symbols"]
    return normal, momentum


def _asset(pair):
    return pair.removeprefix("B-").removeprefix("I-").removesuffix("_USDT")


class FakeRest:
    MAX_REST_TICKER_AGE_SECONDS = 15.0
    _ticker_age_seconds = staticmethod(DeltaExchangeRest._ticker_age_seconds)

    def __init__(self, details, omit_tickers=False):
        self.details = details
        self.omit_tickers = omit_tickers

    async def get_market_details(self):
        return self.details

    async def get_ticker(self):
        if self.omit_tickers:
            return []
        now = int(time.time())
        return [
            {
                "pair": row["pair"], "symbol": row["symbol"],
                "timestamp": now, "last_price": 1.0,
                "turnover_usd": 0,
            }
            for row in self.details
            if str(row.get("status", "active")).lower() == "active"
        ]


def _details(pairs, inactive=(), extras=()):
    rows = []
    inactive = set(inactive)
    for pair in list(pairs) + list(extras):
        asset = _asset(pair)
        rows.append({
            "pair": pair, "symbol": f"{asset}USD",
            "target_currency_short_name": asset,
            "base_currency_short_name": "USD",
            "quote_currency": "USD",
            "status": "inactive" if pair in inactive else "active",
            "contract_type": "perpetual_futures",
        })
    return rows


def test_momentum_whitelist_reports_not_listed_and_inactive_symbols():
    normal, momentum = _fixed_lists()
    missing, inactive = momentum[:2]
    available = [pair for pair in momentum if pair != missing]
    rest = FakeRest(_details(available, inactive=(inactive,)))
    manager = SymbolManager(rest)

    asyncio.run(manager.load_and_validate())

    assert manager.whitelist_diagnostics[manager._norm(missing)]["status"] == "NOT_LISTED"
    assert manager.whitelist_diagnostics[manager._norm(inactive)]["status"] == "INACTIVE"
    assert missing not in manager.momentum_symbols
    assert inactive not in manager.momentum_symbols
    assert manager.normal_symbols == [p for p in normal if p not in (missing, inactive)]


def test_unlisted_live_products_never_replace_missing_static_symbols():
    normal, momentum = _fixed_lists()
    missing = momentum[0]
    available = [pair for pair in momentum if pair != missing]
    rest = FakeRest(_details(available, extras=("B-UNLISTEDXYZ_USDT",)))
    manager = SymbolManager(rest)

    asyncio.run(manager.load_and_validate())

    assert missing not in manager.momentum_symbols
    assert "B-UNLISTEDXYZ_USDT" not in manager.momentum_symbols
    assert manager.momentum_symbols == available


def test_fixed_universe_never_adds_unlisted_crypto_or_stablecoin_pairs():
    normal, momentum = _fixed_lists()
    extras = ("B-UNLISTEDXYZ_USDT", "B-USDC_USDT")
    manager = SymbolManager(FakeRest(_details(momentum, extras=extras)))

    asyncio.run(manager.load_and_validate())

    assert manager.normal_symbols == normal
    assert manager.momentum_symbols == momentum
    assert not any("UNLISTEDXYZ" in pair or "USDC" in pair for pair in manager.normal_symbols + manager.momentum_symbols)
    assert "B-USDC_USDT" not in manager.valid_pairs


def test_normal_40_and_momentum_100_fixed_allowlists():
    normal, momentum = _fixed_lists()
    assert len(normal) == 40
    assert len(set(normal)) == 40
    assert len(momentum) == 100
    assert len(set(momentum)) == 100
    assert set(normal).issubset(set(momentum))
    assert all(symbol.startswith("B-") and symbol.endswith("_USDT") for symbol in momentum)


def test_zero_turnover_does_not_filter_fixed_symbols():
    normal, momentum = _fixed_lists()
    manager = SymbolManager(FakeRest(_details(momentum)))

    asyncio.run(manager.load_and_validate())

    assert manager.normal_symbols == normal
    assert manager.momentum_symbols == momentum
    assert all(manager.ticker_diagnostics[p]["status"] == "FRESH" for p in momentum)


def test_missing_ticker_does_not_change_fixed_universe_and_is_reported():
    normal, momentum = _fixed_lists()
    manager = SymbolManager(FakeRest(_details(momentum), omit_tickers=True))

    asyncio.run(manager.load_and_validate())

    assert manager.normal_symbols == normal
    assert manager.momentum_symbols == momentum
    assert all(manager.ticker_diagnostics[p]["status"] == "TICKER_MISSING" for p in momentum)
