"""Static strategy allowlists validated against live Delta USD-quoted perpetuals."""
import re
import math
import time
import yaml
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime, timezone
from utils.logger import logger
from exchange.delta_rest import DeltaExchangeRest
from config import settings


class SymbolManager:
    def __init__(self, rest: DeltaExchangeRest):
        self.rest = rest
        self.valid_pairs: Set[str] = set()
        self.normal_symbols: List[str] = []
        self.momentum_symbols: List[str] = []
        self.pair_to_symbol: Dict[str, str] = {}
        self.metadata: Dict[str, dict] = {}
        self.skipped_reasons: Dict[str, str] = {}
        self.whitelist_diagnostics: Dict[str, dict] = {}
        self.ticker_diagnostics: Dict[str, dict] = {}
        self._ranked_normal: List[str] = []
        self._ranked_momentum: List[str] = []
        self._quarantined: Dict[str, dict] = {}
        self._quarantine_retry_seconds = 15 * 60

    @staticmethod
    def _norm(value: object) -> str:
        value = str(value or "").upper().strip()
        value = re.sub(r"^(B-|I-)", "", value)
        return re.sub(r"[^A-Z0-9]", "", value)

    # Stablecoin-underlying contracts have no meaningful directional edge for
    # this momentum/trend scanner. Keep this list exact (not substring-based).
    STABLECOIN_BASES = {
        "USDC", "BUSD", "FDUSD", "TUSD", "DAI", "USDE", "USDS",
        "PYUSD", "USDP", "USD1", "UST", "USTC", "USDD", "FRAX",
        "GUSD", "LUSD", "MIM", "CRVUSD", "EURC", "EURS",
    }

    # Traditional commodity contracts share the exchange's Futures feed with
    # crypto perpetuals. Exclude them (and tokenized gold exposure) entirely.
    NON_CRYPTO_BASES = {
        "XAU", "GOLD", "PAXG", "XAUT", "XAG", "SILVER",
        "CL", "BZ", "WTI", "BRENT", "OIL", "CRUDEOIL",
        "NATGAS", "NG", "NATURALGAS",
        "COPPER", "HG", "ALUMINIUM", "ALUMINUM", "ZINC", "LEAD",
        "NICKEL", "TIN", "PLATINUM", "PALLADIUM",
        "WHEAT", "CORN", "SOYBEAN", "SUGAR", "COFFEE", "COTTON",
    }

    @classmethod
    def _stablecoin_base(cls, pair: str, target: str = "") -> str:
        if target:
            return cls._norm(target)
        normalized = str(pair or "").upper().strip()
        normalized = re.sub(r"^(B-|I-)", "", normalized)
        if normalized.endswith("_USDT"):
            return cls._norm(normalized[:-5])
        if normalized.endswith("USDT"):
            return cls._norm(normalized[:-4])
        return ""

    @classmethod
    def _is_stablecoin_pair(cls, pair: str, target: str = "") -> bool:
        return cls._stablecoin_base(pair, target) in cls.STABLECOIN_BASES

    @classmethod
    def _is_non_crypto_pair(cls, pair: str, target: str = "") -> bool:
        """Return True for known commodity/metal/energy contracts, not crypto assets."""
        base = cls._norm(target) if target else ""
        if not base:
            normalized = str(pair or "").upper().strip()
            normalized = re.sub(r"^(B-|I-)", "", normalized)
            base = cls._norm(normalized.split("_", 1)[0])
        return base in cls.NON_CRYPTO_BASES

    def _load_yaml(self, rel_path: str) -> List[str]:
        path = Path(__file__).resolve().parent.parent / rel_path
        if not path.exists():
            logger.warning(f"Optional symbol preference file missing: {path}")
            return []
        try:
            with open(path, encoding="utf-8") as stream:
                data = yaml.safe_load(stream) or {}
            return list(dict.fromkeys(str(x).upper() for x in data.get("symbols", []) if x))
        except Exception as exc:
            logger.warning(f"Could not read optional symbol preferences {path}: {exc}")
            return []

    async def load_and_validate(self) -> None:
        details = []
        try:
            details = await self.rest.get_market_details()
        except Exception as exc:
            logger.error(f"markets_details failed: {exc}")

        # Fail closed: only live Delta perpetual products supplied by the
        # exchange adapter are eligible. Spot and option products are excluded.
        if not details:
            logger.error("No live USD-quoted Delta perpetual products; static universes remain empty")

        # Symbol membership is controlled only by the checked-in static YAML
        # allowlists and live perpetual-product validation.

        self.valid_pairs.clear()
        self.metadata.clear()
        self.pair_to_symbol.clear()
        self.skipped_reasons.clear()

        for market in details:
            if not isinstance(market, dict):
                continue
            status = str(market.get("status") or "active").lower()
            if status not in ("active", "trading", "enabled", "online", "1", ""):
                continue
            target = str(market.get("target_currency_short_name") or "").upper()
            raw_pair = str(market.get("pair") or "")
            pair = raw_pair
            # The _USDT suffix is the preserved internal strategy-key format;
            # it is not the actual Delta quote currency (USD).
            if not pair.upper().endswith("_USDT"):
                continue
            if not pair.upper().startswith(("B-", "I-")):
                if target:
                    pair = f"B-{target}_USDT"
                else:
                    continue
            pair = pair.upper()
            if not pair.endswith("_USDT"):
                continue
            if self._is_non_crypto_pair(pair, target):
                self.skipped_reasons[pair] = "non-crypto commodity contract excluded from crypto-only scanning"
                continue
            if self._is_stablecoin_pair(pair, target):
                self.skipped_reasons[pair] = "stablecoin-to-USDT pair excluded from directional scanning"
                continue
            self.valid_pairs.add(pair)
            self.metadata[pair] = market
            display = f"{target}/USD" if target else pair.removeprefix("B-").removeprefix("I-").replace("_", "/")
            self.pair_to_symbol[pair] = display


        logger.info(
            f"Loaded {len(self.valid_pairs)} eligible active Delta perpetual markets "
            f"(stablecoin pairs excluded: {sum('stablecoin-to-USDT pair' in reason for reason in self.skipped_reasons.values())})"
        )

        normal_preferences = self._load_yaml("config/normal_symbols.yaml")
        momentum_preferences = self._load_yaml("config/momentum_symbols.yaml")
        normal_rank = {self._norm(s): i for i, s in enumerate(normal_preferences)}
        momentum_rank = {self._norm(s): i for i, s in enumerate(momentum_preferences)}

        # Keep the raw active-instrument response beside the filtered universe
        # so every static Momentum symbol gets an actionable reason when absent.
        discovered_by_key = {}
        for market in details:
            if not isinstance(market, dict):
                continue
            aliases = (market.get("pair"), market.get("symbol"))
            for alias in aliases:
                key = self._norm(alias)
                if key:
                    discovered_by_key[key] = market
        self.whitelist_diagnostics.clear()

        def rank_universe(preferences: Dict[str, int], label: str) -> List[str]:
            # Preserve every configured symbol in YAML order. Only live Delta
            # perpetual validation may mark a configured member unavailable.
            configured = list(preferences.keys())
            available_by_key = {self._norm(pair): pair for pair in self.valid_pairs}
            selected: List[str] = []
            missing: List[str] = []
            for key in configured:
                pair = available_by_key.get(key)
                if pair:
                    selected.append(pair)
                    continue
                missing.append(key)
                raw_market = discovered_by_key.get(key)
                if raw_market is None:
                    status = "NOT_LISTED"
                    reason = "not returned by live USD-quoted Delta perpetual products"
                else:
                    market_status = str(raw_market.get("status") or "active").lower()
                    if market_status not in ("active", "trading", "enabled", "online", "1", ""):
                        status = "INACTIVE"
                        reason = f"exchange status={market_status}"
                    elif self._is_stablecoin_pair(
                        str(raw_market.get("pair") or ""),
                        str(raw_market.get("target_currency_short_name") or ""),
                    ):
                        status = "EXCLUDED"
                        reason = "stablecoin-to-USDT pair excluded"
                    else:
                        status = "NOT_ELIGIBLE"
                        reason = "returned by exchange but failed Delta perpetual normalization"
                self.whitelist_diagnostics[key] = {
                    "configured_symbol": key, "status": status, "reason": reason,
                }
                logger.warning(
                    f"{label.upper()}_STATIC_SYMBOL symbol={key} status={status} reason={reason}"
                )
            if missing:
                logger.warning(
                    f"{label.upper()}_STATIC_SYMBOLS_UNAVAILABLE count={len(missing)} "
                    f"symbols={','.join(missing)}"
                )

            # No symbol outside the configured allowlist is ever introduced.
            return selected

        if len(normal_preferences) != 40 or len(set(self._norm(x) for x in normal_preferences)) != 40:
            raise ValueError(
                f"config/normal_symbols.yaml must contain exactly 40 unique fixed symbols; "
                f"found {len(normal_preferences)} entries"
            )
        if len(momentum_preferences) != 100 or len(set(self._norm(x) for x in momentum_preferences)) != 100:
            raise ValueError(
                f"config/momentum_symbols.yaml must contain exactly 100 unique fixed symbols; "
                f"found {len(momentum_preferences)} entries"
            )

        self._ranked_normal = rank_universe(normal_rank, "Normal")
        self._ranked_momentum = rank_universe(momentum_rank, "Momentum")
        self._select_active_symbols()
        # Ticker health is diagnostic only: it never ranks, removes, or replaces
        # a member of either fixed allowlist.
        await self._validate_configured_tickers(normal_preferences, momentum_preferences)
        logger.info(
            f"Static universe ready: normal={len(self.normal_symbols)}, "
            f"momentum={len(self.momentum_symbols)}, eligible={len(self.valid_pairs)}"
        )
        if not self.valid_pairs:
            logger.error("No eligible live Delta perpetual markets were discovered")


    def _refresh_quarantine_from_rest(self) -> None:
        """Quarantine markets whose required candle feeds repeatedly return stale history."""
        getter = getattr(self.rest, "get_stale_market_timeframes", None)
        stale = getter() if callable(getter) else {}
        required = {settings.candle_primary_tf}
        now = __import__("time").time()
        for pair, timeframe_counts in (stale or {}).items():
            relevant = {tf: int(count) for tf, count in timeframe_counts.items() if tf in required}
            # A single required timeframe is not evidence of a persistent
            # outage. Quarantine only after at least two stale observations
            # for a required feed; secondary timeframes are intentionally ignored.
            repeated = any(count >= 2 for count in relevant.values())
            if not repeated:
                continue
            reason = "repeated stale candle data: " + ", ".join(
                f"{tf} (observed {count}x)" for tf, count in sorted(relevant.items())
            )
            existing = self._quarantined.get(pair)
            if existing is None:
                retry_at = now + self._quarantine_retry_seconds
                self._quarantined[pair] = {
                    "since": now, "retry_at": retry_at, "reason": reason,
                }
                logger.warning(
                    f"MARKET_QUARANTINED {pair} reason={reason} "
                    f"retry_in_seconds={self._quarantine_retry_seconds}"
                )
            else:
                existing["reason"] = reason

    async def _validate_configured_tickers(
        self, normal_preferences: List[str], momentum_preferences: List[str]
    ) -> None:
        """Record per-symbol ticker readiness without changing fixed membership."""
        configured = list(dict.fromkeys(normal_preferences + momentum_preferences))
        self.ticker_diagnostics.clear()
        getter = getattr(self.rest, "get_ticker", None)
        if not callable(getter):
            for pair in configured:
                self.ticker_diagnostics[pair] = {
                    "status": "TICKER_CHECK_UNAVAILABLE",
                    "reason": "REST adapter has no ticker method",
                }
            return

        try:
            rows = await getter()
        except Exception as exc:
            reason = f"ticker_request_failed:{type(exc).__name__}:{exc}"
            for pair in configured:
                self.ticker_diagnostics[pair] = {
                    "status": "TICKER_REQUEST_FAILED", "reason": reason,
                }
            logger.error("DELTA_TICKER_VALIDATION_FAILED symbols=%s error=%s", len(configured), exc)
            return

        by_pair = {
            str(row.get("pair") or row.get("market") or "").upper(): row
            for row in (rows or []) if isinstance(row, dict)
        }
        age_reader = getattr(self.rest, "_ticker_age_seconds", None)
        max_age = float(getattr(self.rest, "MAX_REST_TICKER_AGE_SECONDS", 15.0))
        now_ms = int(time.time() * 1000)
        counts = {}

        for pair in configured:
            if pair not in self.valid_pairs:
                diagnostic = {
                    "status": "PRODUCT_UNAVAILABLE",
                    "reason": "configured symbol has no eligible live Delta perpetual product",
                }
            else:
                row = by_pair.get(pair.upper())
                if row is None:
                    diagnostic = {
                        "status": "TICKER_MISSING",
                        "reason": "live perpetual product exists but no ticker row was returned",
                    }
                else:
                    raw_price = row.get("last_price", row.get("close"))
                    try:
                        price = float(raw_price)
                    except (TypeError, ValueError, OverflowError):
                        price = float("nan")
                    timestamp = row.get("timestamp") or row.get("ts")
                    age = None
                    if timestamp is not None and callable(age_reader):
                        try:
                            age = age_reader(timestamp, now_ms)
                        except (TypeError, ValueError, OverflowError):
                            age = None
                    if not math.isfinite(price) or price <= 0:
                        diagnostic = {
                            "status": "INVALID_TICKER_PRICE",
                            "reason": "ticker last price is missing, non-numeric, or non-positive",
                        }
                    elif age is None:
                        diagnostic = {
                            "status": "TICKER_TIMESTAMP_MISSING",
                            "reason": "ticker has no valid exchange timestamp",
                        }
                    elif age < -15.0 or age > max_age:
                        diagnostic = {
                            "status": "TICKER_STALE",
                            "reason": f"ticker age {age:.1f}s exceeds allowed range",
                            "age_seconds": round(age, 1),
                        }
                    else:
                        diagnostic = {
                            "status": "FRESH",
                            "reason": None,
                            "age_seconds": round(max(0.0, age), 1),
                            "last_price": price,
                        }

            self.ticker_diagnostics[pair] = diagnostic
            counts[diagnostic["status"]] = counts.get(diagnostic["status"], 0) + 1
            if diagnostic["status"] != "FRESH":
                logger.warning(
                    "DELTA_TICKER_SYMBOL pair=%s status=%s reason=%s",
                    pair, diagnostic["status"], diagnostic.get("reason"),
                )

        logger.info(
            "DELTA_TICKER_VALIDATION configured=%s fresh=%s issues=%s",
            len(configured), counts.get("FRESH", 0),
            sum(value for key, value in counts.items() if key != "FRESH"),
        )

    def _select_active_symbols(self) -> None:
        """Activate only live exchange products present in the fixed YAML lists.

        Turnover, ticker volume and environment-provided symbol caps never
        filter, rank, add or replace configured symbols. Repeatedly stale
        candle feeds may be temporarily quarantined as a data-health safeguard.
        """
        excluded = set(self._quarantined)
        self.normal_symbols = [
            pair for pair in self._ranked_normal if pair not in excluded
        ]
        self.momentum_symbols = [
            pair for pair in self._ranked_momentum if pair not in excluded
        ]
        selected = set(self.normal_symbols) | set(self.momentum_symbols)

        for pair in selected:
            reason = self.skipped_reasons.get(pair, "")
            if reason.startswith("temporarily quarantined until "):
                self.skipped_reasons.pop(pair, None)

        for pair, item in self._quarantined.items():
            retry_at = datetime.fromtimestamp(item["retry_at"], tz=timezone.utc).isoformat()
            self.skipped_reasons[pair] = (
                f"temporarily quarantined until {retry_at}: {item['reason']}"
            )
            logger.info(
                f"MARKET_EXCLUDED {pair} normal={'yes' if pair in self._ranked_normal else 'no'} "
                f"momentum={'yes' if pair in self._ranked_momentum else 'no'} retry_at={retry_at}"
            )

        for pair in self.valid_pairs - selected - excluded:
            self.skipped_reasons.setdefault(pair, "outside fixed configured strategy allowlists")
        logger.info(
            f"ACTIVE_FIXED_UNIVERSE normal={len(self.normal_symbols)}/40 "
            f"momentum={len(self.momentum_symbols)}/100 "
            f"quarantined={len(self._quarantined)} turnover_filter=disabled"
        )

    async def refresh_universes(self) -> None:
        """Apply stale-feed quarantine and retry quarantined pairs every 15 minutes."""
        import asyncio
        import time

        now = time.time()
        self._refresh_quarantine_from_rest()
        due = [pair for pair, item in self._quarantined.items() if item["retry_at"] <= now]
        if due:
            primary_tf = settings.candle_primary_tf

            async def probe(pair: str):
                # Move the deadline first so concurrent refreshes cannot probe twice.
                self._quarantined[pair]["retry_at"] = time.time() + self._quarantine_retry_seconds
                try:
                    primary = await self.rest.get_candles(pair, primary_tf, limit=100)
                    stale_getter = getattr(self.rest, "get_stale_market_timeframes", None)
                    stale_for_pair = (stale_getter() or {}).get(pair, {}) if callable(stale_getter) else {}
                    primary_stale = primary_tf in stale_for_pair
                    if len(primary) >= 50 and not primary_stale:
                        self._quarantined.pop(pair, None)
                        logger.info(
                            f"MARKET_RESTORED {pair} fresh_{primary_tf}={len(primary)}"
                        )
                    else:
                        item = self._quarantined[pair]
                        item["reason"] = (
                            f"retry still unhealthy: {primary_tf}={len(primary)} candles, "
                            f"stale={primary_stale}"
                        )
                        retry_at = datetime.fromtimestamp(item["retry_at"], tz=timezone.utc).isoformat()
                        logger.warning(
                            f"MARKET_QUARANTINE_RETRY_FAILED {pair} reason={item['reason']} "
                            f"next_retry_at={retry_at}"
                        )
                except Exception as exc:
                    item = self._quarantined.get(pair)
                    if item:
                        item["reason"] = f"retry request failed: {type(exc).__name__}: {exc}"
                        retry_at = datetime.fromtimestamp(item["retry_at"], tz=timezone.utc).isoformat()
                        logger.warning(f"MARKET_QUARANTINE_RETRY_ERROR {pair} error={exc} next_retry_at={retry_at}")

            await asyncio.gather(*(probe(pair) for pair in due))

        self._refresh_quarantine_from_rest()
        self._select_active_symbols()

    def display_name(self, pair: str) -> str:
        return self.pair_to_symbol.get(pair, pair.removeprefix("B-").removeprefix("I-").replace("_", "/"))
