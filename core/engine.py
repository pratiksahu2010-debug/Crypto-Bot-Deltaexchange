"""Central signal processing engine — Normal + Momentum, closed-candle driven."""
import asyncio
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config import settings
from utils.logger import logger
from utils.time_utils import utc_iso, utc_now, parse_interval
from core.candle_timing import candle_age_seconds, seconds_until_next_candle_boundary
from utils.memory_monitor import check_memory, safe_gc
from exchange.delta_rest import DeltaExchangeRest
from exchange.symbol_manager import SymbolManager
from exchange.candle_builder import CandleBuilder
from core.scoring_engine import ScoringEngine
from core.cooldown_manager import CooldownManager
from core.watchlist_manager import WatchlistManager
from core.signal_manager import SignalManager
from core.risk_calculator import calculate_risk, is_entry_price_actionable, classify_entry_state
from strategies.normal.strategy_manager import detect_strategy, diagnose_strategy
from strategies.normal.entry_trigger import (
    detect_fresh_trigger, session_vwap_series,
    resolve_symbol_pullback_touch_atr,
)
from strategies.momentum.strategy_manager import detect_momentum_state, diagnose_momentum_state, is_pre_explosion_signal
from strategies.momentum import orb_detector
from strategies.momentum.live_ignition import LiveConfig, LiveTracker
from indicators import (
    ema, rsi, atr, adx, macd, vwap, relative_volume,
    bollinger_bands, bandwidth, keltner_channels, roc,
)
import numpy as np
from indicators.ema import ema_series


# Covers a full UTC day (288 x 5m / 96 x 15m) plus a small history margin.
PRIMARY_CANDLE_HISTORY = 300
VALID_MOMENTUM_STATES = frozenset({
    "EXPLOSION", "SQUEEZE_RELEASE", "PRE_EXPLOSION_SIGNAL", "CONTINUATION",
    "VOLUME_SPIKE", "ORB_PREBREAKOUT", "ORB_BREAKOUT", "ORB_RETEST",
})


def _normal_adx_floor(configured: Any = 25.0) -> float:
    """Keep the required Normal ADX floor at 25 while honoring stricter configuration."""
    try:
        value = float(configured)
    except (TypeError, ValueError, OverflowError):
        value = 25.0
    return max(25.0, value) if np.isfinite(value) else 25.0


def _score_disposition(score: float, signal_threshold: float) -> str:
    """Classify momentum scores without allowing watchlist scores to become alerts."""
    if score >= max(8.0, signal_threshold):
        return "SIGNAL"
    if score >= 5.5:
        return "WATCHLIST"
    return "REJECT"


def _momentum_entry_gates(
    state: str, atr_value: Any, ema_extension_atr: Any,
) -> Dict[str, bool]:
    """Apply only shared validity and anti-chase checks after each detector qualifies."""
    try:
        atr_number = float(atr_value)
        extension = float(ema_extension_atr)
    except (TypeError, ValueError, OverflowError):
        atr_number = extension = float("nan")
    return {
        "momentum_state": state in VALID_MOMENTUM_STATES,
        "anti_chase": bool(
            np.isfinite(atr_number) and atr_number > 0
            and np.isfinite(extension) and 0.0 <= extension <= 2.0
        ),
    }


def _normal_threshold_gates(
    indicators: Dict[str, Any], direction: str, adx_min: float = 20.0,
    strategy: Optional[str] = None,
) -> Dict[str, bool]:
    """Core Normal indicator gates; participation and trend strength are checked separately."""
    if direction not in ("LONG", "SHORT") or not isinstance(indicators, dict):
        return {
            "ema_9_21": False, "vwap_direction": False, "rsi_direction": False,
            "adx_min_25": False, "atr_valid": False,
        }

    def number(key):
        try:
            value = float(indicators.get(key))
        except (TypeError, ValueError, OverflowError):
            return None
        return value if np.isfinite(value) else None

    ema9, ema21 = number("ema9"), number("ema21")
    price, vwap_value = number("close"), number("vwap")
    rsi_value, adx_value, atr_value = number("rsi"), number("adx"), number("atr")
    is_long = direction == "LONG"
    return {
        "ema_9_21": bool(
            ema9 is not None and ema21 is not None
            and (ema9 > ema21 if is_long else ema9 < ema21)
        ),
        "vwap_direction": bool(
            price is not None and vwap_value is not None
            and (price > vwap_value if is_long else price < vwap_value)
        ),
        "rsi_direction": bool(
            rsi_value is not None
            and (rsi_value > 50.0 if is_long else rsi_value < 50.0)
        ),
        "adx_min_25": bool(adx_value is not None and adx_value >= _normal_adx_floor(adx_min)),
        "atr_valid": bool(atr_value is not None and atr_value > 0),
    }


def _normal_extension_gate(indicators: Dict[str, Any], max_extension_atr: float) -> bool:
    """Reject a Normal trigger only when price is overextended from EMA9 in ATR units."""
    try:
        extension = float(indicators.get("ema_extension_atr"))
        limit = float(max_extension_atr)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(
        np.isfinite(extension) and np.isfinite(limit)
        and extension >= 0.0 and limit > 0.0 and extension <= limit
    )


def _normal_relative_volume_gate(
    indicators: Dict[str, Any], direction: str, min_rvol: float
) -> Dict[str, bool]:
    """Use RVOL as the only Normal volume threshold; candle direction is not a gate."""
    try:
        rvol_value = float(indicators.get("rvol"))
        threshold = float(min_rvol)
    except (TypeError, ValueError, OverflowError):
        rvol_value = float("nan")
        threshold = float("nan")
    relative_volume_ok = (
        np.isfinite(rvol_value) and np.isfinite(threshold)
        and threshold > 0 and rvol_value >= threshold
    )
    # Candle-body direction is deliberately NOT a Normal Engine gate.
    # Delta OHLCV has no aggressor-side volume; keep the user's simple
    # threshold model and use RVOL only to decide alert vs watchlist.
    return {
        "relative_volume": bool(relative_volume_ok),
    }


def _macd_hist_history(closes: Any, points: int = 4) -> List[float]:
    """Return recent MACD histogram observations.

    indicators.macd() intentionally returns a single (line, signal, histogram)
    tuple, not a historical series. Recalculate the last few prefixes to obtain
    comparable closed-candle histogram observations for momentum acceleration.
    """
    try:
        values = np.asarray(closes, dtype=float).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        return []

    # MACD needs slow(26) + signal(9) observations.
    if values.size < 35 or points < 3:
        return []

    first_end = max(35, int(values.size) - int(points) + 1)
    history: List[float] = []
    for end in range(first_end, int(values.size) + 1):
        result = macd(values[:end])
        if result is None:
            continue
        try:
            histogram = float(result[2])
        except (TypeError, ValueError, IndexError, OverflowError):
            continue
        if np.isfinite(histogram):
            history.append(histogram)
    return history


def _confirmed_swing_levels(
    high: Any, low: Any, lookback: int = 30, left: int = 2, right: int = 2
) -> tuple[Optional[float], Optional[float]]:
    """Return latest confirmed pivot high/low using completed bars only.

    A pivot is accepted only after the configured number of later completed
    candles exist. If none is present, use the previous seven completed candles.
    """
    highs = np.asarray(high, dtype=float).reshape(-1)
    lows = np.asarray(low, dtype=float).reshape(-1)
    if highs.size != lows.size or highs.size < left + right + 2:
        return None, None

    start = max(0, highs.size - max(lookback, left + right + 2))
    end = highs.size - right
    pivot_high = pivot_low = None
    for idx in range(start + left, end):
        h_window = highs[idx - left:idx + right + 1]
        l_window = lows[idx - left:idx + right + 1]
        if highs[idx] >= np.max(h_window) and np.count_nonzero(highs[idx] == h_window) == 1:
            pivot_high = float(highs[idx])
        if lows[idx] <= np.min(l_window) and np.count_nonzero(lows[idx] == l_window) == 1:
            pivot_low = float(lows[idx])

    if pivot_low is None and highs.size >= 8:
        pivot_low = float(np.min(lows[-8:-1]))
    if pivot_high is None and highs.size >= 8:
        pivot_high = float(np.max(highs[-8:-1]))
    return pivot_high, pivot_low


class SignalEngine:
    def __init__(
        self,
        rest: DeltaExchangeRest,
        symbols: SymbolManager,
        candles: CandleBuilder,
    ):
        self.rest = rest
        self.symbols = symbols
        self.candles = candles
        # Keep stateful components isolated per strategy engine.
        self.normal_scoring = ScoringEngine()
        self.momentum_scoring = ScoringEngine()
        self.cooldowns = {
            "normal": CooldownManager(default_seconds=1800),
            "momentum": CooldownManager(default_seconds=1800),
        }
        self.watchlists = {
            "normal": WatchlistManager(max_entries=50, ttl_minutes=60),
            "momentum": WatchlistManager(max_entries=50, ttl_minutes=60),
        }
        self.signals = {
            "normal": SignalManager(max_active=30, ttl_hours=4),
            "momentum": SignalManager(max_active=30, ttl_hours=4),
        }
        self._scan_lock = asyncio.Lock()
        self._running = True
        self.last_scan_time: Optional[datetime] = None
        self.telegram = None
        self.scan_errors = {"normal": 0, "momentum": 0}
        # Independent candle cursors prevent one strategy engine suppressing the other.
        self._last_processed = {"normal": {}, "momentum": {}}
        # Durable one-shot retest candidates, restored after process restarts.
        self.pending_retests: Dict[str, dict] = {}
        self.pending_retest_checks = 0
        self.pending_retest_failures = 0
        self.telegram_delivery_failures = 0
        self._pending_retest_cursor = 0
        # Per-engine scan diagnostics; mutated only by the event-loop scan tasks.
        self._scan_rejection_stats: Dict[str, Dict[str, Any]] = {}

    def set_telegram(self, tg) -> None:
        self.telegram = tg

    def on_intrabar_candle(self, pair: str, timeframe: str, candle: Dict[str, Any]) -> None:
        """Record early forming-candle momentum as watchlist context only.

        Intrabar candidates are never Telegram signals. The normal closed-candle
        REST scan must independently confirm the final Momentum setup.
        """
        if timeframe != settings.candle_primary_tf:
            return
        arr = self.candles.get_arrays(pair, timeframe)
        if not arr or len(arr.get("close", [])) < 25:
            return
        try:
            close = np.asarray(arr["close"], dtype=float)
            volume = np.asarray(arr["volume"], dtype=float)
            opens = float(candle["open"])
            live_close = float(candle["close"])
            avg_volume = float(np.mean(volume[-21:-1]))
            if avg_volume <= 0 or live_close <= 0:
                return
            rvol_live = float(candle.get("volume", 0.0)) / avg_volume
            short_roc = (live_close / float(close[-4]) - 1.0) * 100.0
            if rvol_live < 1.2 or abs(short_roc) < 0.08:
                return
            direction = "LONG" if live_close > opens and short_roc > 0 else (
                "SHORT" if live_close < opens and short_roc < 0 else None
            )
            if direction is None:
                return
            self.watchlists["momentum"].add({
                "pair": pair, "display": self.symbols.display_name(pair),
                "direction": direction, "engine": "momentum",
                "strategy": "INTRABAR_PRESETUP", "score": 5.5,
                "grade": "WATCHLIST", "price": live_close,
                "indicators": {"live_rvol": round(rvol_live, 3),
                               "live_roc": round(short_roc, 4)},
                "timestamp": utc_iso(utc_now()),
            })
        except (KeyError, TypeError, ValueError, ZeroDivisionError, FloatingPointError):
            return

    # ------------------------------------------------------------------
    # Live (intrabar) Momentum Explosion + ORB
    # ------------------------------------------------------------------
    def _ensure_live_state(self) -> LiveTracker:
        """Lazy init so partially constructed engines (tests) keep working."""
        if getattr(self, "_live", None) is None:
            self._live = LiveTracker(LiveConfig.from_settings(settings))
            self._live_tasks = set()
            self._live_inflight = set()
            self._live_emitted = deque(maxlen=500)
            self._live_last_tick = {}
            self._live_post_entry = {}
            self.live_stats = {"candidates": 0, "emitted": 0, "rejected": 0, "capped": 0}
        return self._live

    def on_live_candle(self, pair: str, timeframe: str, candle: Dict[str, Any]) -> None:
        """WebSocket forming-candle hook: legacy watchlist logic + live trigger.

        Runs at most ~1x/second per pair and never raises into the stream.
        """
        if timeframe != settings.candle_primary_tf:
            return
        try:
            live = self._ensure_live_state()
            # Track post-entry extremes for live signals (outcome bookkeeping).
            try:
                price = float(candle["close"])
                for (p, _d), ext in self._live_post_entry.items():
                    if p == pair:
                        ext[0], ext[1] = max(ext[0], price), min(ext[1], price)
            except (KeyError, TypeError, ValueError):
                pass
            now_mono = time.monotonic()
            if now_mono - self._live_last_tick.get(pair, 0.0) < 0.8:
                return
            self._live_last_tick[pair] = now_mono
            self.on_intrabar_candle(pair, timeframe, candle)
            mode = str(getattr(settings, "live_ignition_mode", "live")).lower()
            if mode not in ("live", "shadow") or pair not in self.symbols.momentum_symbols:
                return
            interval_ms = int(parse_interval(timeframe).total_seconds() * 1000)
            now_ms = int(utc_now().timestamp() * 1000)
            for res in live.evaluate(
                pair, candle, interval_ms, now_ms,
                lambda: self.candles.get_arrays(pair, timeframe),
            ):
                res["detected_at_ms"] = now_ms
                self._handle_live_candidate(pair, res, mode)
        except Exception as exc:  # the stream must never be broken by this hook
            logger.debug(f"LIVE_IGNITION_CALLBACK_ERROR {pair}: {exc}")

    def _handle_live_candidate(self, pair: str, res: Dict[str, Any], mode: str) -> None:
        engine, direction, candle_ms = "momentum", res["direction"], res["candle_time_ms"]
        live = self._ensure_live_state()
        if self.cooldowns[engine].is_cooling(pair, engine, direction):
            return
        active = self.signals[engine].get(pair, engine, direction)
        if active and active.get("status") == "active":
            return
        if not live.first_attempt(pair, direction, candle_ms):
            return
        score = res["score_result"]
        self.live_stats["candidates"] += 1
        summary = (
            f"pair={pair} direction={direction} trigger={res['state']} score={score.total} "
            f"elapsed_s={res['elapsed_seconds']} move_atr={res['move_atr']} "
            f"volume_pace={res['volume_pace']} price={res['trigger_price']}"
        )
        if mode == "shadow":
            logger.info(f"LIVE_IGNITION_SHADOW {summary}")
            return
        hour_ago = time.time() - 3600.0
        while self._live_emitted and self._live_emitted[0] < hour_ago:
            self._live_emitted.popleft()
        if len(self._live_emitted) >= int(settings.live_max_signals_per_hour):
            self.live_stats["capped"] += 1
            logger.info(f"LIVE_IGNITION_RATE_CAPPED {summary}")
            return
        key = (pair, direction)
        if key in self._live_inflight:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._live_inflight.add(key)
        logger.info(f"LIVE_IGNITION_TRIGGER {summary}")
        task = loop.create_task(self._emit_live_signal(pair, res))
        self._live_tasks.add(task)

        def _done(t, key=key):
            self._live_tasks.discard(t)
            self._live_inflight.discard(key)
            if not t.cancelled() and t.exception():
                logger.error(f"LIVE_IGNITION_EMIT_ERROR {key}: {t.exception()}")

        task.add_done_callback(_done)

    async def _emit_live_signal(self, pair: str, res: Dict[str, Any]) -> None:
        """Validate, claim and queue one live signal (same order as the closed path)."""
        engine, direction = "momentum", res["direction"]
        candle_ts, atr_v = int(res["candle_time_ms"]), float(res["atr"])
        sign = 1.0 if direction == "LONG" else -1.0
        max_age = 20.0

        def reject(reason: str) -> None:
            self.live_stats["rejected"] += 1
            logger.info(f"LIVE_IGNITION_REJECT pair={pair} direction={direction} reason={reason}")

        def age_s() -> float:
            return (utc_now().timestamp() * 1000 - res["detected_at_ms"]) / 1000.0

        try:
            live_price = await self.rest.get_last_price(pair)
        except Exception as exc:
            logger.warning(f"LIVE_ENTRY_PRICE_CHECK_FAILED {pair}: {exc}")
            return reject("ltp_unavailable")
        if live_price is None:
            return reject("ltp_unavailable")
        drift = sign * (live_price - res["trigger_price"]) / atr_v
        if drift < -0.25 or drift > 0.40:
            return reject(f"price_drift_{drift:.2f}atr")
        if res["state"] == "LIVE_ORB" and sign * (live_price - res["breakout_level"]) < 0.5 * 0.15 * atr_v:
            return reject("back_inside_range")
        risk = calculate_risk(
            live_price, atr_v, direction,
            swing_low=res.get("swing_low"), swing_high=res.get("swing_high"),
        )
        if not risk:
            return reject("risk_levels_unavailable")
        if age_s() > max_age:
            return reject("stale_after_checks")

        score = res["score_result"]
        orb = res["state"] == "LIVE_ORB"
        indicators = {
            "ema9": res.get("ema9"), "ema21": res.get("ema21"), "ema50": res.get("ema50"),
            "atr_pct": res["atr_pct"], "rvol": res["volume_pace"], "rvol_basis": "time_normalised",
            "live_move_atr": res["move_atr"], "live_volume_ratio": res["volume_ratio"],
            "live_elapsed_seconds": res["elapsed_seconds"],
            "structure_break": res.get("structure_break"),
            "squeeze_release": res.get("compression_recent"),
        }
        if orb:
            indicators.update({
                "orb_session": res["session"], "orb_session_timezone": res["session_timezone"],
                "orb_session_open_local": res["session_open_local"],
                "orb_range_high": res["range_high"], "orb_range_low": res["range_low"],
                "orb_breakout_level": res["breakout_level"],
                "orb_breakout_extension_atr": res["breakout_extension_atr"],
                "orb_range_width_atr": res["range_width_atr"],
            })
        else:
            indicators["live_speed_atr_per_min"] = res["speed_atr_per_min"]
        now_iso = utc_iso(utc_now())
        signal = {
            "pair": pair, "display": self.symbols.display_name(pair),
            "direction": direction, "engine": engine, "strategy": res["strategy"],
            "entry_type": "LIVE_INTRABAR", "score": score.total, "grade": score.grade,
            "price": live_price,
            "indicators": {k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in indicators.items() if v is not None},
            "components": [{"name": c.name, "points": c.points, "max": c.max_points,
                            "expl": c.explanation} for c in score.components],
            "timestamp": now_iso, "candle_time_ms": candle_ts,
            "candle_close_time_ms": res["candle_close_time_ms"],
            "signal_age_seconds": round(age_s(), 1), "confirm_candle_time_ms": None,
            "live_price": live_price, "entry_checked_at": now_iso,
            "risk": {
                "entry": risk.entry, "entry_zone": [risk.entry_zone_low, risk.entry_zone_high],
                "stop_loss": risk.stop_loss, "tp1": risk.tp1, "tp2": risk.tp2, "tp3": risk.tp3,
                "risk_pct": risk.risk_pct, "rr": [risk.rr1, risk.rr2, risk.rr3],
                "invalidation": risk.invalidation,
            },
        }
        if not self.signals[engine].add(signal):
            return reject("active_signal_exists")
        await self._persist_evicted_signals(engine)
        try:
            from database.repositories import claim_signal_once
            claimed = await claim_signal_once(signal, candle_ts, ttl_hours=4)
        except Exception as exc:
            logger.error(f"Persistent signal deduplication unavailable: {exc}")
            claimed = False
        if not claimed:
            self.signals[engine].invalidate(pair, engine, direction)
            return reject("duplicate_or_claim_unavailable")

        async def release() -> None:
            try:
                from database.repositories import release_signal_claim
                await release_signal_claim(signal, candle_ts)
            except Exception as exc:
                logger.error(f"SIGNAL_CLAIM_RELEASE_FAILED {pair}: {exc}")
            self.signals[engine].invalidate(pair, engine, direction)

        if self.telegram:
            try:
                delivery_price = await self.rest.get_last_price(pair)
            except Exception:
                delivery_price = None
            if (
                delivery_price is None or age_s() > max_age
                or not -0.25 <= sign * (delivery_price - res["trigger_price"]) / atr_v <= 0.40
            ):
                await release()
                return reject("stale_at_delivery")
            signal["live_price"] = delivery_price
            signal["entry_checked_at"] = utc_iso(utc_now())
        signal["signal_age_seconds"] = round(age_s(), 1)
        signal["timestamp"] = utc_iso(utc_now())
        if not await self._queue_telegram_signal(signal):
            self.telegram_delivery_failures += 1
            await release()
            return reject("not_queued")

        self.cooldowns[engine].set(pair, engine, direction)
        self._live_emitted.append(time.time())
        self.live_stats["emitted"] += 1
        self._live_post_entry[(pair, direction)] = [risk.entry, risk.entry]
        if orb:
            self._live.orb_fired.add((pair, res["session"], res["session_date"], direction))
        logger.info(
            f"LIVE_SIGNAL_QUEUED {score.grade} {direction} {pair} strategy={res['strategy']} "
            f"score={score.total} age_seconds={signal['signal_age_seconds']}"
        )
        persisted = self.signals[engine].get(pair, engine, direction)
        if persisted:
            try:
                from database.repositories import save_signal
                if not await save_signal(persisted):
                    logger.error("ACTIVE_SIGNAL_PERSIST_FAILED %s engine=%s", pair, engine)
            except Exception as exc:
                logger.error("ACTIVE_SIGNAL_PERSIST_ERROR %s: %s", pair, exc)

    async def restore_active_signals(self) -> int:
        """Rehydrate active signal lifecycle state from the configured database."""
        try:
            from database.repositories import load_active_signals
            records = await load_active_signals(max_age_hours=4)
        except Exception as exc:
            logger.error(f"Active signal restore failed: {exc}")
            return 0
        restored = 0
        for signal in records:
            engine = signal.get("engine")
            if engine in self.signals and self.signals[engine].restore(signal):
                restored += 1
                try:
                    created = datetime.fromisoformat(
                        str(signal.get("created_at", "")).replace("Z", "+00:00")
                    )
                    elapsed = max(0.0, (utc_now() - created).total_seconds())
                    remaining = int(self.cooldowns[engine].default_seconds - elapsed)
                    if remaining > 0:
                        self.cooldowns[engine].set(
                            signal["pair"], engine, signal["direction"], seconds=remaining
                        )
                except (TypeError, ValueError):
                    logger.warning(
                        f"Could not restore cooldown timestamp for {signal.get('pair')}"
                    )
        for engine_name in self.signals:
            await self._persist_evicted_signals(engine_name)
        logger.info(f"ACTIVE_SIGNALS_RESTORED count={restored}")
        return restored


    async def restore_pending_retests(self) -> int:
        """Restore only unexpired persisted retest candidates."""
        try:
            from database.repositories import load_pending_retests
            records = await load_pending_retests(limit=500)
            now = utc_now().timestamp()
            restored_pair_engine = set()
            for key, pending in records:  # repository returns newest first
                if float(pending.get("expires_at", 0)) <= now:
                    continue
                signal = pending.get("signal") or {}
                pair, engine, direction = signal.get("pair"), signal.get("engine"), signal.get("direction")
                if not pair or not engine or direction not in ("LONG", "SHORT"):
                    continue
                identity = (pair, engine)
                if identity in restored_pair_engine:
                    # Historical database rows may contain opposite or duplicate
                    # direction candidates from before the exclusivity policy.
                    try:
                        from database.repositories import delete_pending_retest
                        await delete_pending_retest(key)
                    except Exception as cleanup_exc:
                        logger.warning(f"Pending retest restore cleanup failed {key}: {cleanup_exc}")
                    continue
                restored_pair_engine.add(identity)
                self.pending_retests[key] = pending
            logger.info(f"PENDING_RETESTS_RESTORED count={len(self.pending_retests)}")
            return len(self.pending_retests)
        except Exception as exc:
            logger.error(f"Pending retest restore failed: {exc}")
            return 0

    async def pending_retest_monitor_loop(self, interval_seconds: float = 2.0) -> None:
        """Rate-limited fresh-LTP monitor; bounded to five checks per interval."""
        while self._running:
            try:
                all_keys = list(self.pending_retests)
                if all_keys:
                    start = self._pending_retest_cursor % len(all_keys)
                    keys = (all_keys[start:] + all_keys[:start])[:5]
                    self._pending_retest_cursor = (start + len(keys)) % max(1, len(all_keys))
                else:
                    keys = []
                for key in keys:
                    pending = self.pending_retests.get(key)
                    if not pending:
                        continue
                    signal = pending.get("signal", {})
                    await self._check_pending_retest(signal.get("pair", ""), signal.get("engine", ""))
                    await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.pending_retest_failures += 1
                logger.warning(f"Pending retest monitor cycle failed: {exc}")
            await asyncio.sleep(max(1.0, min(float(interval_seconds), 10.0)))

    async def _wait_for_next_primary_boundary(self) -> None:
        interval_ms = int(parse_interval(settings.candle_primary_tf).total_seconds() * 1000)
        grace_ms = max(0, int(settings.candle_close_grace_seconds)) * 1000
        now_ms = int(utc_now().timestamp() * 1000)
        delay = seconds_until_next_candle_boundary(now_ms, interval_ms, grace_ms)
        next_scan_utc = datetime.fromtimestamp(
            (now_ms / 1000.0) + delay, tz=timezone.utc
        ).isoformat()
        logger.info(
            f"SCAN_SCHEDULE mode=completed_candle_boundary "
            f"timeframe={settings.candle_primary_tf} "
            f"grace_seconds={grace_ms / 1000:.0f} wait_seconds={delay:.1f} "
            f"next_scan_utc={next_scan_utc} "
            f"max_signal_age_seconds={settings.max_signal_age_seconds}"
        )
        # Keep the event loop observable during a long boundary wait. This
        # also makes Render logs useful even when no candles/signals are emitted.
        remaining = delay
        while remaining > 0:
            chunk = min(30.0, remaining)
            await asyncio.sleep(chunk)
            remaining -= chunk
            state = getattr(getattr(getattr(self, "_app", None), "state", None), "bot_state", None)
            if isinstance(state, dict):
                state["scan_stage"] = "waiting_for_candle_boundary"
                state["scan_progress_updated_at"] = utc_iso(utc_now())
            if remaining > 0:
                logger.info(
                    f"SCAN_HEARTBEAT stage=waiting_for_candle_boundary "
                    f"remaining_seconds={remaining:.1f}"
                )

    async def _sync_websocket_universe(self, app: Any) -> None:
        app_state = getattr(app, "state", None) if app is not None else None
        if app_state is None:
            return
        websocket = getattr(app_state, "websocket", None)
        updater = getattr(websocket, "update_symbols", None)
        if callable(updater) and settings.enable_websocket:
            await updater(set(self.symbols.normal_symbols + self.symbols.momentum_symbols))

    async def run_scan_loop(self, app: Any) -> None:
        """Run Normal and Momentum on independent candle-boundary schedules.

        A long Momentum universe scan must never hold up the next Normal scan.
        Each engine owns its own loop and cursor while sharing the bounded REST
        client and the same completed-candle/freshness rules.
        """
        # Expose preflight progress: Render can otherwise look idle while the
        # first market refresh or WebSocket subscription is still in progress.
        self._app = app
        state = app.state.bot_state
        state.update({
            "scan_task_status": "starting",
            "scan_stage": "universe_refresh",
            "scan_stage_started_at": utc_iso(utc_now()),
            "scan_progress_engine": None,
            "scan_progress_symbol": None,
            "scan_progress_completed": 0,
            "scan_progress_total": 0,
            "scan_progress_updated_at": utc_iso(utc_now()),
        })
        logger.info("SCAN_STARTUP_STAGE stage=universe_refresh")
        await self.symbols.refresh_universes()

        state["scan_stage"] = "websocket_sync"
        state["scan_progress_updated_at"] = utc_iso(utc_now())
        logger.info(
            "SCAN_STARTUP_STAGE stage=websocket_sync "
            f"normal_symbols={len(self.symbols.normal_symbols)} "
            f"momentum_symbols={len(self.symbols.momentum_symbols)}"
        )
        await self._sync_websocket_universe(app)

        # Wait for the next usable completed-candle boundary after preflight.
        state["scan_stage"] = "waiting_for_candle_boundary"
        state["scan_progress_updated_at"] = utc_iso(utc_now())
        logger.info("SCAN_STARTUP_STAGE stage=waiting_for_candle_boundary")
        await self._wait_for_next_primary_boundary()

        state["scan_task_status"] = "running"
        state["scan_stage"] = "scanning"
        state["scan_progress_updated_at"] = utc_iso(utc_now())
        logger.info("SCAN_READY closed_candle_scanning=true")

        tasks = []
        if settings.enable_normal_engine:
            tasks.append(asyncio.create_task(self._run_engine_loop(app, "normal")))
        if settings.enable_momentum_engine:
            tasks.append(asyncio.create_task(self._run_engine_loop(app, "momentum")))
        if tasks:
            await asyncio.gather(*tasks)

    async def _run_engine_loop(self, app: Any, engine: str) -> None:
        """Scan one strategy engine without waiting for the other engine."""
        if engine not in ("normal", "momentum"):
            raise ValueError(f"Unknown strategy engine: {engine}")

        while self._running:
            try:
                state = getattr(getattr(self, "_app", None), "state", None)
                state = getattr(state, "bot_state", None)
                if isinstance(state, dict):
                    state[f"{engine}_engine"] = "scanning"
                    state["scan_stage"] = f"{engine}_scan"
                    state["scan_progress_engine"] = engine
                    state["scan_progress_symbol"] = None
                    state["scan_progress_completed"] = 0
                    state["scan_progress_total"] = len(
                        self.symbols.normal_symbols if engine == "normal"
                        else self.symbols.momentum_symbols
                    )
                    state["scan_progress_updated_at"] = utc_iso(utc_now())
                # Apply stale-feed quarantine before selecting the next scan's
                # symbols. The other engine may have observed the second stale
                # response during its previous cycle.
                self.symbols._refresh_quarantine_from_rest()
                self.symbols._select_active_symbols()
                await self._sync_websocket_universe(app)

                start = utc_now()
                if engine == "normal":
                    await self._scan_normal()
                elif engine == "momentum":
                    await self._scan_momentum()

                # Universe refresh is serialized through the Normal loop; if
                # Normal is disabled, Momentum owns refresh responsibility.
                if engine == "normal" or (engine == "momentum" and not settings.enable_normal_engine):
                    await self.symbols.refresh_universes()
                    await self._sync_websocket_universe(app)

                finished = utc_now()
                duration_ms = int((finished - start).total_seconds() * 1000)
                interval_seconds = parse_interval(settings.candle_primary_tf).total_seconds()
                if duration_ms > interval_seconds * 1000:
                    logger.warning(
                        f"SCAN_CYCLE_OVERRUN engine={engine} duration_ms={duration_ms} "
                        f"candle_interval_seconds={interval_seconds:.0f} "
                        f"late_symbols_may_be_rejected=true"
                    )
                self.last_scan_time = finished
                state = app.state.bot_state
                state["last_scan"] = utc_iso(finished)
                state[f"last_scan_{engine}"] = utc_iso(finished)
                state[f"last_scan_duration_ms_{engine}"] = duration_ms
                state["last_scan_duration_ms"] = duration_ms
                state["scan_count"] = state.get("scan_count", 0) + 1
                state["active_signals"] = sum(manager.count() for manager in self.signals.values())
                state["watchlist_count"] = sum(manager.count() for manager in self.watchlists.values())
                state["scan_errors_normal"] = self.scan_errors["normal"]
                state["scan_errors_momentum"] = self.scan_errors["momentum"]
                state["valid_symbols_normal"] = len(self.symbols.normal_symbols)
                state["valid_symbols_momentum"] = len(self.symbols.momentum_symbols)
                state["eligible_symbols"] = len(self.symbols.valid_pairs)
                state["skipped_symbols"] = len(self.symbols.skipped_reasons)
                state["normal_engine"] = (
                    "running" if settings.enable_normal_engine and self.symbols.normal_symbols
                    else "no_symbols" if settings.enable_normal_engine else "disabled"
                )
                state["momentum_engine"] = (
                    "running" if settings.enable_momentum_engine and self.symbols.momentum_symbols
                    else "no_symbols" if settings.enable_momentum_engine else "disabled"
                )
                state["last_market_update"] = utc_iso(finished)
                websocket = getattr(app.state, "websocket", None)
                if websocket is not None:
                    state["websocket_status"] = websocket.status
                    state["ws_reconnects"] = websocket.reconnect_count
                    state["ws_price_feed_age_seconds"] = websocket.price_feed_age_seconds
                state["uptime_seconds"] = int(
                    (finished - app.state.start_time).total_seconds()
                )

                if engine == "normal" or not settings.enable_normal_engine:
                    mem = check_memory(force_gc=True)
                    if mem["warning"]:
                        safe_gc()

                await self._wait_for_next_primary_boundary()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.exception(f"{engine.title()} scan loop error: {exc}")
                await self._wait_for_next_primary_boundary()

    async def _scan_pairs(self, pairs: List[str], engine: str) -> None:
        """Scan one engine with bounded concurrency matching the REST semaphore."""
        started = asyncio.get_running_loop().time()
        errors_before = self.scan_errors[engine]

        cycle_started_utc = utc_iso(utc_now())
        symbol_durations = []
        state = getattr(getattr(getattr(self, "_app", None), "state", None), "bot_state", None)
        completed = 0
        last_progress_log = started
        if isinstance(state, dict):
            state["scan_progress_engine"] = engine
            state["scan_progress_total"] = len(pairs)
            state["scan_progress_completed"] = 0
            state["scan_progress_symbol"] = None
            state["scan_progress_updated_at"] = utc_iso(utc_now())
        self._scan_rejection_stats[engine] = {
            "evaluated": 0, "signal": 0, "watchlist": 0, "reject": 0,
            "reasons": {},
        }
        logger.info(
            f"SCAN_CYCLE_START engine={engine} symbols_submitted={len(pairs)} "
            f"started_at_utc={cycle_started_utc} "
            f"max_signal_age_seconds={settings.max_signal_age_seconds}"
        )

        async def process(pair: str) -> None:
            nonlocal completed, last_progress_log
            symbol_started = asyncio.get_running_loop().time()
            if isinstance(state, dict):
                state["scan_progress_symbol"] = pair
                state["scan_progress_updated_at"] = utc_iso(utc_now())
            outcome = "ok"
            try:
                await self._process_pair(pair, engine=engine)
            except Exception as exc:
                outcome = "error"
                self.scan_errors[engine] += 1
                logger.exception(f"{engine.title()} scan failed for {pair}: {exc}")
            finally:
                elapsed_ms = int((asyncio.get_running_loop().time() - symbol_started) * 1000)
                symbol_durations.append(elapsed_ms)
                completed += 1
                now_mono = asyncio.get_running_loop().time()
                if isinstance(state, dict):
                    state["scan_progress_completed"] = completed
                    state["scan_progress_symbol"] = pair
                    state["scan_progress_updated_at"] = utc_iso(utc_now())
                logger.info(
                    f"SCAN_SYMBOL_TIMING engine={engine} pair={pair} "
                    f"duration_ms={elapsed_ms} outcome={outcome}"
                )
                if completed == len(pairs) or completed % 10 == 0 or now_mono - last_progress_log >= 30:
                    last_progress_log = now_mono
                    logger.info(
                        f"SCAN_PROGRESS engine={engine} completed={completed}/{len(pairs)} "
                        f"current_symbol={pair} elapsed_seconds={now_mono - started:.1f}"
                    )

        # DeltaExchangeRest applies a five-request semaphore. Scheduling the symbol
        # tasks concurrently lets that limit work instead of serializing markets
        # and allowing later signals to age out before evaluation.
        await asyncio.gather(*(process(pair) for pair in pairs))

        duration_ms = int((asyncio.get_running_loop().time() - started) * 1000)
        active = len(self.signals[engine].list_active())
        watching = self.watchlists[engine].count()
        max_symbol_ms = max(symbol_durations, default=0)
        avg_symbol_ms = round(sum(symbol_durations) / len(symbol_durations), 1) if symbol_durations else 0
        stats = self._scan_rejection_stats.pop(engine, {
            "evaluated": 0, "signal": 0, "watchlist": 0, "reject": 0, "reasons": {},
        })
        reason_summary = ",".join(
            f"{name}:{count}" for name, count in sorted(stats["reasons"].items())
        ) or "none"
        logger.info(
            f"SCAN_REJECTION_SUMMARY engine={engine} evaluated={stats['evaluated']} "
            f"signal={stats['signal']} watchlist={stats['watchlist']} "
            f"rejected={stats['reject']} reasons=[{reason_summary}]"
        )
        logger.info(
            f"SCAN_CYCLE_SUMMARY engine={engine} symbols_submitted={len(pairs)} "
            f"symbols_finished={len(symbol_durations)} "
            f"scan_errors={self.scan_errors[engine]} "
            f"errors_this_cycle={self.scan_errors[engine] - errors_before} "
            f"active_signals={active} watchlist={watching} duration_ms={duration_ms} "
            f"max_symbol_duration_ms={max_symbol_ms} "
            f"avg_symbol_duration_ms={avg_symbol_ms} "
            f"started_at_utc={cycle_started_utc} finished_at_utc={utc_iso(utc_now())}"
        )

    async def _scan_normal(self) -> None:
        await self._scan_pairs(self.symbols.normal_symbols, "normal")

    async def _scan_momentum(self) -> None:
        await self._scan_pairs(self.symbols.momentum_symbols, "momentum")

    async def _process_pair(self, pair: str, engine: str) -> None:
        pair_started = asyncio.get_running_loop().time()
        # Normal entries must be fresh: tighter completed-candle age limit than Momentum.
        max_signal_age = (
            settings.normal_max_signal_age_seconds if engine == "normal"
            else settings.max_signal_age_seconds
        )
        # Refresh primary TF candles
        primary = settings.candle_primary_tf
        candle_fetch_started = asyncio.get_running_loop().time()
        data = await self.rest.get_candles(pair, primary, limit=PRIMARY_CANDLE_HISTORY)
        candle_fetch_ms = (asyncio.get_running_loop().time() - candle_fetch_started) * 1000.0
        logger.info(f"SIGNAL_STAGE pair={pair} engine={engine} stage=primary_candle_fetch duration_ms={candle_fetch_ms:.1f}")
        if len(data) < 50:
            return
        self.candles.seed(pair, primary, data)
        arr = self.candles.get_arrays(pair, primary)
        if not arr or len(arr["close"]) < 50:
            return

        close = arr["close"]
        high = arr["high"]
        low = arr["low"]
        volume = arr["volume"]
        opens = arr["open"]
        # A candle timestamp is its opening time; REST client has already removed
        # unfinished bars. Process each completed bar once per engine.
        if engine not in ("normal", "momentum"):
            raise ValueError(f"Unknown strategy engine: {engine}")
        candle_ts = int(arr["time"][-1])
        primary_interval_ms = int(parse_interval(primary).total_seconds() * 1000)
        candle_close_ms = candle_ts + primary_interval_ms
        # Update active signal outcomes on every fresh closed candle before deciding
        # whether this engine has already evaluated the candle.
        await self._update_signal_outcomes(pair, engine, float(high[-1]), float(low[-1]), candle_ts)
        await self._check_pending_retest(pair, engine)
        if self._last_processed.setdefault(engine, {}).get(pair) == candle_ts:
            return
        initial_age = candle_age_seconds(int(utc_now().timestamp() * 1000), candle_ts, primary_interval_ms)
        if initial_age < 0 or initial_age > max_signal_age:
            self._last_processed[engine][pair] = candle_ts
            logger.info(
                f"STALE_CANDLE_SKIP {pair} engine={engine} candle_open_ms={candle_ts} "
                f"candle_close_ms={candle_close_ms} age_seconds={initial_age:.1f} "
                f"max_age_seconds={max_signal_age}"
            )
            return

        # Indicators
        indicator_started = asyncio.get_running_loop().time()
        ema9 = ema(close, 9)
        ema21 = ema(close, 21)
        ema50 = ema(close, 50)
        rsi_v = rsi(close, 14)
        atr_v = atr(high, low, close, 14)
        adx_res = adx(high, low, close, 14)
        adx_v = adx_res[0] if adx_res is not None else None
        plus_di = adx_res[1] if adx_res is not None else None
        minus_di = adx_res[2] if adx_res is not None else None
        # MACD is calculated only for Momentum; Normal decisions do not depend on it.
        macd_res = macd(close) if engine == "momentum" else None
        macd_hist = macd_res[2] if macd_res is not None else None
        # UTC-session anchored VWAP; avoids an arbitrary rolling 100-bar reset.
        day_start = (int(arr["time"][-1]) // 86_400_000) * 86_400_000
        day_idx = int(np.searchsorted(arr["time"], day_start, side="left"))
        vwap_v = vwap(high[day_idx:], low[day_idx:], close[day_idx:], volume[day_idx:], period=None)
        rvol = relative_volume(volume, 20)
        atr_pct = (atr_v / close[-1] * 100) if atr_v and close[-1] else None
        adx_series = None
        adx_rising = False
        if len(close) >= 3:
            from indicators.adx import adx_series as _adx_series
            _a = _adx_series(high, low, close, 14)
            adx_series = _a[0] if _a is not None else None
            adx_rising = bool(adx_series is not None and np.isfinite(adx_series[-1]) and np.isfinite(adx_series[-2]) and adx_series[-1] > adx_series[-2])

        # Signal qualification uses completed primary-timeframe candles only.
        # Higher-timeframe candles are not fetched or used as a signal dependency.
        confirm_candle_time_ms = None

        # Market structure: last close must break the prior confirmed swing.
        structure_ok = False
        resistance = support = None
        if len(close) >= 22:
            lookback_high = high[-21:-1]
            lookback_low = low[-21:-1]
            resistance = float(np.max(lookback_high)) if lookback_high.size > 0 else None
            support = float(np.min(lookback_low)) if lookback_low.size > 0 else None
            structure_ok = bool(resistance is not None and support is not None and
                               (close[-1] > resistance or close[-1] < support))

        indicators = {
            # Direction is resolved after ORB and ROC evaluation below.
            "direction": None,
            "close": close[-1],
            "ema9": ema9,
            "ema21": ema21,
            "ema50": ema50,
            "rsi": rsi_v,
            "atr": atr_v,
            "atr_pct": atr_pct,
            "adx": adx_v,
            "macd_hist": macd_hist,
            "vwap": vwap_v,
            "rvol": rvol,
            "high": float(high[-1]), "low": float(low[-1]),
            "resistance": resistance, "support": support,
            "plus_di": plus_di, "minus_di": minus_di,
            "adx_rising": adx_rising,
            "rsi_prev": rsi(close[:-1],14),
        }

        # Evaluate independent global session ORBs on completed 5m Futures bars.
        # IANA time zones in orb_detector handle UK/US daylight-saving changes.
        orb_result = {}
        if engine == "momentum" and settings.orb_enabled:
            try:
                orb_data = await self.rest.get_candles(pair, "5m", limit=PRIMARY_CANDLE_HISTORY)
                if len(orb_data) >= 25:
                    orb_arr = {name: np.asarray([row[name] for row in orb_data], dtype=float)
                               for name in ("open", "high", "low", "close", "volume")}
                    orb_arr["time"] = np.asarray([row["time"] for row in orb_data], dtype=np.int64)
                    orb_atr = atr(orb_arr["high"], orb_arr["low"], orb_arr["close"], 14)
                    orb_avg_volume = float(np.mean(orb_arr["volume"][-21:-1]))
                    orb_rvol = float(orb_arr["volume"][-1] / orb_avg_volume) if orb_avg_volume > 0 else None
                    session_names = []
                    session_names.append("utc")
                    session_names.append("crypto_peak")
                    if settings.orb_london_enabled:
                        session_names.append("london")
                    if settings.orb_new_york_enabled:
                        session_names.append("new_york")
                    if getattr(settings, "orb_asia_enabled", False):
                        session_names.append("asia")
                    if getattr(settings, "orb_sydney_enabled", False):
                        session_names.append("sydney")
                    if getattr(settings, "orb_cme_enabled", True):
                        session_names.append("cme")
                    results = [
                        orb_detector.evaluate(
                            orb_arr, orb_atr, orb_rvol, session=session_name,
                            min_rvol=settings.orb_min_rvol,
                            max_extension_atr=settings.orb_max_extension_atr,
                            active_minutes=settings.orb_active_minutes,
                            adx_value=adx_v,
                            plus_di=plus_di,
                            minus_di=minus_di,
                            adx_rising=adx_rising,
                            adx_min=settings.orb_adx_min,
                        )
                        for session_name in session_names
                    ]
                    valid = [item for item in results
                             if item.get("state") in ("ORB_PREBREAKOUT", "ORB_BREAKOUT", "ORB_RETEST")]
                    if valid:
                        # If both qualify on one candle, prefer the newer session's range.
                        orb_result = max(valid, key=lambda item: item.get("session_start_utc_ms", 0))
                    elif results:
                        orb_result = max(results, key=lambda item: item.get("session_start_utc_ms", 0))
            except Exception as exc:
                logger.warning(f"ORB_SESSION_DATA_UNAVAILABLE {pair}: {exc}")
        indicators["orb_state"] = orb_result.get("state")
        indicators["orb_session"] = orb_result.get("session")
        indicators["orb_session_timezone"] = orb_result.get("session_timezone")
        indicators["orb_session_open_local"] = orb_result.get("session_open_local")
        indicators["orb_range_start_local"] = orb_result.get("range_start_local")
        indicators["orb_range_end_local"] = orb_result.get("range_end_local")
        indicators["orb_range_high"] = orb_result.get("range_high")
        indicators["orb_range_low"] = orb_result.get("range_low")
        indicators["orb_breakout_level"] = orb_result.get("breakout_level")
        indicators["orb_breakout_extension_atr"] = orb_result.get("breakout_extension_atr")
        indicators["orb_range_width_atr"] = orb_result.get("range_width_atr")
        indicators["orb_trend_confirmed"] = orb_result.get("trend_confirmed", False)
        indicators["orb_di_aligned"] = orb_result.get("di_aligned")
        indicators["orb_adx_rising"] = orb_result.get("adx_rising", False)
        indicators["orb_wick_quality"] = orb_result.get("wick_quality", False)
        indicators["orb_indicator_confirmation"] = orb_result.get("indicator_confirmation")
        orb_confirmation = orb_result.get("indicator_confirmation")
        indicators["orb_vwap"] = (
            orb_confirmation.get("vwap") if isinstance(orb_confirmation, dict) else None
        )

        # Determine direction from EMA9/EMA21 only. VWAP is evaluated as its
        # own mandatory gate below; candle direction/body is not a gate.
        direction = None
        if engine == "normal":
            if ema9 is not None and ema21 is not None:
                if ema9 > ema21:
                    direction = "LONG"
                elif ema9 < ema21:
                    direction = "SHORT"
        else:
            # Momentum may lead EMA/MACD confirmation: short ROC seeds the
            # directional candidate. MACD remains a trade-alert gate, while
            # PRE_EXPLOSION can be observed before MACD fully turns.
            short_roc = ((float(close[-1]) / float(close[-4]) - 1.0) * 100.0
                         if len(close) >= 4 and float(close[-4]) > 0 else 0.0)
            if orb_result.get("state") in ("ORB_PREBREAKOUT", "ORB_BREAKOUT", "ORB_RETEST"):
                direction = orb_result.get("direction")
            elif orb_result.get("state") == "ORB_FAILED_BREAKOUT":
                # Do not let a failed ORB breakout immediately re-enter as a
                # same-direction Momentum Explosion on the same completed bar.
                failed_direction = orb_result.get("direction")
                if short_roc > 0 and failed_direction != "LONG":
                    direction = "LONG"
                elif short_roc < 0 and failed_direction != "SHORT":
                    direction = "SHORT"
            elif short_roc > 0:
                # ROC can seed an early watch candidate; MACD remains a hard
                # confirmation for Momentum trade alerts later in the pipeline.
                direction = "LONG"
            elif short_roc < 0:
                direction = "SHORT"

        if not direction:
            if engine == "normal":
                self.watchlists["normal"].remove_pair(pair, "normal")
            logger.info(
                f"CANDLE_EVAL pair={pair} engine={engine} direction=NONE "
                f"decision=REJECT reason=direction_alignment "
                f"candle_open_ms={candle_ts} candle_close_ms={candle_close_ms} "
                f"close={float(close[-1]):.8g} ema9={ema9} ema21={ema21} vwap={vwap_v} "
            )
            return
        # Direction is assigned above after the ORB result and momentum ROC
        # are known. Populate it now; the initial indicators snapshot is built
        # earlier and must not read this local before assignment.
        indicators["direction"] = direction
        # MACD histogram direction is an indicator confirmation for Momentum;
        # Normal never depends on MACD and no candle-volume proxy is used.
        indicators["macd_direction_ok"] = bool(
            macd_hist is not None and (
                (direction == "LONG" and macd_hist > 0) or
                (direction == "SHORT" and macd_hist < 0)
            )
        )
        indicators["structure_direction_ok"] = bool(
            (direction == "LONG" and close[-1] > (resistance or float("inf"))) or
            (direction == "SHORT" and close[-1] < (support or -float("inf"))))
        indicators["di_direction_ok"] = bool(
            plus_di is not None and minus_di is not None and
            ((direction == "LONG" and plus_di > minus_di) or
             (direction == "SHORT" and minus_di > plus_di)))
        extension = abs(float(close[-1])-float(ema9))/atr_v if atr_v and ema9 is not None else 999
        indicators["ema_extension_atr"] = extension
        indicators["candle_range_atr"] = (candle_range / atr_v) if atr_v else 999.0
        # Normal entries must be close to EMA9 (fresh); Momentum keeps its own limits.
        indicators["not_chasing"] = extension <= (
            settings.normal_max_extension_atr if engine == "normal" else 1.25
        )
        if engine == "normal":
            # Fresh indicator transition: EMA cross, VWAP reclaim or EMA pullback.
            # No candle-colour/body/wick predicate participates in the trigger.
            ema9_s = ema_series(close, 9)
            ema21_s = ema_series(close, 21)
            vwap_s = session_vwap_series(high, low, close, volume, arr["time"])
            symbol_touch_atr = resolve_symbol_pullback_touch_atr(
                pair, settings.normal_pullback_touch_atr,
                settings.normal_pullback_touch_atr_by_symbol,
            )
            trigger = (
                detect_fresh_trigger(
                    opens, high, low, close, ema9_s, ema21_s, vwap_s,
                    atr_v, direction,
                    max_trigger_age=settings.normal_max_trigger_age_candles,
                    pullback_lookback=settings.normal_pullback_lookback,
                    pullback_touch_atr=symbol_touch_atr,
                )
                if ema9_s is not None and ema21_s is not None and atr_v else None
            )
            if trigger is None:
                logger.info(
                    f"NORMAL_NO_FRESH_TRIGGER pair={pair} direction={direction} "
                    f"ema9={ema9} ema21={ema21} adx={adx_v} close={float(close[-1]):.8g} "
                    f"vwap={vwap_v} atr={atr_v}"
                )
                self.watchlists["normal"].remove_pair(pair, "normal")
                self._log_candle_evaluation(
                    pair, engine, direction, candle_ts, candle_close_ms,
                    confirm_candle_time_ms, indicators, "REJECT",
                    failed=["no_fresh_trigger"], strategy="NONE",
                )
                return
            strategy, trigger_age = trigger
            indicators["trigger_age_candles"] = trigger_age
            gates = _normal_threshold_gates(
                indicators, direction, settings.normal_adx_min,
                strategy=strategy,
            )
            # The configured EMA9-distance limit is a derived indicator safeguard
            # against chasing an already-extended trigger candle.
            gates["not_chasing"] = _normal_extension_gate(
                indicators, settings.normal_max_extension_atr,
            )
            # RVOL is optional: when disabled it remains diagnostic data only.
            if settings.normal_rvol_gate_enabled:
                gates.update(_normal_relative_volume_gate(
                    indicators, direction, settings.normal_rvol_min,
                ))
            try:
                adx_value = float(indicators.get("adx"))
                plus_di = float(indicators.get("plus_di"))
                minus_di = float(indicators.get("minus_di"))
            except (TypeError, ValueError, OverflowError):
                adx_value = plus_di = minus_di = float("nan")
            # ADX >= the configured floor is the strength requirement.
            # Rising/ADX >= 30 is not an extra veto; it delays valid entries.
            gates["di_alignment"] = bool(
                np.isfinite(plus_di) and np.isfinite(minus_di)
                and (plus_di > minus_di if direction == "LONG" else minus_di > plus_di)
            )
            failed = [name for name, passed in gates.items() if not passed]
            logger.info(
                "NORMAL_GATE_DIAGNOSTICS pair=%s direction=%s strategy=%s "
                "trigger=%s trigger_age=%s gate_results=%s failed=%s",
                pair, direction, strategy, trigger[0], trigger_age,
                ",".join(f"{name}:{'PASS' if passed else 'FAIL'}" for name, passed in gates.items()),
                ",".join(failed) if failed else "none",
            )
            if failed:
                # A previously watched setup is no longer valid once one of the
                # five core Normal conditions fails.
                self.watchlists["normal"].remove_pair(pair, "normal")
                self._log_candle_evaluation(
                    pair, engine, direction, candle_ts, candle_close_ms,
                    confirm_candle_time_ms, indicators, "REJECT", failed=failed,
                    strategy=strategy,
                )
                return
            # Score the five core Normal indicators separately from the RVOL participation gate.
            self.watchlists["normal"].remove_pair(pair, "normal")
            result = self.normal_scoring.score_normal(indicators, direction)
        else:
            # Do not impose ADX before regime classification: PRE_EXPLOSION and
            # first ignition often occur before lagging ADX reaches trend thresholds.
            # ADX remains a quality feature for CONTINUATION, not an early-setup veto.
            # Momentum extras
            bb = bollinger_bands(close, 20, 2.0)
            kc = keltner_channels(high, low, close, 20, 1.5)
            squeeze_release = False
            compression_active = False
            bw_expand = False
            bw_now = bw_prev = None
            if bb is not None and kc is not None and len(close) >= 22:
                upper_bb, mid_bb, lower_bb = bb
                upper_kc, _, lower_kc = kc
                bw_now = bandwidth(upper_bb, lower_bb, mid_bb)
                prev_bb = bollinger_bands(close[:-1],20,2.0)
                prev_kc = keltner_channels(high[:-1],low[:-1],close[:-1],20,1.5)
                if prev_bb is not None and prev_kc is not None:
                    p_upper,p_mid,p_lower=prev_bb
                    p_kupper,_,p_klower=prev_kc
                    # Compression is a rolling state, not a one-candle event.
                    recent_compression = False
                    for back in range(1, min(7, len(close) - 20)):
                        hist_bb = bollinger_bands(close[:-back], 20, 2.0)
                        hist_kc = keltner_channels(high[:-back], low[:-back], close[:-back], 20, 1.5)
                        if hist_bb is None or hist_kc is None:
                            continue
                        hu, _, hl = hist_bb
                        hku, _, hkl = hist_kc
                        if hu < hku and hl > hkl:
                            recent_compression = True
                            break
                    now_released = upper_bb > upper_kc or lower_bb < lower_kc
                    compression_active = recent_compression and not now_released
                    squeeze_release = recent_compression and now_released
                    bw_prev = bandwidth(p_upper,p_lower,p_mid)
                    bw_expand = bw_now is not None and bw_prev is not None and bw_now > bw_prev * 1.03
            atr_series = [atr(high[:i+1],low[:i+1],close[:i+1],14) for i in range(max(14,len(close)-8),len(close))]
            valid_atr = [x for x in atr_series if x is not None]
            atr_expansion = (valid_atr[-1]/np.mean(valid_atr[:-1])) if len(valid_atr)>=3 and np.mean(valid_atr[:-1])>0 else 1.0
            hist_series = _macd_hist_history(close, points=4)
            macd_accel = (
                len(hist_series) >= 3
                and (
                    hist_series[-1] > hist_series[-2] > hist_series[-3]
                    if direction == "LONG"
                    else hist_series[-1] < hist_series[-2] < hist_series[-3]
                )
            )
            rsi_series = [rsi(close[:i+1],14) for i in range(max(15,len(close)-4),len(close))]
            valid_rsi = [x for x in rsi_series if x is not None]
            rsi_accel = len(valid_rsi)>=3 and ((valid_rsi[-1]>valid_rsi[-2]>valid_rsi[-3]) if direction=="LONG" else (valid_rsi[-1]<valid_rsi[-2]<valid_rsi[-3]))
            roc_v = roc(close,10)
            indicators["squeeze_release"] = squeeze_release
            indicators["compression_active"] = compression_active
            indicators["bw_expand"] = bw_expand
            indicators["atr_expansion"] = float(atr_expansion)
            indicators["ema_aligned"] = bool(
                ema9 is not None and ema21 is not None
                and ((direction == "LONG" and ema9 > ema21)
                     or (direction == "SHORT" and ema9 < ema21))
            )
            indicators["macd_accel"] = bool(macd_accel)
            indicators["rsi_accel"] = bool(rsi_accel)
            indicators["structure_break"] = structure_ok
            indicators["roc"] = roc_v
            indicators["roc_direction_ok"] = roc_v is not None and ((direction=="LONG" and roc_v>0) or (direction=="SHORT" and roc_v<0))
            indicators["structure_break"] = bool(
                (direction == "LONG" and resistance is not None and close[-1] > resistance) or
                (direction == "SHORT" and support is not None and close[-1] < support))
            indicators["structure_direction_ok"] = indicators["structure_break"]
            # Evaluate session ORBs on the same active Momentum universe.
            # ORB has its own range/volume/extension rules and shares only the
            # common risk, freshness, deduplication and delivery pipeline.
            # Momentum must be a directional expansion, not merely a high score.
            momentum_state = (
                orb_result.get("state") if orb_result.get("state") in ("ORB_PREBREAKOUT", "ORB_BREAKOUT", "ORB_RETEST")
                and orb_result.get("direction") == direction
                else detect_momentum_state(indicators)
            )
            if momentum_state == "PRE_EXPLOSION":
                if is_pre_explosion_signal(indicators):
                    # Alert on a tightly qualified compression-phase ignition,
                    # before the full BB/Keltner release and large range candle.
                    momentum_state = "PRE_EXPLOSION_SIGNAL"
                    indicators["pre_explosion_signal"] = True
                else:
                    self._log_candle_evaluation(
                        pair, engine, direction, candle_ts, candle_close_ms,
                        confirm_candle_time_ms, indicators, "WATCHLIST",
                        failed=["awaiting_directional_ignition"], strategy=momentum_state,
                    )
                    self._last_processed.setdefault(engine, {})[pair] = candle_ts
                    self.watchlists[engine].add({
                        "pair": pair, "display": self.symbols.display_name(pair),
                        "direction": direction, "engine": engine,
                        "strategy": momentum_state, "score": 0.0,
                        "grade": "WATCHLIST", "price": float(close[-1]),
                        "timestamp": utc_iso(utc_now()),
                    })
                    return
            if momentum_state in ("MOMENTUM_SETUP", "POSSIBLE_EXHAUSTION"):
                state_checks = diagnose_momentum_state(indicators)
                checks_text = ",".join(
                    f"{name}:{'pass' if passed else 'fail'}"
                    for name, passed in state_checks.items()
                )
                logger.info(
                    f"MOMENTUM_CLASSIFICATION_DIAGNOSTIC pair={pair} direction={direction} "
                    f"fallback_state={momentum_state} checks=[{checks_text}] "
                    f"rvol={rvol} atr_expansion={atr_expansion:.5g} "
                    f"adx={adx_v} "
                    f"macd_direction_ok={indicators.get('macd_direction_ok')} "
                    f"roc_direction_ok={indicators.get('roc_direction_ok')} "
                    f"structure_break={indicators.get('structure_break')} "
                    f"bw_expand={indicators.get('bw_expand')}"
                )
            # Each Momentum/ORB detector owns its directional indicator rules.
            # Do not add a universal MACD, ADX/DI or duplicate RVOL veto here:
            # that would narrow valid detector states and delay early breakouts.
            momentum_gates = _momentum_entry_gates(
                momentum_state, atr_v, extension,
            )
            failed = [name for name, passed in momentum_gates.items() if not passed]
            if failed:
                self._log_candle_evaluation(
                    pair, engine, direction, candle_ts, candle_close_ms,
                    confirm_candle_time_ms, indicators, "REJECT", failed=failed,
                    strategy=momentum_state,
                )
                return
            result = (
                self.momentum_scoring.score_orb(indicators, direction)
                if momentum_state in ("ORB_PREBREAKOUT", "ORB_BREAKOUT", "ORB_RETEST")
                else self.momentum_scoring.score_momentum_explosion(indicators, direction)
                if momentum_state in ("EXPLOSION", "SQUEEZE_RELEASE", "PRE_EXPLOSION_SIGNAL")
                else self.momentum_scoring.score_momentum(indicators, direction)
            )
            strategy = momentum_state

        # Build score detail before any branch can log a low-scoring result.
        score_parts = ",".join(
            f"{c.name}:{c.points:g}/{c.max_points:g}" for c in result.components
        )

        # Normal score is diagnostic after its explicit indicator gates.
        # SIGNAL_THRESHOLD belongs to Momentum, not the Normal engine.
        if engine == "momentum" and result.total < 5.5:
            self._log_candle_evaluation(
                pair, engine, direction, candle_ts, candle_close_ms,
                confirm_candle_time_ms, indicators, "REJECT",
                failed=[f"score_below_watchlist_min_5_5({result.total:.2f})"], strategy=strategy,
                score=result.total, score_parts=score_parts,
            )
            return

        if self.cooldowns[engine].is_cooling(pair, engine, direction):
            self._log_candle_evaluation(
                pair, engine, direction, candle_ts, candle_close_ms,
                confirm_candle_time_ms, indicators, "REJECT",
                failed=["cooldown"], strategy=strategy, score=result.total,
                score_parts=score_parts,
            )
            return

        # Indicator-qualified setups are eligible immediately; no second-candle
        # follow-through, candle-colour, or wick-shape confirmation is required.
        self.watchlists[engine].remove_pair(pair, engine)
        indicators["confirmation_candles"] = 1

        swing_high, swing_low = _confirmed_swing_levels(high, low, lookback=30, left=2, right=2)
        if engine == "momentum" and strategy in ("ORB_PREBREAKOUT", "ORB_BREAKOUT", "ORB_RETEST"):
            # ORB invalidation belongs beyond the opening-range opposite edge,
            # not an arbitrary last-seven-candle micro swing.
            swing_low = indicators.get("orb_range_low")
            swing_high = indicators.get("orb_range_high")
        risk = calculate_risk(
            close[-1], atr_v or 0, direction,
            swing_low=swing_low, swing_high=swing_high,
        ) if atr_v else None
        strategy = strategy if engine == "normal" else momentum_state
        if not risk:
            self._log_candle_evaluation(
                pair, engine, direction, candle_ts, candle_close_ms,
                confirm_candle_time_ms, indicators, "REJECT",
                failed=["risk_levels_unavailable"], strategy=strategy,
                score=result.total, score_parts=score_parts,
            )
            return

        signal = {
            "pair": pair,
            "display": self.symbols.display_name(pair),
            "direction": direction,
            "engine": engine,
            "strategy": strategy,
            "score": result.total,
            "grade": result.grade,
            "price": close[-1],
            "indicators": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in indicators.items() if v is not None},
            "components": [{"name": c.name, "points": c.points, "max": c.max_points, "expl": c.explanation} for c in result.components],
            "timestamp": utc_iso(utc_now()),
            "candle_time_ms": candle_ts,
            "candle_close_time_ms": candle_close_ms,
            "signal_age_seconds": round(candle_age_seconds(int(utc_now().timestamp() * 1000), candle_ts, primary_interval_ms), 1),
        }
        if risk:
            signal["risk"] = {
                "entry": risk.entry,
                "entry_zone": [risk.entry_zone_low, risk.entry_zone_high],
                "stop_loss": risk.stop_loss,
                "tp1": risk.tp1,                "tp2": risk.tp2,
                "tp3": risk.tp3,
                "risk_pct": risk.risk_pct,
                "rr": [risk.rr1, risk.rr2, risk.rr3],
                "invalidation": risk.invalidation,
            }

        # Normal signals pass explicit indicator/optional participation and
        # actionable-entry checks; their score is diagnostic. Momentum has its own disposition.
        disposition = (
            "SIGNAL" if engine == "normal"
            else _score_disposition(result.total, settings.signal_threshold)
        )
        if disposition == "SIGNAL":
            detection_elapsed_ms = (asyncio.get_running_loop().time() - pair_started) * 1000.0
            detected_age = candle_age_seconds(int(utc_now().timestamp() * 1000), candle_ts, primary_interval_ms)
            logger.info(
                f"SIGNAL_DETECTED pair={pair} engine={engine} direction={direction} "
                f"strategy={strategy} candle_close_ms={candle_close_ms} "
                f"candle_age_seconds={detected_age:.1f} "
                f"pair_processing_ms={detection_elapsed_ms:.1f} "
                f"indicator_and_strategy_ms={(asyncio.get_running_loop().time() - indicator_started) * 1000.0:.1f}"
            )
            self._log_candle_evaluation(
                pair, engine, direction, candle_ts, candle_close_ms,
                confirm_candle_time_ms, indicators, "CANDIDATE",
                failed=[], strategy=strategy, score=result.total,
                score_parts=score_parts,
            )
            # Recheck immediately before claiming/sending: the scan may have
            # taken long enough for this otherwise-valid candle to go stale.
            final_age = candle_age_seconds(int(utc_now().timestamp() * 1000), candle_ts, primary_interval_ms)
            if final_age < 0 or final_age > max_signal_age:
                self._last_processed.setdefault(engine, {})[pair] = candle_ts
                logger.info(
                    f"STALE_SIGNAL_SUPPRESSED {pair} engine={engine} direction={direction} "
                    f"candle_open_ms={candle_ts} age_seconds={final_age:.1f}"
                )
                return
            signal["signal_age_seconds"] = round(final_age, 1)
            # Require a fresh public ticker to remain inside the planned entry zone.
            entry_price_started = asyncio.get_running_loop().time()
            try:
                live_price = await self.rest.get_last_price(pair)
            except Exception as exc:
                logger.warning(f"ENTRY_PRICE_CHECK_FAILED {pair}: {exc}")
                self._last_processed.setdefault(engine, {})[pair] = candle_ts
                return
            logger.info(f"SIGNAL_STAGE pair={pair} engine={engine} stage=initial_entry_price_check duration_ms={(asyncio.get_running_loop().time() - entry_price_started) * 1000.0:.1f}")
            zone_low, zone_high = risk.entry_zone_low, risk.entry_zone_high
            entry_state = (
                classify_entry_state(live_price, risk.entry, zone_low, zone_high, direction)
                if live_price is not None else "EXPIRED"
            )
            if entry_state == "RETEST":
                # Preserve this qualified setup briefly. It can alert only if a
                # fresh Futures LTP returns to the original entry band without
                # touching invalidation; it is never promoted on candle close alone.
                interval_seconds = parse_interval(primary).total_seconds()
                retest_ttl = (
                    float(settings.normal_retest_ttl_seconds) if engine == "normal"
                    else min(900, max(180, interval_seconds * 3))
                )
                pending_key = f"{pair}:{engine}:{direction}"
                pending_record = {
                    "signal": signal, "created_at": utc_iso(utc_now()),
                    "expires_at": utc_now().timestamp() + retest_ttl,
                    "entry": risk.entry, "zone_low": zone_low, "zone_high": zone_high,
                    "stop_loss": risk.stop_loss, "candle_ts": candle_ts,
                }
                if not await self._persist_pending_retest_candidate(pending_key, pending_record):
                    self._last_processed.setdefault(engine, {})[pair] = candle_ts
                    return
                self._last_processed.setdefault(engine, {})[pair] = candle_ts
                logger.info(
                    f"ENTRY_RETEST_PENDING {pair} engine={engine} direction={direction} "
                    f"live_price={live_price} entry_zone={zone_low}-{zone_high} "
                    f"ttl_seconds={retest_ttl:.0f}"
                )
                return
            if entry_state != "IMMEDIATE":
                self._last_processed.setdefault(engine, {})[pair] = candle_ts
                logger.info(
                    f"ENTRY_EXPIRED_SKIP {pair} engine={engine} direction={direction} "
                    f"live_price={live_price} entry_zone={zone_low}-{zone_high}"
                )
                return
            # Recheck freshness after the extra ticker request.
            final_age = candle_age_seconds(int(utc_now().timestamp() * 1000), candle_ts, primary_interval_ms)
            if final_age < 0 or final_age > max_signal_age:
                self._last_processed.setdefault(engine, {})[pair] = candle_ts
                logger.info(f"STALE_AFTER_TICKER_SKIP {pair} age_seconds={final_age:.1f}")
                return
            signal["live_price"] = live_price
            signal["entry_checked_at"] = utc_iso(utc_now())
            signal["signal_age_seconds"] = round(final_age, 1)
            signal["timestamp"] = utc_iso(utc_now())
            self._last_processed.setdefault(engine, {})[pair] = candle_ts
            signal_added = self.signals[engine].add(signal)
            await self._persist_evicted_signals(engine)
            if signal_added:
                # Durable claim blocks duplicate alerts after a Render restart or
                # a second worker sharing the same database.
                try:
                    from database.repositories import claim_signal_once
                    claim_started = asyncio.get_running_loop().time()
                    claimed = await claim_signal_once(signal, candle_ts, ttl_hours=4)
                    logger.info(f"SIGNAL_STAGE pair={pair} engine={engine} stage=database_claim duration_ms={(asyncio.get_running_loop().time() - claim_started) * 1000.0:.1f} claimed={claimed}")
                except Exception as exc:
                    logger.error(f"Persistent signal deduplication unavailable: {exc}")
                    claimed = False
                if not claimed:
                    self.signals[engine].invalidate(pair, engine, direction)
                    logger.warning(f"Signal suppressed: persistent duplicate/claim unavailable for {pair} {engine} {direction}")
                    return
                logger.info(
                    f"SIGNAL_CLAIMED pair={pair} engine={engine} direction={direction} "
                    f"candle_open_ms={candle_ts}"
                )
                # Recheck price and age after the durable DB claim, immediately
                # before Telegram delivery; a slow DB write must not stale the entry.
                if self.telegram:
                    delivery_price_started = asyncio.get_running_loop().time()
                    try:
                        delivery_price = await self.rest.get_last_price(pair)
                    except Exception as exc:
                        delivery_price = None
                        logger.warning(f"DELIVERY_PRICE_CHECK_FAILED {pair}: {exc}")
                    logger.info(f"SIGNAL_STAGE pair={pair} engine={engine} stage=final_delivery_price_check duration_ms={(asyncio.get_running_loop().time() - delivery_price_started) * 1000.0:.1f}")
                    delivery_age = candle_age_seconds(int(utc_now().timestamp() * 1000), candle_ts, primary_interval_ms)
                    if (
                        delivery_price is None
                        or classify_entry_state(delivery_price, risk.entry, zone_low, zone_high, direction) != "IMMEDIATE"
                        or delivery_age < 0
                        or delivery_age > max_signal_age
                    ):
                        try:
                            from database.repositories import release_signal_claim
                            released = await release_signal_claim(signal, candle_ts)
                            logger.info(
                                f"SIGNAL_CLAIM_RELEASE_AFTER_VALIDATION_SKIP pair={pair} "
                                f"released={released}"
                            )
                        except Exception as exc:
                            logger.error(f"SIGNAL_CLAIM_RELEASE_FAILED {pair}: {exc}")
                        self.signals[engine].invalidate(pair, engine, direction)
                        logger.info(
                            f"DELIVERY_STALE_SKIP {pair} engine={engine} price={delivery_price} "
                            f"entry_zone={zone_low}-{zone_high} age_seconds={delivery_age:.1f}"
                        )
                        return
                    signal["live_price"] = delivery_price
                    signal["entry_checked_at"] = utc_iso(utc_now())
                    signal["signal_age_seconds"] = round(delivery_age, 1)
                    signal["timestamp"] = utc_iso(utc_now())
                # Queue durable per-recipient deliveries. The background worker
                # retries transient Telegram errors while the original entry window remains valid.
                queued = await self._queue_telegram_signal(signal)
                if queued:
                    self.cooldowns[engine].set(pair, engine, direction)
                    logger.info(
                        "SIGNAL_QUEUED %s %s %s score=%s age_seconds=%s",
                        result.grade, direction, pair, result.total, signal.get("signal_age_seconds"),
                    )
                    persisted_signal = self.signals[engine].get(pair, engine, direction)
                    if persisted_signal:
                        try:
                            from database.repositories import save_signal
                            if not await save_signal(persisted_signal):
                                logger.error("ACTIVE_SIGNAL_PERSIST_FAILED %s engine=%s", pair, engine)
                        except Exception as exc:
                            logger.error("ACTIVE_SIGNAL_PERSIST_ERROR %s: %s", pair, exc)
                else:
                    self.telegram_delivery_failures += 1
                    try:
                        from database.repositories import release_signal_claim
                        await release_signal_claim(signal, candle_ts)
                    except Exception as exc:
                        logger.error("SIGNAL_CLAIM_RELEASE_FAILED %s: %s", pair, exc)
                    self.signals[engine].invalidate(pair, engine, direction)
                    logger.warning(
                        "SIGNAL_NOT_QUEUED %s engine=%s direction=%s; claim released",
                        pair, engine, direction,
                    )
        elif disposition == "WATCHLIST":
            self._log_candle_evaluation(
                pair, engine, direction, candle_ts, candle_close_ms,
                confirm_candle_time_ms, indicators, "WATCHLIST",
                failed=[f"signal_threshold({settings.signal_threshold:g})"],
                strategy=strategy, score=result.total, score_parts=score_parts,
            )
            self._last_processed.setdefault(engine, {})[pair] = candle_ts
            self.watchlists[engine].add(signal)
            logger.info(f"WATCHLIST {direction} {pair} score={result.total}")

    async def _queue_telegram_signal(self, signal: dict) -> bool:
        """Persist one delivery per admin before treating an alert as accepted."""
        if (
            not self.telegram
            or not getattr(self.telegram, "bot", None)
            or not getattr(self.telegram, "admin_ids", None)
        ):
            logger.warning("TELEGRAM_OUTBOX_NOT_QUEUED reason=no_bot_or_recipients")
            return False
        try:
            from database.repositories import enqueue_signal_deliveries
            queued = await enqueue_signal_deliveries(signal, self.telegram.admin_ids)
            if queued > 0:
                logger.info(
                    "TELEGRAM_OUTBOX_QUEUED pair=%s engine=%s direction=%s recipients=%s",
                    signal.get("pair"), signal.get("engine"), signal.get("direction"), queued,
                )
                return True
            logger.error(
                "TELEGRAM_OUTBOX_QUEUE_FAILED pair=%s engine=%s direction=%s",
                signal.get("pair"), signal.get("engine"), signal.get("direction"),
            )
            return False
        except Exception as exc:
            logger.exception("TELEGRAM_OUTBOX_ENQUEUE_ERROR: %s", exc)
            return False

    async def _persist_evicted_signals(self, engine: str) -> None:
        """Mirror in-memory capacity evictions to durable signal lifecycle rows."""
        evicted = self.signals[engine].drain_evicted()
        if not evicted:
            return
        try:
            from database.repositories import update_signal_record
            for signal in evicted:
                updated = await update_signal_record(signal, "evicted")
                if updated:
                    logger.info(
                        f"ACTIVE_SIGNAL_EVICTED_PERSISTED pair={signal.get('pair')} "
                        f"engine={engine} direction={signal.get('direction')}"
                    )
                else:
                    logger.warning(
                        f"ACTIVE_SIGNAL_EVICTION_NOT_PERSISTED pair={signal.get('pair')} "
                        f"engine={engine} direction={signal.get('direction')}"
                    )
        except Exception as exc:
            logger.error(f"ACTIVE_SIGNAL_EVICTION_PERSIST_ERROR engine={engine}: {exc}")

    async def _persist_pending_retest_candidate(self, key: str, pending: dict) -> bool:
        """Persist before retaining a retest candidate in memory.

        Both a False result and an exception fail closed. Keeping an in-memory
        candidate after a database exception would create a non-restart-safe
        alert that can disappear silently on a Render restart.
        """
        try:
            from database.repositories import save_pending_retest
            saved = await save_pending_retest(key, pending)
        except Exception as exc:
            logger.error(f"PENDING_RETEST_PERSIST_ERROR {key}: {exc}")
            saved = False
        if not saved:
            self.pending_retests.pop(key, None)
            logger.error(
                f"PENDING_RETEST_PERSIST_FAILED_CLOSED {key}; "
                "candidate discarded because durable state was not committed"
            )
            return False

        # Keep at most one pending direction per pair+engine. Persist the new
        # candidate first; only then remove the opposite candidate. If deletion
        # fails, discard the new candidate so the durable store cannot retain
        # competing LONG and SHORT retests across a restart.
        new_signal = pending.get("signal") or {}
        pair = new_signal.get("pair")
        engine = new_signal.get("engine")
        direction = new_signal.get("direction")
        opposite = "SHORT" if direction == "LONG" else "LONG" if direction == "SHORT" else None
        if opposite and pair and engine:
            opposite_keys = [
                old_key for old_key, old in self.pending_retests.items()
                if old_key != key
                and (old_signal := (old.get("signal") or {})).get("pair") == pair
                and old_signal.get("engine") == engine
                and old_signal.get("direction") == opposite
            ]
            try:
                from database.repositories import delete_pending_retest
                for old_key in opposite_keys:
                    if not await delete_pending_retest(old_key):
                        raise RuntimeError(f"durable opposite retest deletion failed: {old_key}")
            except Exception as exc:
                try:
                    from database.repositories import delete_pending_retest
                    await delete_pending_retest(key)
                except Exception:
                    pass
                logger.error(f"PENDING_RETEST_OPPOSITE_REPLACE_FAILED {key}: {exc}")
                return False
            for old_key in opposite_keys:
                self.pending_retests.pop(old_key, None)
                logger.info(f"PENDING_RETEST_REPLACED_OPPOSITE old={old_key} new={key}")

        self.pending_retests[key] = pending
        return True

    async def _check_pending_retest(self, pair: str, engine: str) -> None:
        """Promote a persisted setup only while fresh LTP is in its original zone."""
        for key, pending in list(self.pending_retests.items()):
            signal = pending.get("signal") or {}
            if signal.get("pair") != pair or signal.get("engine") != engine:
                continue
            async def remove(reason: str) -> None:
                self.pending_retests.pop(key, None)
                try:
                    from database.repositories import delete_pending_retest
                    await delete_pending_retest(key)
                except Exception as exc:
                    logger.warning(f"PENDING_RETEST_DELETE_FAILED {key}: {exc}")
                logger.info(f"PENDING_RETEST_REMOVED {key} reason={reason}")
            if utc_now().timestamp() >= float(pending.get("expires_at", 0)):
                await remove("expired")
                continue
            # Normal retests must obey the same completed-candle freshness
            # budget as first-pass Normal alerts. The longer retest TTL is only
            # a storage upper bound; it must never authorize a stale entry.
            if engine == "normal":
                signal_age = candle_age_seconds(
                    int(utc_now().timestamp() * 1000), int(pending.get("candle_ts", 0)),
                    int(parse_interval(settings.candle_primary_tf).total_seconds() * 1000),
                )
                if signal_age < 0 or signal_age > int(settings.normal_max_signal_age_seconds):
                    await remove("signal_stale")
                    continue
            try:
                live_price = await self.rest.get_last_price(pair)
                self.pending_retest_checks += 1
            except Exception as exc:
                self.pending_retest_failures += 1
                logger.warning(f"RETEST_PRICE_CHECK_FAILED {pair}: {exc}")
                continue
            if live_price is None:
                continue
            try:
                price = float(live_price)
                direction = signal["direction"]
                stop = float(pending["stop_loss"])
                candle_ts = int(pending["candle_ts"])
                if direction not in ("LONG", "SHORT"):
                    raise ValueError("invalid pending direction")
            except (KeyError, TypeError, ValueError, OverflowError):
                await remove("invalid_payload")
                continue
            if (price <= stop if direction == "LONG" else price >= stop):
                await remove("stop_invalidated")
                continue
            existing_signal = self.signals[engine].get(pair, engine, direction)
            if (
                (existing_signal and existing_signal.get("status") == "active")
                or self.cooldowns[engine].is_cooling(pair, engine, direction)
            ):
                await remove("active_signal_or_cooldown")
                continue
            if classify_entry_state(price, pending["entry"], pending["zone_low"], pending["zone_high"], direction) != "IMMEDIATE":
                continue
            try:
                delivery_price = await self.rest.get_last_price(pair)
            except Exception as exc:
                self.pending_retest_failures += 1
                logger.warning(f"RETEST_DELIVERY_PRICE_FAILED {pair}: {exc}")
                continue
            if delivery_price is None or classify_entry_state(float(delivery_price), pending["entry"], pending["zone_low"], pending["zone_high"], direction) != "IMMEDIATE":
                continue
            signal = dict(signal)
            signal_age = candle_age_seconds(
                int(utc_now().timestamp() * 1000), candle_ts,
                int(parse_interval(settings.candle_primary_tf).total_seconds() * 1000),
            )
            signal.update({
                "live_price": float(delivery_price),
                "entry_type": "RETEST_CONFIRMED",
                "entry_checked_at": utc_iso(utc_now()),
                "timestamp": utc_iso(utc_now()),
                "signal_age_seconds": round(signal_age, 1),
            })
            try:
                from database.repositories import claim_signal_once
                claimed = await claim_signal_once(signal, int(pending["candle_ts"]), ttl_hours=4)
            except Exception as exc:
                self.pending_retest_failures += 1
                logger.error(f"RETEST_DEDUP_ERROR {key}: {exc}")
                continue
            if not claimed:
                await remove("duplicate")
                continue
            signal_added = self.signals[engine].add(signal)
            await self._persist_evicted_signals(engine)
            if not signal_added:
                try:
                    from database.repositories import release_signal_claim
                    await release_signal_claim(signal, int(pending["candle_ts"]))
                except Exception as exc:
                    logger.error(f"RETEST_CLAIM_RELEASE_FAILED {key}: {exc}")
                await remove("active_capacity_or_duplicate")
                continue
            queued = await self._queue_telegram_signal(signal)
            if queued:
                self.cooldowns[engine].set(pair, engine, direction)
                from database.repositories import save_signal
                active = self.signals[engine].get(pair, engine, direction) or signal
                if not await save_signal(active):
                    logger.error(f"RETEST_ACTIVE_SIGNAL_PERSIST_FAILED {key}")
                logger.info(f"RETEST_SIGNAL_QUEUED {key} price={delivery_price}")
            else:
                self.signals[engine].invalidate(pair, engine, direction)
                self.telegram_delivery_failures += 1
                try:
                    from database.repositories import release_signal_claim
                    await release_signal_claim(signal, int(pending["candle_ts"]))
                except Exception as exc:
                    logger.error(f"RETEST_CLAIM_RELEASE_FAILED {key}: {exc}")
                logger.warning(f"RETEST_SIGNAL_NOT_QUEUED {key}; candidate remains retryable")
                continue
            await remove("queued")

    def _log_candle_evaluation(
        self, pair: str, engine: str, direction: str, candle_ts: int,
        candle_close_ms: int, confirm_candle_time_ms: Optional[int],
        indicators: Dict[str, Any], decision: str, failed: List[str],
        strategy: str, score: Optional[float] = None,
        score_parts: str = "",
    ) -> None:
        """Emit a compact, timestamped audit trail for each directional candidate."""
        def value(name: str) -> str:
            item = indicators.get(name)
            if item is None:
                return "NA"
            if isinstance(item, (float, np.floating)):
                return f"{float(item):.5g}"
            return str(item)

        stats = self._scan_rejection_stats.get(engine)
        if stats is not None:
            stats["evaluated"] += 1
            disposition = decision.lower()
            if disposition in ("signal", "watchlist", "reject"):
                stats[disposition] += 1
            for reason in failed:
                stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1

        age = candle_age_seconds(
            int(utc_now().timestamp() * 1000), candle_ts,
            int(parse_interval(settings.candle_primary_tf).total_seconds() * 1000),
        )
        logger.info(
            f"CANDLE_EVAL pair={pair} engine={engine} direction={direction} "
            f"decision={decision} strategy={strategy} "
            f"candle_open_ms={candle_ts} candle_close_ms={candle_close_ms} "
            f"age_seconds={age:.1f} "
            f"close={value('close')} ema9={value('ema9')} ema21={value('ema21')} "
            f"ema50={value('ema50')} vwap={value('vwap')} rsi={value('rsi')} "
            f"adx={value('adx')} plus_di={value('plus_di')} minus_di={value('minus_di')} "
            f"macd_hist={value('macd_hist')} rvol={value('rvol')} atr_pct={value('atr_pct')} "
            f"ema_extension_atr={value('ema_extension_atr')} "
            f"adx_rising={value('adx_rising')} "
            f"score={score if score is not None else 'NA'} "
            f"score_parts=[{score_parts}] failed={','.join(failed) if failed else 'none'}"
        )

    async def _update_signal_outcomes(
        self, pair: str, engine: str, high: float, low: float,
        candle_ts: Optional[int] = None,
    ) -> None:
        """Conservatively monitor active alerts; if SL and TP touch same bar, SL wins."""
        sweep_high, sweep_low = high, low
        for sig in self.signals[engine].list_active():
            if sig.get("pair") != pair:
                continue
            risk = sig.get("risk") or {}
            direction = sig.get("direction")
            high, low = sweep_high, sweep_low
            if (
                candle_ts is not None and sig.get("entry_type") == "LIVE_INTRABAR"
                and int(sig.get("candle_time_ms") or 0) == int(candle_ts)
            ):
                # Entry candle of a live signal: its high/low include price action
                # BEFORE entry. Use only the post-entry extremes seen on the feed.
                ext = getattr(self, "_live_post_entry", {}).pop((pair, direction), None)
                if not ext:
                    continue
                high, low = ext[0], ext[1]
            sl, t1, t2, t3 = (risk.get(k) for k in ("stop_loss", "tp1", "tp2", "tp3"))
            if sl is None or t1 is None:
                continue
            sl_hit = low <= sl if direction == "LONG" else high >= sl
            try:
                from database.repositories import record_signal_outcome
                await record_signal_outcome(sig, high, "ACTIVE")
                await record_signal_outcome(sig, low, "ACTIVE")
            except Exception as exc:
                logger.warning(f"Outcome metrics update failed for {pair}: {exc}")
            if sl_hit:
                if self.signals[engine].mark_sl(pair, engine, direction):
                    try:
                        from database.repositories import update_signal_record, record_signal_outcome
                        await update_signal_record(sig, "closed_sl")
                        final_outcome = "FAILED_BREAKOUT" if str(sig.get("strategy", "")).startswith("ORB") else "STOP"
                        await record_signal_outcome(sig, float(sl), final_outcome)
                    except Exception as exc:
                        logger.error(f"Persist SL outcome failed for {pair}: {exc}")
                continue
            hits = [
                i for i, target in enumerate((t1, t2, t3), 1)
                if target is not None
                and not sig.get(f"tp{i}_hit", False)
                and (high >= target if direction == "LONG" else low <= target)
            ]
            for level in hits:
                self.signals[engine].mark_tp(pair, engine, direction, level)
            if hits:
                current = self.signals[engine].get(pair, engine, direction) or sig
                status = current.get("status", "active")
                try:
                    from database.repositories import update_signal_record, record_signal_outcome
                    await update_signal_record(current, status)
                    highest_target = max(hits)
                    target_price = (t1, t2, t3)[highest_target - 1]
                    await record_signal_outcome(
                        current, float(target_price), f"TP{highest_target}"
                    )
                except Exception as exc:
                    logger.error(f"Persist TP outcome failed for {pair}: {exc}")

    async def shutdown(self) -> None:
        self._running = False
        logger.info("Signal engine stopped.")
