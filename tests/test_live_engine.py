"""Engine wiring for live Explosion/ORB: trigger -> validate -> claim -> queue."""
import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from config import settings
from core import engine as engine_module
from core.engine import SignalEngine
from database import repositories

INTERVAL = 300_000
T0 = 1_800_000_000_000 - (1_800_000_000_000 % INTERVAL)   # forming candle open
NOW_MS = T0 + 90_000                                       # 30% into the candle
PAIR = "B-TEST_USDT"


def arrays(forming_open=T0, n=100):
    times = forming_open - INTERVAL * np.arange(n, 0, -1)
    closes = 100.0 + np.where(np.arange(n) % 2 == 0, 0.1, -0.1)
    # Build a mild completed-candle uptrend so live direction comes from EMA9/21,
    # not from the forming candle's colour.
    closes[-25:] = np.linspace(99.5, 100.0, 25)
    opens = np.full(n, 100.0)
    return {"time": times.astype(np.int64), "open": opens,
            "high": np.maximum(opens, closes) + 0.1, "low": np.minimum(opens, closes) - 0.1,
            "close": closes, "volume": np.full(n, 1000.0)}


class Candles:
    def get_arrays(self, pair, timeframe):
        return arrays()


class Symbols:
    momentum_symbols = [PAIR]

    def display_name(self, pair):
        return "TESTUSDT"


class Rest:
    def __init__(self, price):
        self.price = price
        self.calls = 0

    async def get_last_price(self, pair):
        self.calls += 1
        return self.price


@pytest.fixture
def harness(monkeypatch):
    calls = {"claim": [], "queue": [], "release": []}

    async def claim(signal, candle_ts, ttl_hours=4):
        calls["claim"].append((signal["pair"], candle_ts))
        return True

    async def release(signal, candle_ts):
        calls["release"].append(candle_ts)
        return True

    async def enqueue(signal, admin_ids):
        calls["queue"].append(signal)
        return 1

    async def save(signal):
        return True

    for name, fn in (("claim_signal_once", claim), ("release_signal_claim", release),
                     ("enqueue_signal_deliveries", enqueue), ("save_signal", save)):
        monkeypatch.setattr(repositories, name, fn)
    monkeypatch.setattr(settings, "candle_primary_tf", "5m")
    monkeypatch.setattr(settings, "live_ignition_mode", "live")
    monkeypatch.setattr(settings, "live_max_signals_per_hour", 8)
    monkeypatch.setattr(settings, "futures_round_trip_cost_bps", 2.0)
    monkeypatch.setattr(engine_module, "utc_now",
                        lambda: datetime.fromtimestamp(NOW_MS / 1000, tz=timezone.utc))

    def build(price):
        eng = SignalEngine(Rest(price), Symbols(), Candles())
        eng.telegram = SimpleNamespace(bot=object(), admin_ids=[1])
        return eng

    return build, calls


def impulse_candle(atr_multiple=1.5, atr=0.3):
    c = 100.0 + atr_multiple * atr
    return {"time": T0, "open": 100.0, "high": c + 0.01, "low": 99.98, "close": c, "volume": 600.0}


async def run_candle(eng, candle):
    eng.on_live_candle(PAIR, "5m", candle)
    if eng._live_tasks:
        await asyncio.gather(*list(eng._live_tasks))


def test_live_explosion_is_claimed_queued_and_cooled_down(harness):
    build, calls = harness
    candle = impulse_candle()
    eng = build(candle["close"])

    async def scenario():
        await run_candle(eng, candle)
        # Same candle again (next WS tick): must not produce a second signal.
        eng._live_last_tick.clear()
        await run_candle(eng, candle)

    asyncio.run(scenario())
    assert len(calls["claim"]) == 1 and len(calls["queue"]) == 1
    signal = calls["queue"][0]
    assert signal["entry_type"] == "LIVE_INTRABAR" and signal["strategy"] == "EXPLOSION"
    assert signal["direction"] == "LONG" and signal["engine"] == "momentum"
    assert signal["candle_time_ms"] == T0 and signal["score"] > 0.0
    risk = signal["risk"]
    assert risk["stop_loss"] < risk["entry"] < risk["tp1"] < risk["tp2"] < risk["tp3"]
    assert eng.cooldowns["momentum"].is_cooling(PAIR, "momentum", "LONG")
    assert eng.signals["momentum"].get(PAIR, "momentum", "LONG")["status"] == "active"
    assert eng.live_stats["emitted"] == 1


def test_shadow_mode_logs_but_never_emits(harness, monkeypatch):
    build, calls = harness
    monkeypatch.setattr(settings, "live_ignition_mode", "shadow")
    candle = impulse_candle()
    eng = build(candle["close"])
    asyncio.run(run_candle(eng, candle))
    assert not calls["claim"] and not calls["queue"]
    assert eng.live_stats["candidates"] == 1 and eng.live_stats["emitted"] == 0


def test_off_mode_does_nothing(harness, monkeypatch):
    build, calls = harness
    monkeypatch.setattr(settings, "live_ignition_mode", "off")
    candle = impulse_candle()
    eng = build(candle["close"])
    asyncio.run(run_candle(eng, candle))
    assert not calls["claim"] and eng.live_stats["candidates"] == 0


def test_price_that_ran_away_before_validation_is_rejected(harness):
    build, calls = harness
    candle = impulse_candle()
    eng = build(candle["close"] + 0.5)          # LTP already ~1.7 ATR beyond the trigger
    asyncio.run(run_candle(eng, candle))
    assert not calls["claim"] and eng.live_stats["rejected"] == 1


def test_price_that_reversed_before_validation_is_rejected(harness):
    build, calls = harness
    candle = impulse_candle()
    eng = build(candle["close"] - 0.2)
    asyncio.run(run_candle(eng, candle))
    assert not calls["claim"] and eng.live_stats["rejected"] == 1


def test_hourly_cap_blocks_extra_live_signals(harness, monkeypatch):
    build, calls = harness
    monkeypatch.setattr(settings, "live_max_signals_per_hour", 1)
    candle = impulse_candle()
    eng = build(candle["close"])
    eng._ensure_live_state()
    eng._live_emitted.append(__import__("time").time())
    asyncio.run(run_candle(eng, candle))
    assert not calls["claim"] and eng.live_stats["capped"] == 1


def test_non_momentum_pair_and_other_timeframes_are_ignored(harness):
    build, calls = harness
    candle = impulse_candle()
    eng = build(candle["close"])

    async def scenario():
        eng.on_live_candle("B-OTHER_USDT", "5m", candle)
        eng.on_live_candle(PAIR, "15m", candle)
        assert not eng._live_tasks

    asyncio.run(scenario())
    assert not calls["claim"]


def test_failed_queue_releases_claim_and_invalidates(harness, monkeypatch):
    build, calls = harness

    async def enqueue_fails(signal, admin_ids):
        return 0

    monkeypatch.setattr(repositories, "enqueue_signal_deliveries", enqueue_fails)
    candle = impulse_candle()
    eng = build(candle["close"])
    asyncio.run(run_candle(eng, candle))
    assert len(calls["claim"]) == 1 and calls["release"] == [T0]
    assert eng.signals["momentum"].get(PAIR, "momentum", "LONG").get("status") != "active"
    assert not eng.cooldowns["momentum"].is_cooling(PAIR, "momentum", "LONG")


def test_entry_candle_outcome_uses_post_entry_extremes_only(harness, monkeypatch):
    build, calls = harness
    candle = impulse_candle()
    eng = build(candle["close"])
    recorded = []

    async def record(sig, price, outcome):
        recorded.append((round(price, 4), outcome))
        return True

    async def update(sig, status):
        return True

    monkeypatch.setattr(repositories, "record_signal_outcome", record, raising=False)
    monkeypatch.setattr(repositories, "update_signal_record", update, raising=False)

    async def scenario():
        await run_candle(eng, candle)
        entry = calls["queue"][0]["risk"]["entry"]
        # Feed two later ticks; then the entry candle closes with its FULL range
        # (low 99.98 is PRE-entry and far below the stop).
        eng._live_last_tick.clear()
        eng.on_live_candle(PAIR, "5m", {**candle, "close": entry + 0.05})
        eng._live_last_tick.clear()
        eng.on_live_candle(PAIR, "5m", {**candle, "close": entry - 0.02})
        await eng._update_signal_outcomes(PAIR, "momentum", entry + 0.2, 99.0, T0)
        return entry

    entry = asyncio.run(scenario())
    lows = [p for p, _ in recorded]
    assert 99.0 not in lows                                   # pre-entry low ignored
    assert round(entry - 0.02, 4) in lows and round(entry + 0.05, 4) in lows
    assert eng.signals["momentum"].get(PAIR, "momentum", "LONG")["status"] == "active"
