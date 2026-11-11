"""Scoring engine unit tests."""
from core.scoring_engine import ScoringEngine


def test_normal_score_high():
    engine = ScoringEngine()
    indicators = {
        "close": 67000,
        "ema9": 67100,
        "ema21": 66800,
        "vwap": 66900,
        "adx": 32,
        "rsi": 58,
        "macd_hist": 12.5,
        "rvol": 1.8,
        "atr_pct": 0.9,
        "body_ratio": 0.7,
        "confirm_15m": True,
        "structure_ok": True,
    }
    result = engine.score_normal(indicators, "LONG")
    assert result.total >= 7.0
    assert result.grade in ("SIGNAL", "STRONG", "HIGH-CONVICTION")
    assert result.direction == "LONG"


def test_normal_score_low_adx():
    engine = ScoringEngine()
    indicators = {
        "close": 67000,
        "ema9": 67100,
        "ema21": 66800,
        "vwap": 66900,
        "adx": 15,
        "rsi": 58,
        "macd_hist": 12.5,
        "rvol": 1.1,
        "atr_pct": 0.2,
        "body_ratio": 0.3,
        "confirm_15m": False,
        "structure_ok": False,
    }
    result = engine.score_normal(indicators, "LONG")
    assert result.total < 6.0


def test_normal_five_condition_gate_long_and_short():
    from core.engine import _normal_threshold_gates

    long_indicators = {
        "ema9": 101.0, "ema21": 100.0,
        "close": 102.0, "vwap": 101.0,
        "adx": 25.0, "confirm_15m": True,
        "rsi": 50.1, "atr": 0.5, "directional_candle": False,
    }
    assert all(_normal_threshold_gates(long_indicators, "LONG", 25.0).values())

    short_indicators = {
        "ema9": 99.0, "ema21": 100.0,
        "close": 98.0, "vwap": 99.0,
        "adx": 25.0, "confirm_15m": True,
        "rsi": 49.9, "atr": 0.5, "directional_candle": False,
    }
    assert all(_normal_threshold_gates(short_indicators, "SHORT", 25.0).values())


def test_normal_five_condition_gate_rejects_each_missing_confirmation():
    from core.engine import _normal_threshold_gates

    base = {
        "ema9": 101.0, "ema21": 100.0,
        "close": 102.0, "vwap": 101.0,
        "adx": 25.0, "confirm_15m": True,
        "rsi": 55.0, "atr": 0.5, "directional_candle": False,
    }
    assert all(_normal_threshold_gates(base, "LONG", 25.0).values())

    for key, value in (
        ("ema9", 99.0),
        ("close", 100.0),
        ("adx", 19.99),
        ("rsi", 50.0),
    ):
        changed = dict(base)
        changed[key] = value
        gates = _normal_threshold_gates(changed, "LONG", 25.0)
        assert not all(gates.values()), key
        assert [name for name, passed in gates.items() if not passed]


def test_normal_rsi_scoring_does_not_penalize_strong_directional_momentum():
    engine = ScoringEngine()
    long_result = engine.score_normal({
        "close": 102, "ema9": 101, "ema21": 100, "vwap": 101,
        "adx": 30, "rsi": 76, "atr": 0.5, "macd_hist": 1,
    }, "LONG")
    short_result = engine.score_normal({
        "close": 98, "ema9": 99, "ema21": 100, "vwap": 99,
        "adx": 30, "rsi": 24, "atr": 0.5, "macd_hist": -1,
    }, "SHORT")
    assert next(c for c in long_result.components if c.name == "RSI 14").points == 1.5
    assert next(c for c in short_result.components if c.name == "RSI 14").points == 1.5

def test_orb_scoring_rewards_clean_breakout_and_penalizes_extension():
    from core.scoring_engine import ScoringEngine

    engine = ScoringEngine()
    clean = engine.score_orb({
        "rvol": 1.8, "orb_session_rvol": 1.8, "orb_rolling_rvol": 1.8, "close": 102.0, "vwap": 101.0,
        "ema9": 101.5, "ema21": 100.0, "rsi": 60.0, "atr": 0.5,
        "orb_breakout_extension_atr": 0.25, "macd_direction_ok": True,
        "roc_direction_ok": True,
    }, "LONG")
    stretched = engine.score_orb({
        "rvol": 1.8, "close": 102.0, "vwap": 101.0,
        "ema9": 101.5, "ema21": 100.0, "rsi": 60.0, "atr": 0.5,
        "orb_breakout_extension_atr": 0.95, "macd_direction_ok": True,
        "roc_direction_ok": True,
    }, "LONG")
    assert clean.total >= 7.0
    assert clean.total > stretched.total


def test_normal_relative_volume_gate_uses_strength_without_direction():
    from core.engine import _normal_relative_volume_gate

    # Passing RVOL is sufficient even when the candle-direction proxy opposes
    # the signal direction or is absent entirely.
    for indicators in (
        {"rvol": 1.4, "directional_volume_proxy_ok": False},
        {"rvol": 1.4},
    ):
        assert _normal_relative_volume_gate(indicators, "LONG", 1.2) == {
            "relative_volume": True,
        }

    # The configured 1.2 threshold passes 1.39; direction proxy is irrelevant.
    weak_volume = _normal_relative_volume_gate(
        {"rvol": 1.39, "directional_volume_proxy_ok": True}, "LONG", 1.2
    )
    assert weak_volume == {"relative_volume": True}

    missing_volume = _normal_relative_volume_gate(
        {"rvol": None, "directional_volume_proxy_ok": False}, "SHORT", 1.3
    )
    assert missing_volume == {"relative_volume": False}

def test_normal_relative_volume_below_threshold_stays_watchlist_eligible():
    from core.engine import _normal_relative_volume_gate

    for rvol in (0.0, 0.75, 0.99, 1.0, 1.19):
        gates = _normal_relative_volume_gate(
            {"rvol": rvol, "directional_volume_proxy_ok": True}, "LONG", 1.2
        )
        assert gates["relative_volume"] is False
        # Engine disposition for failed RVOL is WATCHLIST,
        # never a hard rejection; next completed candle is evaluated afresh.


def test_normal_relative_volume_threshold_is_configurable():
    from config import settings

    old = settings.normal_rvol_min
    try:
        settings.normal_rvol_min = 1.5
        from core.engine import _normal_relative_volume_gate
        assert not all(_normal_relative_volume_gate(
            {"rvol": 1.4, "directional_volume_proxy_ok": True}, "LONG",
            settings.normal_rvol_min
        ).values())
    finally:
        settings.normal_rvol_min = old

def test_watchlist_removes_stale_normal_setup_when_rvol_recovers_or_core_setup_fails():
    from core.watchlist_manager import WatchlistManager

    watchlist = WatchlistManager(max_entries=10, ttl_minutes=60)
    watchlist.add({
        "pair": "B-BTC_USDT", "engine": "normal", "direction": "LONG",
        "score": 0.0, "grade": "WATCHLIST", "watch_reason": "WAITING_FOR_RVOL_1_2",
    })
    watchlist.add({
        "pair": "B-BTC_USDT", "engine": "normal", "direction": "SHORT",
        "score": 0.0, "grade": "WATCHLIST", "watch_reason": "WAITING_FOR_RVOL_1_2",
    })
    watchlist.add({
        "pair": "B-BTC_USDT", "engine": "momentum", "direction": "LONG",
        "score": 5.5, "grade": "WATCHLIST",
    })

    removed = watchlist.remove_pair("B-BTC_USDT", "normal")

    assert removed == 2
    assert watchlist.get("B-BTC_USDT", "normal", "LONG") is None
    assert watchlist.get("B-BTC_USDT", "normal", "SHORT") is None
    # Normal cleanup must not touch the independent Momentum watchlist.
    assert watchlist.get("B-BTC_USDT", "momentum", "LONG") is not None


def test_normal_rvol_gate_rejects_nonfinite_and_nonpositive_thresholds():
    from core.engine import _normal_relative_volume_gate

    for value in (float("nan"), float("inf"), -float("inf"), None):
        gates = _normal_relative_volume_gate(
            {"rvol": value, "directional_volume_proxy_ok": True}, "LONG", 1.2
        )
        assert gates["relative_volume"] is False

    assert _normal_relative_volume_gate(
        {"rvol": 2.0, "directional_volume_proxy_ok": True}, "LONG", 0.0
    )["relative_volume"] is False



def test_normal_rsi_gate_requires_directional_midline_confirmation():
    from core.engine import _normal_threshold_gates

    long_values = {
        "ema9": 101.0, "ema21": 100.0, "close": 102.0, "vwap": 101.0,
        "adx": 25.0, "atr": 0.5, "confirm_15m": True, "rsi": 55.0,
    }
    short_values = {
        "ema9": 99.0, "ema21": 100.0, "close": 98.0, "vwap": 99.0,
        "adx": 25.0, "atr": 0.5, "confirm_15m": True, "rsi": 45.0,
    }
    assert all(_normal_threshold_gates(long_values, "LONG", 20.0).values())
    assert all(_normal_threshold_gates(short_values, "SHORT", 20.0).values())
    assert not _normal_threshold_gates({**long_values, "rsi": 48.0}, "LONG", 20.0)["rsi_direction"]
    assert not _normal_threshold_gates({**short_values, "rsi": 52.0}, "SHORT", 20.0)["rsi_direction"]


def test_normal_rsi_scoring_matches_directional_midline_gate():
    result = ScoringEngine().score_normal({
        "close": 102.0, "ema9": 101.0, "ema21": 100.0, "vwap": 101.0,
        "adx": 30.0, "rsi": 58.0, "atr": 0.5, "confirm_15m": True,
    }, "LONG")
    rsi_component = next(c for c in result.components if c.name == "RSI 14")
    assert rsi_component.points == 1.5
    opposite = ScoringEngine().score_normal({
        "close": 102.0, "ema9": 101.0, "ema21": 100.0, "vwap": 101.0,
        "adx": 30.0, "rsi": 38.0, "atr": 0.5, "confirm_15m": True,
    }, "LONG")
    assert next(c for c in opposite.components if c.name == "RSI 14").points == 0.0


def test_orb_scoring_uses_indicator_edge_proximity_not_candle_shape():
    result = ScoringEngine().score_orb({
        "rvol": 1.8, "close": 98.0, "vwap": 99.0,
        "ema9": 99.0, "ema21": 100.0, "rsi": 42.0, "atr": 0.5,
        "orb_breakout_extension_atr": 0.0,
        "macd_direction_ok": True, "roc_direction_ok": True,
    }, "SHORT")
    proximity = next(c for c in result.components if c.name == "ORB Edge Proximity")
    assert proximity.points == 1.0


def test_normal_adx_score_uses_only_the_adx_threshold():
    engine = ScoringEngine()
    base = {
        "close": 102.0, "ema9": 101.0, "ema21": 100.0, "vwap": 101.0,
        "adx": 32.0, "rsi": 58.0, "atr": 0.5, "confirm_15m": True,
    }
    score = engine.score_normal(base, "LONG")
    adx_component = next(c for c in score.components if c.name == "ADX 14")
    assert adx_component.points == 2.0
    weaker = engine.score_normal({**base, "adx": 20.0}, "LONG")
    weaker_adx = next(c for c in weaker.components if c.name == "ADX 14")
    assert weaker_adx.points == 0.0
    assert not weaker_adx.valid
    assert "below 25" in weaker_adx.explanation


def test_momentum_macd_direction_scores_without_candle_body():
    engine = ScoringEngine()
    aligned = engine.score_momentum({"macd_direction_ok": True}, "LONG")
    component = next(c for c in aligned.components if c.name == "MACD")
    assert component.points == 1.5
    assert component.valid


def test_normal_classifier_uses_indicator_states_only():
    from strategies.normal.strategy_manager import detect_strategy

    indicators = {
        "close": 102.0, "ema9": 101.0, "ema21": 100.0,
        "vwap": 101.0, "rsi": 58.0, "adx": 25.0, "atr": 0.5,
    }
    assert detect_strategy(indicators, "LONG") == "TREND_CONTINUATION"


def test_all_scoring_modes_fail_closed_on_malformed_and_nonfinite_indicators():
    import math

    engine = ScoringEngine()
    malformed = {
        "ema9": "not-a-number",
        "ema21": float("inf"),
        "close": float("nan"),
        "vwap": "bad",
        "adx": 140,
        "rsi": -1,
        "rsi_prev": float("-inf"),
        "rvol": "unknown",
        "atr_pct": float("nan"),
        "body_ratio": 1.5,
        "atr_expansion": float("inf"),
        "close_location": -0.2,
        "orb_breakout_extension_atr": float("nan"),
        "orb_range_width_atr": -1,
        "roc": "broken",
        "squeeze_release": "false",
        "bw_expand": "false",
        "ema_aligned": "false",
        "macd_accel": "false",
        "macd_direction_ok": "false",
        "rsi_accel": "false",
        "structure_break": "false",
        "roc_direction_ok": "false",
        "confirm_15m": "false",
        "structure_direction_ok": "false",
    }
    results = (
        engine.score_normal(malformed, "LONG"),
        engine.score_momentum(malformed, "LONG"),
        engine.score_momentum_explosion(malformed, "LONG"),
        engine.score_orb(malformed, "LONG"),
    )
    for result in results:
        assert math.isfinite(result.total)
        assert 0.0 <= result.total <= 10.0
        assert all(math.isfinite(component.points) for component in result.components)

    # Malformed truthy strings must never count as confirmation.
    normal = engine.score_normal(malformed, "LONG")
    assert all("15m" not in c.name for c in normal.components)
    assert next(c for c in normal.components if c.name == "EMA 9/21").points == 0.0
    momentum = engine.score_momentum(malformed, "LONG")
    assert next(c for c in momentum.components if c.name == "Squeeze State").points == 0.0
    assert next(c for c in momentum.components if c.name == "EMA Alignment").points == 0.0


def test_scoring_modes_reject_unknown_direction():
    import pytest

    engine = ScoringEngine()
    for method in (
        engine.score_normal,
        engine.score_momentum,
        engine.score_momentum_explosion,
        engine.score_orb,
    ):
        with pytest.raises(ValueError, match="direction must be LONG or SHORT"):
            method({}, "BUY")
