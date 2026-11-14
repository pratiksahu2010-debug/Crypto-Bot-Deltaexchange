"""Regression tests for the MACD-free Normal Engine strategy stack."""


def test_normal_trend_strategy_does_not_require_macd_for_long_or_short():
    from strategies.normal.strategy_manager import detect_strategy

    long_indicators = {
        "ema9": 103.0,
        "ema21": 102.0,
        "ema50": 101.0,
        "adx": 25.0,
        "close": 105.0,
        "vwap": 104.0,
        # Normal Engine intentionally does not calculate MACD.
        "macd_hist": None,
    }
    short_indicators = {
        "ema9": 97.0,
        "ema21": 98.0,
        "ema50": 99.0,
        "adx": 25.0,
        "close": 95.0,
        "vwap": 96.0,
        "macd_hist": None,
    }

    assert detect_strategy(long_indicators, "LONG") == "TREND_CONTINUATION"
    assert detect_strategy(short_indicators, "SHORT") == "TREND_CONTINUATION"


def test_normal_trend_strategy_ignores_opposing_macd_value():
    from strategies.normal.trend import evaluate

    indicators = {
        "ema9": 103.0,
        "ema21": 102.0,
        "ema50": 101.0,
        "adx": 25.0,
        "close": 105.0,
        "vwap": 104.0,
        "macd_hist": -999.0,
    }

    assert evaluate(indicators, "LONG") is True


def test_normal_strategy_diagnostics_report_all_predicate_results():
    from strategies.normal.strategy_manager import diagnose_strategy

    checks = diagnose_strategy({"ema9": 2.0, "ema21": 1.0}, "LONG")
    assert set(checks) == {
        "BREAKOUT", "TREND_CONTINUATION", "EMA_PULLBACK", "REVERSAL"
    }
    assert not any(checks.values())


def test_normal_trend_strategy_uses_adx_17_floor():
    from strategies.normal.trend import evaluate

    base = {
        "ema9": 103.0, "ema21": 102.0, "ema50": 101.0,
        "close": 105.0, "vwap": 104.0,
    }
    assert evaluate({**base, "adx": 17.0}, "LONG") is True
    assert evaluate({**base, "adx": 16.99}, "LONG") is False



def test_normal_core_gates_are_indicator_led_not_candle_colour_led():
    from core.engine import _normal_threshold_gates

    indicators = {
        "ema9": 101.0, "ema21": 100.0, "close": 102.0, "vwap": 101.0,
        "rsi": 58.0, "adx": 25.0, "atr": 0.5, "confirm_15m": True,
        "directional_candle": False,
    }
    gates = _normal_threshold_gates(indicators, "LONG", 20.0)
    assert all(gates.values())
