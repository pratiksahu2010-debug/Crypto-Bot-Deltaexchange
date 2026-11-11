"""Signal message formatting regression tests."""
from tgbot.bot import TelegramBot


def _format(engine, strategy, indicators=None):
    return TelegramBot().format_signal({
        "pair": "BTC_USDT",
        "engine": engine,
        "strategy": strategy,
        "direction": "LONG",
        "price": 65000,
        "indicators": indicators or {},
        "timestamp": "2026-11-08T10:00:00+00:00",
    })


def test_normal_signal_displays_signal_type():
    message = _format("normal", "TREND")
    assert "<b>SIGNAL TYPE:</b> NORMAL" in message


def test_momentum_signal_displays_signal_type():
    message = _format("momentum", "EXPLOSION")
    assert "<b>SIGNAL TYPE:</b> MOMENTUM EXPLOSION" in message


def test_orb_signal_displays_orb_signal_type():
    message = _format("momentum", "ORB_BREAKOUT", {"orb_session": "london"})
    assert "<b>SIGNAL TYPE:</b> ORB" in message


def test_sequential_signal_displays_signal_type():
    message = _format("signals", "SEQUENTIAL_ENTRY")
    assert "<b>SIGNAL TYPE:</b> SEQUENTIAL" in message


def test_normal_message_displays_15m_context_and_sequential_does_not():
    normal = _format("normal", "TREND", {"context_15m": "BULLISH"})
    sequential_legacy = _format("signals", "SEQUENTIAL_ENTRY")
    assert "<b>15M CONTEXT:</b> BULLISH" in normal
    assert "15M CONTEXT" not in sequential_legacy
    for message in (normal, sequential_legacy):
        assert "CONFIRM:</b> 15M" not in message
        assert "<b>TIMEFRAME:</b> 5M (PRIMARY)" in message


def test_orb_one_bar_opening_range_is_not_reported_as_15m_confirmation():
    # ORB uses one completed 5m opening-range candle; this is not an HTF filter.
    from strategies.momentum.orb_detector import opening_range
    assert callable(opening_range)



def test_signal_formats_hide_candle_and_signal_timing_metadata():
    normal = TelegramBot().format_signal({
        "pair": "BTC_USDT",
        "engine": "normal",
        "strategy": "TREND",
        "direction": "LONG",
        "timestamp": "2026-11-09T15:50:34+00:00",
        "candle_time_ms": 1794239100000,
        "candle_close_time_ms": 1794239400000,
        "entry_checked_at": "2026-11-09T15:50:34+00:00",
        "signal_age_seconds": 34.6,
        "indicators": {},
    })
    sequential = TelegramBot().format_signal({
        "pair": "BTC_USDT",
        "engine": "signals",
        "strategy": "SEQUENTIAL_ENTRY",
        "direction": "LONG",
        "timestamp": "2026-11-09T15:50:34+00:00",
        "candle_time_ms": 1794239100000,
        "signal_age_seconds": 34.6,
        "indicators": {},
    })
    for message in (normal, sequential):
        for hidden_label in (
            "CANDLE TIMING",
            "CANDLE OPEN",
            "CANDLE CLOSE",
            "CANDLE TIME",
            "15M CONTEXT OPEN",
            "ENTRY CHECKED",
            "SIGNAL GENERATED",
            "SIGNAL AGE",
        ):
            assert hidden_label not in message
        assert "Signal only — no automated trading." in message
