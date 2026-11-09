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
