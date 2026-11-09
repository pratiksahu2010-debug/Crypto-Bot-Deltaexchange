"""Regression tests: indicator-qualified signals are not delayed by extra price-action vetoes."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = (ROOT / "core" / "engine.py").read_text(encoding="utf-8")
SCORING = (ROOT / "core" / "scoring_engine.py").read_text(encoding="utf-8")


def test_second_completed_candle_follow_through_is_not_required():
    assert "_signal_follow_through_confirmed" not in ENGINE
    assert "awaiting_second_closed_candle" not in ENGINE
    assert "second_candle_no_follow_through" not in ENGINE


def test_15m_context_is_not_a_normal_signal_veto_or_score_component():
    assert '"context_15m": indicators.get("confirm_15m") is True' not in ENGINE
    assert '"15m EMA Context"' not in SCORING


def test_two_risk_reward_and_opposing_pivot_room_do_not_reject_signals():
    assert "risk.rr1 < 2.0" not in ENGINE
    assert "_has_room_for_2r_target" not in ENGINE
    # Risk calculations still populate stop loss and target values for alerts.
    assert '"stop_loss": risk.stop_loss' in ENGINE
    assert '"tp1": risk.tp1' in ENGINE


def test_normal_engine_does_not_use_macd_as_a_required_indicator():
    normal_section = ENGINE.split('if engine == "normal":', 1)[1].split("else:", 1)[0]
    assert 'macd(' not in normal_section
