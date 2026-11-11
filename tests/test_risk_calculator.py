"""Entry-state and gross ATR risk tests."""
from core.risk_calculator import calculate_risk, classify_entry_state


def test_entry_state_immediate_retest_and_expired():
    assert classify_entry_state(100.0, 100.0, 99.0, 100.5, "LONG") == "IMMEDIATE"
    assert classify_entry_state(101.0, 100.0, 99.0, 100.5, "LONG") == "RETEST"
    assert classify_entry_state(103.0, 100.0, 99.0, 100.5, "LONG") == "EXPIRED"
    assert classify_entry_state(98.0, 100.0, 99.5, 101.0, "SHORT") == "RETEST"


def test_risk_levels_report_gross_reward_risk_without_fee_adjustments():
    risk = calculate_risk(100.0, 2.0, "LONG", swing_low=98.0)
    assert risk is not None
    assert risk.rr1 == 2.0
    assert risk.rr2 == 3.0
    assert risk.rr3 == 4.0
    distance = risk.entry - risk.stop_loss
    assert risk.tp1 == risk.entry + 2.0 * distance
    assert risk.tp2 == risk.entry + 3.0 * distance
    assert risk.tp3 == risk.entry + 4.0 * distance


def test_configured_reward_multipliers_are_applied_as_r_multiples():
    risk = calculate_risk(
        100.0, 2.0, "LONG",
        atr_mult_tp1=1.0, atr_mult_tp2=2.0, atr_mult_tp3=3.0, swing_low=98.0,
    )
    assert risk is not None
    assert risk.rr1 == 1.0
    assert risk.rr2 == 2.0
    assert risk.rr3 == 3.0


def test_invalid_reward_multipliers_fail_closed():
    assert calculate_risk(100.0, 2.0, "LONG", atr_mult_tp1=2.0,
                          atr_mult_tp2=1.0, atr_mult_tp3=3.0) is None
    assert calculate_risk(100.0, 2.0, "LONG", atr_mult_tp1=float("nan")) is None


def test_extreme_volatility_rejects_non_positive_stop_or_target_prices():
    assert calculate_risk(100.0, 30.0, "SHORT", swing_high=110.0) is None
    assert calculate_risk(100.0, 50.0, "LONG", swing_low=50.0) is None


def test_extreme_volatility_rejects_non_positive_entry_zone():
    assert calculate_risk(10.0, 100.0, "LONG", swing_low=5.0) is None
    assert calculate_risk(10.0, 100.0, "SHORT", swing_high=20.0) is None


def test_short_risk_levels_use_the_same_gross_reward_multipliers():
    risk = calculate_risk(100.0, 2.0, "SHORT", swing_high=102.0)
    assert risk is not None
    assert risk.rr1 == 2.0
    assert risk.rr2 == 3.0
    assert risk.rr3 == 4.0
    distance = risk.stop_loss - risk.entry
    assert risk.tp1 == risk.entry - 2.0 * distance
    assert risk.tp2 == risk.entry - 3.0 * distance
    assert risk.tp3 == risk.entry - 4.0 * distance
