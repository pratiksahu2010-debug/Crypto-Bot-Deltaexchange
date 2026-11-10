"""Regression tests for detector-owned Momentum and ORB qualification."""

import pytest

from core.engine import _momentum_entry_gates


@pytest.mark.parametrize(
    "state",
    [
        "EXPLOSION",
        "SQUEEZE_RELEASE",
        "PRE_EXPLOSION_SIGNAL",
        "CONTINUATION",
        "VOLUME_SPIKE",
        "ORB_PREBREAKOUT",
        "ORB_BREAKOUT",
        "ORB_RETEST",
    ],
)
def test_qualified_momentum_and_orb_states_need_only_shared_anti_chase_gate(state):
    # Directional MACD/ROC/EMA/RVOL/range checks belong to the state detector.
    # This shared layer must not impose a second universal MACD, ADX/DI or RVOL veto.
    assert _momentum_entry_gates(state, 0.5, 1.0) == {
        "momentum_state": True,
        "anti_chase": True,
    }


def test_unqualified_momentum_fallback_state_is_not_alertable():
    assert _momentum_entry_gates("MOMENTUM_SETUP", 0.5, 1.0)["momentum_state"] is False


@pytest.mark.parametrize("extension", [2.0001, float("nan"), -0.01])
def test_momentum_and_orb_keep_shared_ema_extension_protection(extension):
    gates = _momentum_entry_gates("ORB_PREBREAKOUT", 0.5, extension)
    assert gates["momentum_state"] is True
    assert gates["anti_chase"] is False


def test_momentum_requires_valid_atr_before_alert():
    assert _momentum_entry_gates("EXPLOSION", 0.0, 0.5)["anti_chase"] is False
    assert _momentum_entry_gates("EXPLOSION", float("nan"), 0.5)["anti_chase"] is False
