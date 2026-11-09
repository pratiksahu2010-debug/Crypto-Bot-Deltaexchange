"""Regression tests for Phase 1 signal configuration and scoring fixes."""
import pytest

from core.engine import _normal_relative_volume_gate
from core.scoring_engine import ScoringEngine


@pytest.mark.parametrize(
    ("rvol", "threshold", "expected"),
    [
        (1.39, 1.4, False),
        (1.4, 1.4, True),
        (1.2, 1.2, True),
        (1.19, 1.2, False),
        (None, 1.4, False),
        (float("nan"), 1.4, False),
    ],
)
def test_normal_rvol_gate_uses_configured_threshold(rvol, threshold, expected):
    result = _normal_relative_volume_gate({"rvol": rvol}, "LONG", threshold)
    assert result["relative_volume"] is expected


def test_normal_score_reports_actual_8_5_point_maximum():
    result = ScoringEngine().score_normal(
        {
            "ema9": 110.0,
            "ema21": 100.0,
            "close": 112.0,
            "vwap": 105.0,
            "rsi": 60.0,
            "adx": 30.0,
            "atr": 2.0,
        },
        "LONG",
    )
    assert result.total == pytest.approx(8.5)
    assert result.max_total == pytest.approx(8.5)
    assert sum(component.max_points for component in result.components) == pytest.approx(8.5)


def test_normal_score_grade_uses_its_actual_maximum():
    result = ScoringEngine().score_normal(
        {
            "ema9": 110.0,
            "ema21": 100.0,
            "close": 112.0,
            "vwap": 105.0,
            "rsi": 60.0,
            "adx": 30.0,
            "atr": 2.0,
        },
        "LONG",
    )
    assert result.grade == "HIGH-CONVICTION"
