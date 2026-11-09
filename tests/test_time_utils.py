"""Time utility regression tests."""
import pytest

from utils.time_utils import parse_interval


@pytest.mark.parametrize(("value", "seconds"), [
    ("5m", 300), ("15m", 900), ("1h", 3600), ("1d", 86400), ("1w", 604800),
])
def test_parse_interval_accepts_positive_supported_intervals(value, seconds):
    assert parse_interval(value).total_seconds() == seconds


@pytest.mark.parametrize("value", ["", "0m", "-5m", "5s", "abc", "5M!", None])
def test_parse_interval_rejects_invalid_or_nonpositive_intervals(value):
    with pytest.raises(ValueError):
        parse_interval(value)
