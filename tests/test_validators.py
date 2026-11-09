"""Validation helper regression tests."""
from utils.validators import normalize_pair, safe_float


def test_normalize_pair_preserves_all_supported_coin_dcx_prefixes():
    assert normalize_pair("BTC/USDT") == "B-BTC_USDT"
    assert normalize_pair("B-BTC_USDT") == "B-BTC_USDT"
    assert normalize_pair("I-BTC_USDT") == "I-BTC_USDT"
    assert normalize_pair("HB-BTC_USDT") == "HB-BTC_USDT"
    assert normalize_pair("KC-BTC_USDT") == "KC-BTC_USDT"


def test_safe_float_rejects_nan_and_infinity():
    assert safe_float("12.5") == 12.5
    assert safe_float("nan", 7.0) == 7.0
    assert safe_float("inf", 7.0) == 7.0
    assert safe_float(None, 7.0) == 7.0
