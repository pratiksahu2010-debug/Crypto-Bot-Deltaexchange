"""Defensive indicator validation regression tests."""
import numpy as np

from indicators import atr, ema, ema_series, rsi, macd, roc, vwap
from indicators.adx import adx
from indicators.bollinger import bollinger_bands
from indicators.keltner import keltner_channels


def sample_ohlc(size=80):
    close = np.linspace(100.0, 120.0, size)
    high = close + 1.0
    low = close - 1.0
    return high, low, close


def test_ema_rejects_nonfinite_values_and_invalid_periods():
    assert ema([1.0, 2.0, float("nan")], 2) is None
    assert ema_series([1.0, 2.0, 3.0], 0) is None
    assert ema([1.0, 2.0], 3) is None


def test_atr_and_adx_reject_malformed_ohlc():
    high, low, close = sample_ohlc()
    assert atr(high[:-1], low, close, 14) is None
    assert atr(high, low, np.where(np.arange(len(close)) == 10, np.nan, close), 14) is None
    assert atr(high, low, close, 0) is None
    assert adx(high, low, close, 0) is None
    invalid_high = high.copy()
    invalid_high[20] = low[20] - 1
    assert adx(invalid_high, low, close, 14) is None


def test_rsi_bollinger_and_roc_reject_invalid_inputs():
    _, _, close = sample_ohlc()
    bad = close.copy()
    bad[-1] = np.inf
    assert rsi(bad, 14) is None
    assert rsi(close, 0) is None
    assert bollinger_bands(bad, 20, 2.0) is None
    assert bollinger_bands(close, 0, 2.0) is None
    assert roc([0.0, 1.0, 2.0], 2) is None


def test_macd_accepts_minimum_valid_history_and_rejects_bad_period_order():
    _, _, close = sample_ohlc(34)
    assert macd(close, 12, 26, 9) is not None
    assert macd(close, 26, 12, 9) is None


def test_keltner_and_vwap_reject_invalid_parameters_and_ohlc():
    high, low, close = sample_ohlc()
    assert keltner_channels(high, low, close, 20, 0) is None
    bad_close = close.copy()
    bad_close[5] = high[5] + 1
    assert vwap(high, low, bad_close, np.ones_like(close), period=None) is None
    assert vwap(high, low, close, np.ones_like(close), period=0) is None



def test_relative_volume_rejects_malformed_shapes_periods_and_values():
    from indicators.volume import relative_volume

    assert relative_volume(np.ones((21, 1)), 20) is None
    assert relative_volume(np.ones(21), 0) is None
    assert relative_volume(np.ones(21), 1.5) is None
    assert relative_volume(np.ones(21), True) is None
    assert relative_volume([1.0, 2.0, float("nan")], 2) is None
    assert relative_volume([1.0, -1.0, 2.0], 2) is None
    assert relative_volume(np.zeros(21), 20) is None


def test_relative_volume_uses_only_prior_bars_for_baseline():
    from indicators.volume import relative_volume

    assert relative_volume([10.0] * 20 + [20.0], 20) == 2.0
