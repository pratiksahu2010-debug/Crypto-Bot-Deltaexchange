from core.engine import _normal_relative_volume_gate


def test_normal_rvol_gate_does_not_require_directional_candle():
    indicators = {
        "rvol": 1.4,
        "directional_volume_proxy_ok": False,
    }
    assert _normal_relative_volume_gate(indicators, "LONG", 1.2) == {
        "relative_volume": True,
    }


def test_normal_low_rvol_remains_watchlist_eligible():
    indicators = {
        "rvol": 1.39,
        "directional_volume_proxy_ok": True,
    }
    # 1.39 clears the explicitly configured 1.2 threshold.
    assert _normal_relative_volume_gate(indicators, "SHORT", 1.2) == {
        "relative_volume": True,
    }


def test_normal_recommended_rvol_floor_rejects_small_participation():
    assert _normal_relative_volume_gate({"rvol": 1.39}, "LONG", 1.4) == {
        "relative_volume": False,
    }
    assert _normal_relative_volume_gate({"rvol": 1.4}, "LONG", 1.4) == {
        "relative_volume": True,
    }
