# Invalid VIX tenor regressions.
"""Regression: an invalid (non-numeric / non-finite) CBOE tenor value must be
rejected by the _num helper so it takes the existing missing-tenor path — a
warning is recorded and adjacent curve comparisons are broken — while valid
tenors, including numeric strings, are preserved."""
import json

from unittest.mock import patch

import api.sectors_service as svc


def test_invalid_middle_vix_tenor_is_skipped_with_valid_neighbors():
    svc._cache.clear()
    try:
        with patch("data_collectors.sector_collector.SectorCollector") as MockSC, \
             patch("data_collectors.cboe_collector.CBOECollector") as MockCBOE:
            MockSC.return_value.get_sector_performance.return_value = {}
            MockSC.return_value.get_rotation_signal.return_value = {}
            # Valid neighbors (including a numeric string for VIX9D), invalid middle.
            MockCBOE.return_value.get_vix9d.return_value = "12.0"
            MockCBOE.return_value.get_vix.return_value = 15.0
            MockCBOE.return_value.get_vix3m.return_value = "not-a-number"
            MockCBOE.return_value.get_vix6m.return_value = 19.5
            MockCBOE.return_value.get_vix1y.return_value = 20.0
            MockCBOE.return_value.estimated_fields = []
            MockCBOE.return_value.stale_fields = []
            body = svc._build_live()

        # The invalid middle tenor is skipped; valid tenors (incl. numeric string) retained.
        symbols = [point["symbol"] for point in body["vix_term"]]
        expected = ["VIX9D", "VIX", "VIX6M", "VIX1Y"]
        assert symbols == expected

        # Numeric string neighbor preserved as a float value.
        vix9d = next(point for point in body["vix_term"] if point["symbol"] == "VIX9D")
        assert vix9d["value"] == 12.0

        # The missing middle tenor breaks adjacent comparisons and warns.
        six_month = next(point for point in body["vix_term"] if point["symbol"] == "VIX6M")
        assert six_month["change"] is None
        assert six_month["change_pct"] is None
        assert any("VIX3M" in warning and "unavailable" in warning for warning in body["warnings"])

        # Payload remains strict-JSON serializable (no inf/nan leaked in).
        json.dumps(body, allow_nan=False)
    finally:
        svc._cache.clear()
