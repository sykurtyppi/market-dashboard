"""Regression tests for the liquidity_history pipeline.

Covers the two defects found auditing the phase-10b wiring commit (004d5d0):

1. Unit normalization inferred FRED units from magnitude, which misreads a
   low TGA (debt-ceiling regime) as billions and inflates it 1000x.
2. `row.get('rrp_on') or row.get('rrp')` treats a legitimate 0.0 as absent,
   nulling net liquidity exactly when ON RRP is drained.
"""

import pandas as pd
import pytest

from scheduler.daily_update import MarketDataUpdater


class TestUnitNormalization:
    """FRED units are a fixed property of each series, never a magnitude guess."""

    @pytest.mark.parametrize(
        "field, raw, expected_billions",
        [
            # WTREGEN (TGA) — Millions of USD
            ("tga", 910_776.0, 910.776),      # typical
            ("tga", 46_000.0, 46.0),          # Oct-2021 debt-ceiling low
            ("tga", 23_000.0, 23.0),          # Jun-2023 debt-ceiling low
            # WALCL (Fed balance sheet) — Millions of USD
            ("fed_bs", 6_738_190.0, 6738.19),
            # RRPONTSYD (ON RRP) — already Billions of USD
            ("rrp_on", 2.576, 2.576),
            ("rrp_on", 2_554.0, 2_554.0),     # Dec-2022 peak
            ("rrp_on", 0.03, 0.03),           # drained
        ],
    )
    def test_series_converted_by_documented_unit(self, field, raw, expected_billions):
        got = MarketDataUpdater._to_billions(pd.Series([raw]), field).iloc[0]
        assert got == pytest.approx(expected_billions, rel=1e-9)

    def test_low_tga_is_not_misread_as_billions(self):
        """The original magnitude heuristic (<=50_000 assumed billions) left a
        $23B TGA at 23_000, producing net liquidity near -$16T."""
        tga = MarketDataUpdater._to_billions(pd.Series([23_000.0]), "tga").iloc[0]
        fed_bs, rrp = 6738.19, 2.6
        net_liquidity = fed_bs - tga - rrp
        assert tga == pytest.approx(23.0)
        assert net_liquidity > 0
        assert net_liquidity == pytest.approx(6712.59, abs=0.01)

    def test_nan_is_preserved_not_coerced(self):
        out = MarketDataUpdater._to_billions(pd.Series([910_776.0, None]), "tga")
        assert out.iloc[0] == pytest.approx(910.776)
        assert pd.isna(out.iloc[1])

    def test_implausible_values_warn_but_are_not_rescaled(self, caplog):
        """A unit change upstream must surface, not be silently corrected."""
        with caplog.at_level("WARNING"):
            out = MarketDataUpdater._to_billions(pd.Series([9_999_999.0]), "rrp_on")
        assert out.iloc[0] == pytest.approx(9_999_999.0)  # unchanged
        assert any("outside the plausible" in r.message for r in caplog.records)


class TestNetLiquidityFieldResolution:
    """Zero is a value, not an absence."""

    @staticmethod
    def _resolve(row, *names):
        """Mirrors _first_present in DatabaseManager.save_liquidity_history."""
        for n in names:
            v = row.get(n)
            if v is not None and pd.notna(v):
                return v
        return None

    @pytest.mark.parametrize("rrp", [2.576, 0.03, 0.0])
    def test_zero_rrp_still_yields_net_liquidity(self, rrp):
        row = pd.Series({"rrp_on": rrp, "tga": 911.0, "fed_bs": 6738.19})
        resolved = self._resolve(row, "rrp_on", "rrp")
        assert resolved == pytest.approx(rrp)
        net = float(row["fed_bs"]) - float(row["tga"]) - float(resolved)
        assert net == pytest.approx(6738.19 - 911.0 - rrp)

    def test_missing_field_falls_through_to_alias(self):
        row = pd.Series({"rrp_on": float("nan"), "rrp": 5.0})
        assert self._resolve(row, "rrp_on", "rrp") == pytest.approx(5.0)

    def test_all_absent_resolves_to_none(self):
        row = pd.Series({"rrp_on": float("nan")})
        assert self._resolve(row, "rrp_on", "rrp") is None


class TestNetLiquidityFormula:
    def test_uses_fed_bs_minus_tga_minus_rrp(self):
        """Net liquidity is a level, not the negated drain of the fallback."""
        fed_bs, tga, rrp = 6738.19, 910.776, 2.576
        assert fed_bs - tga - rrp == pytest.approx(5824.838, abs=0.001)
