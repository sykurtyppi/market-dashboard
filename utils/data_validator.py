"""
Comprehensive Data Validation Layer for Market Dashboard

This module provides schema validation for all data before it enters the database.
It catches malformed, out-of-range, and invalid data before it corrupts the database.

Usage:
    from utils.data_validator import DataValidator, ValidationResult

    validator = DataValidator()
    result = validator.validate_daily_snapshot(snapshot_dict)

    if result.is_valid:
        db.save_daily_snapshot(result.data)
    else:
        logger.error(f"Validation failed: {result.errors}")
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation operation"""
    is_valid: bool
    data: Optional[Dict] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    estimated_fields: List[str] = field(default_factory=list)  # Track which fields are estimates

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def mark_estimated(self, field_name: str):
        """Mark a field as containing estimated (not real) data"""
        self.estimated_fields.append(field_name)


class DataValidator:
    """
    Centralized data validation for all market data.

    Validates data types, ranges, and business rules before database saves.
    Tracks which fields contain estimated vs real data.
    """

    # ========================================
    # VALIDATION SCHEMAS
    # ========================================

    # VIX-related ranges (annualized volatility %)
    VIX_MIN = 5.0
    VIX_MAX = 100.0
    VVIX_MIN = 50.0
    VVIX_MAX = 200.0
    SKEW_MIN = 100.0
    SKEW_MAX = 180.0

    # Credit spreads (basis points / 100 = %)
    CREDIT_SPREAD_MIN = 0.5   # 50 bps
    CREDIT_SPREAD_MAX = 25.0  # 2500 bps

    # Interest rates (%)
    RATE_MIN = -2.0   # Negative rates exist
    RATE_MAX = 20.0

    # Liquidity (in billions)
    LIQUIDITY_MIN = -10000.0  # Can be negative (drain)
    LIQUIDITY_MAX = 20000.0   # ~$20T max

    # MOVE Index
    MOVE_MIN = 30.0
    MOVE_MAX = 250.0

    # Breadth (%)
    BREADTH_MIN = 0.0
    BREADTH_MAX = 100.0

    # VRP (Volatility Risk Premium)
    VRP_MIN = -30.0
    VRP_MAX = 50.0

    # Put/Call ratio
    PC_RATIO_MIN = 0.1
    PC_RATIO_MAX = 3.0

    # Fear & Greed
    FEAR_GREED_MIN = 0.0
    FEAR_GREED_MAX = 100.0

    # ========================================
    # CORE VALIDATION METHODS
    # ========================================

    def _validate_numeric(
        self,
        value: Any,
        field_name: str,
        min_val: float,
        max_val: float,
        required: bool = False
    ) -> Tuple[Optional[float], Optional[str], Optional[str]]:
        """
        Validate a numeric value.

        Returns:
            Tuple of (validated_value, error_message, warning_message)
        """
        # Handle None
        if value is None:
            if required:
                return None, f"{field_name}: required field is missing", None
            return None, None, None

        # Convert to float
        try:
            val = float(value)
        except (ValueError, TypeError):
            return None, f"{field_name}: cannot convert '{value}' to number", None

        # Check for NaN/Inf
        if math.isnan(val) or math.isinf(val):
            return None, f"{field_name}: value is NaN or Infinite", None

        # Check range
        if val < min_val or val > max_val:
            return None, f"{field_name}: {val} outside valid range [{min_val}, {max_val}]", None

        return val, None, None

    def _validate_string(
        self,
        value: Any,
        field_name: str,
        allowed_values: Optional[List[str]] = None,
        required: bool = False
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Validate a string value.

        Returns:
            Tuple of (validated_value, error_message)
        """
        if value is None:
            if required:
                return None, f"{field_name}: required field is missing"
            return None, None

        val = str(value).strip()

        if not val and required:
            return None, f"{field_name}: empty string not allowed"

        if allowed_values and val not in allowed_values:
            return None, f"{field_name}: '{val}' not in allowed values {allowed_values}"

        return val, None

    def _validate_date(
        self,
        value: Any,
        field_name: str,
        required: bool = True
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Validate and normalize a date value to YYYY-MM-DD string.

        Returns:
            Tuple of (validated_date_string, error_message)
        """
        if value is None:
            if required:
                return None, f"{field_name}: required date is missing"
            return None, None

        # Already a string in correct format
        if isinstance(value, str):
            try:
                datetime.strptime(value, '%Y-%m-%d')
                return value, None
            except ValueError:
                return None, f"{field_name}: invalid date format '{value}', expected YYYY-MM-DD"

        # datetime or date object
        if isinstance(value, (datetime, date)):
            return value.strftime('%Y-%m-%d'), None

        # pandas Timestamp
        if hasattr(value, 'strftime'):
            try:
                return value.strftime('%Y-%m-%d'), None
            except Exception:
                pass

        return None, f"{field_name}: cannot parse date from {type(value)}: {value}"

    # ========================================
    # TABLE-SPECIFIC VALIDATORS
    # ========================================

    def validate_daily_snapshot(self, snapshot: Dict) -> ValidationResult:
        """
        Validate data for daily_snapshots table.

        Expected fields:
            - date (required)
            - credit_spread_hy, credit_spread_ig
            - treasury_10y, fed_funds
            - vix_spot, vix9d, vvix, vvix_signal, skew
            - vrp, vix_contango, put_call_ratio
            - fear_greed_score, market_breadth
            - left_signal
        """
        result = ValidationResult(is_valid=True, data={})

        # Date (required)
        date_val, date_err = self._validate_date(snapshot.get('date'), 'date', required=True)
        if date_err:
            result.add_error(date_err)
        else:
            result.data['date'] = date_val

        # Credit spreads
        for field in ['credit_spread_hy', 'credit_spread_ig']:
            val, err, warn = self._validate_numeric(
                snapshot.get(field), field,
                self.CREDIT_SPREAD_MIN, self.CREDIT_SPREAD_MAX
            )
            if err:
                result.add_error(err)
            else:
                result.data[field] = val
            if warn:
                result.add_warning(warn)

        # Interest rates
        for field in ['treasury_10y', 'fed_funds']:
            val, err, warn = self._validate_numeric(
                snapshot.get(field), field,
                self.RATE_MIN, self.RATE_MAX
            )
            if err:
                result.add_error(err)
            else:
                result.data[field] = val

        # VIX family
        vix_val, vix_err, _ = self._validate_numeric(
            snapshot.get('vix_spot'), 'vix_spot',
            self.VIX_MIN, self.VIX_MAX
        )
        if vix_err:
            result.add_error(vix_err)
        else:
            result.data['vix_spot'] = vix_val

        # VIX9D - check if estimated
        vix9d_val, vix9d_err, _ = self._validate_numeric(
            snapshot.get('vix9d'), 'vix9d',
            self.VIX_MIN, self.VIX_MAX
        )
        if vix9d_err:
            result.add_error(vix9d_err)
        else:
            result.data['vix9d'] = vix9d_val
            # Check if this looks like an estimate (VIX * 0.85 pattern)
            if vix_val and vix9d_val and abs(vix9d_val - vix_val * 0.85) < 0.01:
                result.mark_estimated('vix9d')
                result.add_warning("vix9d appears to be estimated (VIX * 0.85)")

        # VVIX
        vvix_val, vvix_err, _ = self._validate_numeric(
            snapshot.get('vvix'), 'vvix',
            self.VVIX_MIN, self.VVIX_MAX
        )
        if vvix_err:
            result.add_error(vvix_err)
        else:
            result.data['vvix'] = vvix_val

        # VVIX Signal
        vvix_signal, signal_err = self._validate_string(
            snapshot.get('vvix_signal'), 'vvix_signal',
            allowed_values=['BUY', 'NEUTRAL', 'HIGH_VOL', None]
        )
        if signal_err:
            result.add_warning(signal_err)  # Non-critical
        result.data['vvix_signal'] = vvix_signal

        # Skew
        skew_val, skew_err, _ = self._validate_numeric(
            snapshot.get('skew'), 'skew',
            self.SKEW_MIN, self.SKEW_MAX
        )
        if skew_err:
            result.add_error(skew_err)
        else:
            result.data['skew'] = skew_val

        # VRP
        vrp_val, vrp_err, _ = self._validate_numeric(
            snapshot.get('vrp'), 'vrp',
            self.VRP_MIN, self.VRP_MAX
        )
        if vrp_err:
            result.add_error(vrp_err)
        else:
            result.data['vrp'] = vrp_val

        # VIX Contango (can be negative in backwardation)
        contango_val, contango_err, _ = self._validate_numeric(
            snapshot.get('vix_contango'), 'vix_contango',
            -50.0, 50.0
        )
        if contango_err:
            result.add_error(contango_err)
        else:
            result.data['vix_contango'] = contango_val

        # Put/Call ratio
        pc_val, pc_err, _ = self._validate_numeric(
            snapshot.get('put_call_ratio'), 'put_call_ratio',
            self.PC_RATIO_MIN, self.PC_RATIO_MAX
        )
        if pc_err:
            result.add_error(pc_err)
        else:
            result.data['put_call_ratio'] = pc_val

        # Fear & Greed
        fg_val, fg_err, _ = self._validate_numeric(
            snapshot.get('fear_greed_score'), 'fear_greed_score',
            self.FEAR_GREED_MIN, self.FEAR_GREED_MAX
        )
        if fg_err:
            result.add_error(fg_err)
        else:
            result.data['fear_greed_score'] = fg_val

        # Market breadth
        breadth_val, breadth_err, _ = self._validate_numeric(
            snapshot.get('market_breadth'), 'market_breadth',
            self.BREADTH_MIN, self.BREADTH_MAX
        )
        if breadth_err:
            result.add_error(breadth_err)
        else:
            result.data['market_breadth'] = breadth_val

        # LEFT signal
        left_signal, left_err = self._validate_string(
            snapshot.get('left_signal'), 'left_signal',
            allowed_values=['BUY', 'SELL', 'NEUTRAL', 'INSUFFICIENT_DATA', None]
        )
        if left_err:
            result.add_warning(left_err)
        result.data['left_signal'] = left_signal

        return result

    def validate_vrp_data(self, vrp: Dict) -> ValidationResult:
        """
        Validate data for vrp_data table.
        """
        result = ValidationResult(is_valid=True, data={})

        # Date
        date_val, date_err = self._validate_date(vrp.get('date'), 'date')
        if date_err:
            result.add_error(date_err)
        else:
            result.data['date'] = date_val

        # VIX (required)
        vix_val, vix_err, _ = self._validate_numeric(
            vrp.get('vix'), 'vix',
            self.VIX_MIN, self.VIX_MAX, required=True
        )
        if vix_err:
            result.add_error(vix_err)
        else:
            result.data['vix'] = vix_val

        # Realized vol (required)
        rv_val, rv_err, _ = self._validate_numeric(
            vrp.get('realized_vol'), 'realized_vol',
            0.0, 100.0, required=True
        )
        if rv_err:
            result.add_error(rv_err)
        else:
            result.data['realized_vol'] = rv_val

        # VRP
        vrp_val, vrp_err, _ = self._validate_numeric(
            vrp.get('vrp'), 'vrp',
            self.VRP_MIN, self.VRP_MAX
        )
        if vrp_err:
            result.add_error(vrp_err)
        else:
            result.data['vrp'] = vrp_val

        # Regime
        regime_value = vrp.get('regime')
        if isinstance(regime_value, str):
            regime_value = regime_value.strip().upper()
        regime, regime_err = self._validate_string(
            regime_value, 'regime',
            allowed_values=['LOW', 'NORMAL', 'ELEVATED', 'HIGH', 'EXTREME', None]
        )
        if regime_err:
            result.add_warning(regime_err)
        result.data['regime'] = regime

        # Expected 6m return
        ret_val, ret_err, _ = self._validate_numeric(
            vrp.get('expected_6m_return'), 'expected_6m_return',
            -50.0, 100.0
        )
        if ret_err:
            result.add_warning(ret_err)  # Not critical
        result.data['expected_6m_return'] = ret_val

        return result

    def validate_fed_balance_sheet(self, data: Dict) -> ValidationResult:
        """
        Validate data for fed_balance_sheet table.
        """
        result = ValidationResult(is_valid=True, data={})

        # Date
        date_val, date_err = self._validate_date(data.get('date'), 'date')
        if date_err:
            result.add_error(date_err)
        else:
            result.data['date'] = date_val

        # Total assets (in billions, typically $7-9T)
        assets_val, assets_err, _ = self._validate_numeric(
            data.get('total_assets'), 'total_assets',
            1000.0, 15000.0, required=True
        )
        if assets_err:
            result.add_error(assets_err)
        else:
            result.data['total_assets'] = assets_val

        # Reserve balances
        reserves_val, reserves_err, _ = self._validate_numeric(
            data.get('reserve_balances'), 'reserve_balances',
            0.0, 10000.0
        )
        if reserves_err:
            result.add_warning(reserves_err)
        result.data['reserve_balances'] = reserves_val or 0.0

        # Loans
        loans_val, loans_err, _ = self._validate_numeric(
            data.get('loans'), 'loans',
            0.0, 5000.0
        )
        if loans_err:
            result.add_warning(loans_err)
        result.data['loans'] = loans_val or 0.0

        # QT cumulative
        qt_val, qt_err, _ = self._validate_numeric(
            data.get('qt_cumulative'), 'qt_cumulative',
            -5000.0, 5000.0
        )
        if qt_err:
            result.add_warning(qt_err)
        result.data['qt_cumulative'] = qt_val or 0.0

        # QT monthly pace
        pace_val, pace_err, _ = self._validate_numeric(
            data.get('qt_monthly_pace'), 'qt_monthly_pace',
            -200.0, 200.0
        )
        if pace_err:
            result.add_warning(pace_err)
        result.data['qt_monthly_pace'] = pace_val or 0.0

        return result

    def validate_move_data(self, data: Dict) -> ValidationResult:
        """
        Validate data for move_index table.
        """
        result = ValidationResult(is_valid=True, data={})

        # Date
        date_val, date_err = self._validate_date(data.get('date'), 'date')
        if date_err:
            result.add_error(date_err)
        else:
            result.data['date'] = date_val

        # MOVE value (required)
        move_val, move_err, _ = self._validate_numeric(
            data.get('move'), 'move',
            self.MOVE_MIN, self.MOVE_MAX, required=True
        )
        if move_err:
            result.add_error(move_err)
        else:
            result.data['move'] = move_val

        # Percentile
        pct_val, pct_err, _ = self._validate_numeric(
            data.get('percentile'), 'percentile',
            0.0, 100.0
        )
        if pct_err:
            result.add_warning(pct_err)
        result.data['percentile'] = pct_val or 50.0

        # Stress level
        stress, stress_err = self._validate_string(
            data.get('stress_level'), 'stress_level',
            allowed_values=['LOW', 'NORMAL', 'ELEVATED', 'HIGH', 'EXTREME', None]
        )
        if stress_err:
            result.add_warning(stress_err)
        result.data['stress_level'] = stress

        return result

    def validate_repo_data(self, data: Dict) -> ValidationResult:
        """
        Validate data for repo_market table.
        """
        result = ValidationResult(is_valid=True, data={})

        # Date
        date_val, date_err = self._validate_date(data.get('date'), 'date')
        if date_err:
            result.add_error(date_err)
        else:
            result.data['date'] = date_val

        # SOFR (required)
        sofr_val, sofr_err, _ = self._validate_numeric(
            data.get('sofr'), 'sofr',
            0.0, 15.0, required=True
        )
        if sofr_err:
            result.add_error(sofr_err)
        else:
            result.data['sofr'] = sofr_val

        # GC Repo
        gc_val, gc_err, _ = self._validate_numeric(
            data.get('gc_repo'), 'gc_repo',
            0.0, 15.0
        )
        if gc_err:
            result.add_warning(gc_err)
        result.data['gc_repo'] = gc_val or 0.0

        # RRP ON
        rrp_val, rrp_err, _ = self._validate_numeric(
            data.get('rrp_on'), 'rrp_on',
            0.0, 3000.0  # Can be up to $2-3T
        )
        if rrp_err:
            result.add_warning(rrp_err)
        result.data['rrp_on'] = rrp_val or 0.0

        # Triparty repo
        tri_val, tri_err, _ = self._validate_numeric(
            data.get('triparty_repo'), 'triparty_repo',
            0.0, 1000.0
        )
        if tri_err:
            result.add_warning(tri_err)
        result.data['triparty_repo'] = tri_val or 0.0

        # SOFR z-score
        z_val, z_err, _ = self._validate_numeric(
            data.get('sofr_z_score'), 'sofr_z_score',
            -10.0, 10.0
        )
        if z_err:
            result.add_warning(z_err)
        result.data['sofr_z_score'] = z_val or 0.0

        return result

    def validate_breadth_data(self, data: Dict) -> ValidationResult:
        """
        Validate data for breadth_history table.
        """
        result = ValidationResult(is_valid=True, data={})

        # Date
        date_val, date_err = self._validate_date(data.get('date'), 'date')
        if date_err:
            result.add_error(date_err)
        else:
            result.data['date'] = date_val

        # Advancing (required)
        adv_val, adv_err, _ = self._validate_numeric(
            data.get('advancing'), 'advancing',
            0, 600, required=True  # Max ~500 S&P stocks
        )
        if adv_err:
            result.add_error(adv_err)
        else:
            result.data['advancing'] = int(adv_val) if adv_val else 0

        # Declining (required)
        dec_val, dec_err, _ = self._validate_numeric(
            data.get('declining'), 'declining',
            0, 600, required=True
        )
        if dec_err:
            result.add_error(dec_err)
        else:
            result.data['declining'] = int(dec_val) if dec_val else 0

        # Unchanged
        unch_val, unch_err, _ = self._validate_numeric(
            data.get('unchanged'), 'unchanged',
            0, 600
        )
        if unch_err:
            result.add_warning(unch_err)
        result.data['unchanged'] = int(unch_val) if unch_val else 0

        # Total
        total_val, total_err, _ = self._validate_numeric(
            data.get('total'), 'total',
            0, 600
        )
        if total_err:
            result.add_warning(total_err)
        result.data['total'] = int(total_val) if total_val else 0

        # Breadth percentage
        breadth_val, breadth_err, _ = self._validate_numeric(
            data.get('breadth_pct'), 'breadth_pct',
            self.BREADTH_MIN, self.BREADTH_MAX
        )
        if breadth_err:
            result.add_error(breadth_err)
        else:
            result.data['breadth_pct'] = breadth_val

        # A/D Line
        ad_line_val, ad_line_err, _ = self._validate_numeric(
            data.get('ad_line'), 'ad_line',
            -100000, 100000
        )
        if ad_line_err:
            result.add_warning(ad_line_err)
        result.data['ad_line'] = ad_line_val or 0.0

        # A/D Diff
        ad_diff_val, ad_diff_err, _ = self._validate_numeric(
            data.get('ad_diff'), 'ad_diff',
            -600, 600
        )
        if ad_diff_err:
            result.add_warning(ad_diff_err)
        result.data['ad_diff'] = int(ad_diff_val) if ad_diff_val else 0

        # McClellan
        mc_val, mc_err, _ = self._validate_numeric(
            data.get('mcclellan'), 'mcclellan',
            -500, 500
        )
        if mc_err:
            result.add_warning(mc_err)
        result.data['mcclellan'] = mc_val or 0.0

        return result

    def validate_liquidity_data(self, data: Dict) -> ValidationResult:
        """
        Validate data for liquidity_history table.
        """
        result = ValidationResult(is_valid=True, data={})

        # Date
        date_val, date_err = self._validate_date(data.get('date'), 'date')
        if date_err:
            result.add_error(date_err)
        else:
            result.data['date'] = date_val

        # RRP ON (Reverse Repo)
        rrp_val, rrp_err, _ = self._validate_numeric(
            data.get('rrp_on'), 'rrp_on',
            0.0, 3000.0
        )
        if rrp_err:
            result.add_warning(rrp_err)
        result.data['rrp_on'] = rrp_val

        # TGA (Treasury General Account)
        tga_val, tga_err, _ = self._validate_numeric(
            data.get('tga'), 'tga',
            0.0, 2000.0
        )
        if tga_err:
            result.add_warning(tga_err)
        result.data['tga'] = tga_val

        # SOFR
        sofr_val, sofr_err, _ = self._validate_numeric(
            data.get('sofr'), 'sofr',
            0.0, 15.0
        )
        if sofr_err:
            result.add_warning(sofr_err)
        result.data['sofr'] = sofr_val

        # Net Liquidity
        net_liq_val, net_liq_err, _ = self._validate_numeric(
            data.get('net_liquidity'), 'net_liquidity',
            self.LIQUIDITY_MIN, self.LIQUIDITY_MAX
        )
        if net_liq_err:
            result.add_warning(net_liq_err)
        result.data['net_liquidity'] = net_liq_val

        return result

    # ========================================
    # DATAFRAME VALIDATORS
    # ========================================

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        table_type: str
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Validate an entire DataFrame row by row.

        Args:
            df: DataFrame to validate
            table_type: One of 'daily_snapshot', 'vrp', 'fed_bs', 'move', 'repo', 'breadth', 'liquidity'

        Returns:
            Tuple of (valid_df, errors, warnings)
        """
        validator_map = {
            'daily_snapshot': self.validate_daily_snapshot,
            'vrp': self.validate_vrp_data,
            'fed_bs': self.validate_fed_balance_sheet,
            'move': self.validate_move_data,
            'repo': self.validate_repo_data,
            'breadth': self.validate_breadth_data,
            'liquidity': self.validate_liquidity_data,
        }

        if table_type not in validator_map:
            return df, [f"Unknown table type: {table_type}"], []

        validator = validator_map[table_type]
        valid_rows = []
        all_errors = []
        all_warnings = []

        for idx, row in df.iterrows():
            result = validator(row.to_dict())

            if result.is_valid:
                valid_rows.append(result.data)
            else:
                all_errors.extend([f"Row {idx}: {e}" for e in result.errors])

            all_warnings.extend([f"Row {idx}: {w}" for w in result.warnings])

        valid_df = pd.DataFrame(valid_rows) if valid_rows else pd.DataFrame()

        return valid_df, all_errors, all_warnings


# ========================================
# CONVENIENCE FUNCTIONS
# ========================================

def validate_before_save(data: Dict, table_type: str) -> ValidationResult:
    """
    Convenience function to validate data before database save.

    Args:
        data: Dictionary of data to validate
        table_type: Type of table ('daily_snapshot', 'vrp', etc.)

    Returns:
        ValidationResult with is_valid, data, errors, warnings
    """
    validator = DataValidator()

    validator_map = {
        'daily_snapshot': validator.validate_daily_snapshot,
        'vrp': validator.validate_vrp_data,
        'fed_bs': validator.validate_fed_balance_sheet,
        'move': validator.validate_move_data,
        'repo': validator.validate_repo_data,
        'breadth': validator.validate_breadth_data,
        'liquidity': validator.validate_liquidity_data,
    }

    if table_type not in validator_map:
        result = ValidationResult(is_valid=False)
        result.add_error(f"Unknown table type: {table_type}")
        return result

    return validator_map[table_type](data)
