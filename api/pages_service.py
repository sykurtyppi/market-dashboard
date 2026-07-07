"""Payload builders for the cached-SQLite pages (Volatility, Breadth, Credit &
Liquidity, Treasury Stress, Repo Market).

All read cached SQLite history — no live collector calls per request.
"""
from statistics import mean, pstdev
from typing import Any, Dict, List

from api.deps import get_db
from api.overview_service import _num, _series


def _aligned_series(hist, cols: List[str], max_points: int = 180) -> Dict[str, List[Dict[str, Any]]]:
    """Build several [{date, value}] series that share dates index-for-index.

    Charts that compare columns (VIX vs realized vs VRP) zip the series by index
    on the frontend, so they must stay aligned. Dropping NaNs and downsampling
    each column independently (as _series does) can desync them when the columns
    have different NaN patterns — so keep only rows where *all* requested columns
    are present, then downsample once on a shared index.
    """
    empty = {c: [] for c in cols}
    if hist is None or getattr(hist, "empty", True):
        return empty
    date_col = "date" if "date" in hist.columns else hist.columns[0]
    present = [c for c in cols if c in hist.columns]
    if not present:
        return empty
    clean = hist[[date_col, *present]].dropna().reset_index(drop=True)
    if clean.empty:
        return empty
    if len(clean) > max_points:
        step = len(clean) / max_points
        idx = sorted({int(i * step) for i in range(max_points)} | {len(clean) - 1})
        clean = clean.iloc[idx]
    out: Dict[str, List[Dict[str, Any]]] = {c: [] for c in cols}
    for _, row in clean.iterrows():
        d = row[date_col]
        d_str = str(d.date()) if hasattr(d, "date") else str(d)
        for c in present:
            v = _num(row[c])
            if v is not None:
                out[c].append({"date": d_str, "value": v})
    return out


def _asc(df):
    """Normalize a history frame to ascending-by-date order.

    Some DatabaseManager history methods return newest-first (DESC), which makes
    iloc[-1] the OLDEST row and reverses charts. Sorting here keeps the API
    correct without changing db_manager (the live Streamlit app depends on it).
    """
    if df is None or df.empty:
        return df
    date_col = "date" if "date" in df.columns else df.columns[0]
    return df.sort_values(date_col).reset_index(drop=True)


def _vrp_stats(hist, current_vrp: float | None) -> Dict[str, Any]:
    if hist is None or hist.empty or "vrp" not in hist.columns:
        return {
            "avg_vrp": None,
            "std_dev": None,
            "current_percentile": None,
            "max_vrp": None,
            "min_vrp": None,
            "observations": 0,
        }

    values = [_num(v) for v in hist["vrp"].tolist()]
    values = [v for v in values if v is not None]
    if not values:
        return {
            "avg_vrp": None,
            "std_dev": None,
            "current_percentile": None,
            "max_vrp": None,
            "min_vrp": None,
            "observations": 0,
        }

    current = current_vrp if current_vrp is not None else values[-1]
    percentile = (sum(1 for v in values if v < current) / len(values)) * 100
    return {
        "avg_vrp": round(mean(values), 2),
        "std_dev": round(pstdev(values), 2) if len(values) > 1 else 0.0,
        "current_percentile": round(percentile, 1),
        "max_vrp": round(max(values), 2),
        "min_vrp": round(min(values), 2),
        "observations": len(values),
    }


def _vrp_state(vrp: float | None) -> str:
    if vrp is None:
        return "neutral"
    return "good" if vrp > 0 else "warn"


def build_volatility() -> Dict[str, Any]:
    db = get_db()
    latest = db.get_latest_vrp() or {}
    # Fetch a wide window so the frontend range control (1M/3M/6M/All) has full
    # history to filter; downsampling to <=180 points happens in _aligned_series.
    hist = _asc(db.get_vrp_history(days=730))
    aligned = _aligned_series(hist, ["vix", "realized_vol", "vrp"])

    vix = _num(latest.get("vix"))
    rv = _num(latest.get("realized_vol"))
    vrp = _num(latest.get("vrp"))
    regime = latest.get("regime")
    exp_ret = _num(latest.get("expected_6m_return"))

    metrics = [
        {"key": "vix", "label": "Implied Vol (VIX)", "value": vix, "unit": "",
         "state": "neutral", "source": "CBOE / Yahoo (^VIX)"},
        {"key": "realized_vol", "label": "Realized Vol (21d)", "value": rv, "unit": "",
         "state": "neutral", "source": "SPX returns"},
        {"key": "vrp", "label": "VRP", "value": vrp, "unit": "",
         "state": _vrp_state(vrp), "source": "Implied − realized"},
        {"key": "expected_6m_return", "label": "Expected 6M Return", "value": exp_ret, "unit": "%",
         "state": "good" if (exp_ret is not None and exp_ret > 0) else "neutral",
         "source": "VRP regime model"},
    ]

    return {
        "as_of": latest.get("date"),
        "regime": regime,
        "regime_note": (
            "No VRP data available"
            if vrp is None
            else "Implied above realized — premium is rich (sell-vol favorable)"
            if vrp > 0
            else "Realized above implied — recent turbulence"
        ),
        "metrics": metrics,
        "stats": _vrp_stats(hist, vrp),
        "warnings": [],
        # Aligned so the frontend can zip vix/realized/vrp by index safely.
        "charts": {
            k: aligned[src]
            for k, src in (("vrp_history", "vrp"), ("vix", "vix"), ("realized_vol", "realized_vol"))
        },
    }


def _breadth_pct_state(pct: float | None) -> str:
    if pct is None:
        return "neutral"
    if pct >= 55:
        return "good"
    if pct >= 45:
        return "neutral"
    return "warn"


def _mcclellan_state(m: float | None) -> str:
    if m is None:
        return "neutral"
    if m > 0:
        return "good"
    if m > -50:
        return "neutral"
    return "warn"


def build_breadth() -> Dict[str, Any]:
    db = get_db()
    hist = db.get_breadth_history(days=120)

    if hist is None or hist.empty:
        return {"as_of": None, "metrics": [], "warnings": [],
                "charts": {"ad_line": [], "mcclellan": [], "breadth_pct": []}}

    warnings: List[str] = []

    # The stored mcclellan column is dead weight: save_breadth historically
    # defaulted missing values to 0, so the whole series reads as a flat-zero
    # oscillator — which the frontend would faithfully (and misleadingly) plot.
    # ad_diff IS stored per row, so compute the real McClellan (EMA19 − EMA39 of
    # net advances) from history instead of trusting the dead column.
    stored = [_num(v) for v in hist.get("mcclellan", []).tolist()] if "mcclellan" in hist.columns else []
    stored_dead = not any(v for v in stored if v)  # all None/0
    if stored_dead and "ad_diff" in hist.columns and len(hist) >= 39:
        ema19 = hist["ad_diff"].ewm(span=19, adjust=False).mean()
        ema39 = hist["ad_diff"].ewm(span=39, adjust=False).mean()
        hist = hist.assign(mcclellan=(ema19 - ema39).round(2))
    elif stored_dead:
        # Too little history to compute honestly — say so rather than plot zeros.
        hist = hist.assign(mcclellan=None)
        warnings.append("McClellan oscillator unavailable — needs 39+ days of A/D history.")

    latest = hist.iloc[-1]
    # breadth_pct may be stored 0–1 or 0–100; normalize to a percentage
    raw_pct = _num(latest.get("breadth_pct"))
    pct = (raw_pct * 100) if (raw_pct is not None and raw_pct <= 1.0) else raw_pct
    advancing = _num(latest.get("advancing"))
    declining = _num(latest.get("declining"))
    mcclellan = _num(latest.get("mcclellan"))
    ad_line = _num(latest.get("ad_line"))

    # The collector samples ~100 representative S&P 500 constituents (a ~20%
    # proxy) — labeling the raw counts "S&P 500" implied a 500-stock universe.
    metrics = [
        {"key": "breadth_pct", "label": "Breadth %", "value": pct, "unit": "%",
         "state": _breadth_pct_state(pct), "source": "S&P 500 100-stock sample"},
        {"key": "advancing", "label": "Advancing", "value": advancing, "unit": "",
         "state": "neutral", "source": "of 100-stock sample"},
        {"key": "declining", "label": "Declining", "value": declining, "unit": "",
         "state": "neutral", "source": "of 100-stock sample"},
        {"key": "mcclellan", "label": "McClellan Osc.", "value": mcclellan, "unit": "",
         "state": _mcclellan_state(mcclellan), "source": "EMA(19)−EMA(39) of A/D"},
    ]

    # Normalize the breadth_pct series the same way for the chart
    pct_series = _series(hist, value_col="breadth_pct")
    if pct_series and all(p["value"] <= 1.0 for p in pct_series):
        pct_series = [{"date": p["date"], "value": p["value"] * 100} for p in pct_series]

    return {
        "as_of": str(latest.get("date")),
        "metrics": metrics,
        "warnings": warnings,
        "charts": {
            "ad_line": _series(hist, value_col="ad_line"),
            "mcclellan": _series(hist, value_col="mcclellan"),
            "breadth_pct": pct_series,
        },
    }


def _latest_val(df) -> float | None:
    if df is None or df.empty:
        return None
    date_col = "date" if "date" in df.columns else df.columns[0]
    cols = [c for c in df.columns if c != date_col]
    return _num(df[cols[0]].iloc[-1]) if cols else None


def _hy_state(hy: float | None) -> str:
    if hy is None:
        return "neutral"
    if hy < 3.0:
        return "good"
    if hy < 5.0:
        return "warn"
    return "crit"


def build_credit_liquidity() -> dict:
    db = get_db()
    hy_hist = db.get_indicator_history("credit_spread_hy", days=365)
    ig_hist = db.get_indicator_history("credit_spread_ig", days=365)
    fed = _asc(db.get_fed_balance_sheet_history(days=365))  # method returns DESC

    hy = _latest_val(hy_hist)
    ig = _latest_val(ig_hist)

    total_assets = qt_cum = qt_pace = reserves = None
    fed_date = None
    if fed is not None and not fed.empty:
        row = fed.iloc[-1]
        fed_date = str(row.get("date"))
        total_assets = _num(row.get("total_assets"))
        qt_cum = _num(row.get("qt_cumulative"))
        qt_pace = _num(row.get("qt_monthly_pace"))
        reserves = _num(row.get("reserve_balances"))

    # Fed figures are stored in $ millions; present in $T / $B.
    to_t = lambda v: round(v / 1_000_000, 2) if v is not None else None
    to_b = lambda v: round(v / 1_000, 1) if v is not None else None

    metrics = [
        {"key": "hy", "label": "HY Credit Spread", "value": hy, "unit": "%",
         "state": _hy_state(hy), "source": "FRED (BAMLH0A0HYM2)"},
        {"key": "ig", "label": "IG Credit Spread", "value": ig, "unit": "%",
         "state": "neutral", "source": "FRED (BAMLC0A0CM)"},
        {"key": "fed_assets", "label": "Fed Total Assets", "value": to_t(total_assets), "unit": "T",
         "state": "neutral", "source": "FRED (WALCL)"},
        {"key": "reserves", "label": "Reserve Balances", "value": to_t(reserves), "unit": "T",
         "state": "neutral", "source": "FRED (WRESBAL)"},
        {"key": "qt_cumulative", "label": "QT Cumulative", "value": to_b(qt_cum), "unit": "B",
         "state": "neutral", "source": "Derived (Fed balance sheet)"},
        {"key": "qt_pace", "label": "QT Monthly Pace", "value": to_b(qt_pace), "unit": "B",
         "state": "neutral", "source": "Derived (Fed balance sheet)"},
    ]

    fed_assets_series = [
        {"date": p["date"], "value": round(p["value"] / 1_000_000, 3)}
        for p in _series(fed, value_col="total_assets")
    ]
    qt_series = [
        {"date": p["date"], "value": round(p["value"] / 1_000, 1)}
        for p in _series(fed, value_col="qt_cumulative")
    ]

    return {
        "as_of": fed_date,
        "metrics": metrics,
        "warnings": [],
        "charts": {
            "credit_spreads": {
                "hy": _series(hy_hist),
                "ig": _series(ig_hist),
            },
            "fed_assets": fed_assets_series,
            "qt_cumulative": qt_series,
        },
        "notes": {
            "net_liquidity": (
                "Net liquidity (Fed BS − TGA − RRP) needs the liquidity pipeline, "
                "which isn't currently populated."
            ),
        },
    }


# --- Phase 3 pages: Treasury Stress (MOVE) & Repo Market (SOFR) ---

def _stress_state(level: str | None) -> str:
    return {
        "LOW": "good", "NORMAL": "good", "AMPLE": "good", "ABUNDANT": "good",
        "ELEVATED": "warn", "TIGHTENING": "warn",
        "HIGH": "crit", "STRESS": "crit",
    }.get((level or "").upper(), "neutral")


# Describe the fixed MOVE-level thresholds the regime is actually derived from.
# The old wording claimed percentiles ("below the 25th percentile"), which the
# rolling percentile metric shown right next to it can flatly contradict —
# e.g. regime LOW (MOVE 65 < 80) while the 2Y percentile reads 50%.
_MOVE_NOTES = {
    "LOW": "Calm Treasury market — MOVE below 80.",
    "NORMAL": "Typical Treasury volatility — MOVE 80–120.",
    "ELEVATED": "Elevated Treasury uncertainty — MOVE 120–150.",
    "HIGH": "Treasury market stress — MOVE above 150.",
}


def build_treasury_stress() -> dict:
    db = get_db()
    # 730d so the "(2Y)" percentile window is honest, not a 1Y window in disguise.
    hist = _asc(db.get_move_history(days=730))  # method returns DESC
    if hist is None or hist.empty:
        return {"as_of": None, "regime": None, "regime_note": "No MOVE data available",
                "metrics": [], "warnings": [], "charts": {"move_history": [], "percentile_history": []}}

    # The stored percentile column is dead weight: the collector's
    # division-by-zero fallback wrote a constant 50.0 into every row, which
    # charted as a flat line and pinned the headline percentile to 50%. The
    # MOVE levels ARE stored, so compute each row's percentile rank within the
    # fetched window instead of trusting the degenerate column.
    stored_pct = hist["percentile"].dropna() if "percentile" in hist.columns else None
    if (stored_pct is None or stored_pct.nunique() <= 1) and "move" in hist.columns and len(hist) >= 20:
        hist = hist.assign(percentile=(hist["move"].rank(pct=True) * 100).round(1))

    latest = hist.iloc[-1]
    move = _num(latest.get("move"))
    pct = _num(latest.get("percentile"))
    stress = latest.get("stress_level")

    metrics = [
        {"key": "move", "label": "MOVE Index", "value": move, "unit": "",
         "state": _stress_state(stress), "source": "ICE BofA / Yahoo (^MOVE)"},
        {"key": "percentile", "label": "Percentile (2Y)", "value": pct, "unit": "%",
         "state": "neutral", "source": "Historical distribution"},
    ]
    return {
        "as_of": str(latest.get("date")),
        "regime": stress,
        "state": _stress_state(stress),
        "regime_note": _MOVE_NOTES.get((stress or "").upper(), "Treasury volatility regime"),
        "metrics": metrics,
        "warnings": [],
        "charts": {
            "move_history": _series(hist, value_col="move"),
            "percentile_history": _series(hist, value_col="percentile"),
        },
    }


def build_repo() -> dict:
    db = get_db()
    hist = _asc(db.get_repo_history(days=365))  # method returns DESC
    if hist is None or hist.empty:
        return {"as_of": None, "regime": None, "regime_note": "No repo data available",
                "metrics": [], "warnings": [], "charts": {"sofr_history": [], "rrp_history": []}}

    latest = hist.iloc[-1]
    sofr = _num(latest.get("sofr"))
    rrp = _num(latest.get("rrp_on"))
    zscore = _num(latest.get("sofr_z_score"))
    stress = latest.get("stress_level")

    metrics = [
        {"key": "sofr", "label": "SOFR", "value": sofr, "unit": "%",
         "state": "neutral", "source": "FRED (SOFR)"},
        # The collector stores rrp_on in $ trillions (FRED's $B value / 1000,
        # despite its docstring) — shown raw it rendered "0.00" with no unit,
        # a unitless mystery number. Present in $B.
        {"key": "rrp", "label": "RRP (Overnight)",
         "value": round(rrp * 1000, 2) if rrp is not None else None, "unit": "B",
         "state": "neutral", "source": "FRED (RRPONTSYD), $B"},
        {"key": "sofr_z", "label": "SOFR Z-Score", "value": zscore, "unit": "",
         "state": "neutral", "source": "Derived (252d)"},
    ]
    state = _stress_state(stress)
    return {
        "as_of": str(latest.get("date")),
        "regime": stress,
        "state": state,
        "regime_note": (
            "Ample reserves — no funding stress." if state == "good"
            else "Funding conditions tightening." if state == "warn"
            else "Funding stress." if state == "crit"
            else "Repo market status"
        ),
        "metrics": metrics,
        "warnings": [],
        "charts": {
            "sofr_history": _series(hist, value_col="sofr"),
            # Same $T→$B conversion as the metric so chart and card agree.
            "rrp_history": [
                {"date": p["date"], "value": round(p["value"] * 1000, 2)}
                for p in _series(hist, value_col="rrp_on")
            ],
        },
    }
