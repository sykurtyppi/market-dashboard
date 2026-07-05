"""Payload builders for Phase 1 pages (Volatility & VRP, Market Breadth).

Both read cached SQLite history — no live collector calls per request.
"""
from typing import Any, Dict, List

from api.deps import get_db
from api.overview_service import _num, _series


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


def _vrp_state(vrp: float | None) -> str:
    if vrp is None:
        return "neutral"
    return "good" if vrp > 0 else "warn"


def build_volatility() -> Dict[str, Any]:
    db = get_db()
    latest = db.get_latest_vrp() or {}
    hist = db.get_vrp_history(days=180)

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
        "charts": {
            "vrp_history": _series(hist, value_col="vrp"),
            "vix": _series(hist, value_col="vix"),
            "realized_vol": _series(hist, value_col="realized_vol"),
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
        return {"as_of": None, "metrics": [], "charts": {"ad_line": [], "mcclellan": [], "breadth_pct": []}}

    latest = hist.iloc[-1]
    # breadth_pct may be stored 0–1 or 0–100; normalize to a percentage
    raw_pct = _num(latest.get("breadth_pct"))
    pct = (raw_pct * 100) if (raw_pct is not None and raw_pct <= 1.0) else raw_pct
    advancing = _num(latest.get("advancing"))
    declining = _num(latest.get("declining"))
    mcclellan = _num(latest.get("mcclellan"))
    ad_line = _num(latest.get("ad_line"))

    metrics = [
        {"key": "breadth_pct", "label": "Breadth %", "value": pct, "unit": "%",
         "state": _breadth_pct_state(pct), "source": "S&P 500 advancers"},
        {"key": "advancing", "label": "Advancing", "value": advancing, "unit": "",
         "state": "neutral", "source": "S&P 500"},
        {"key": "declining", "label": "Declining", "value": declining, "unit": "",
         "state": "neutral", "source": "S&P 500"},
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


_MOVE_NOTES = {
    "LOW": "Calm Treasury market — below the 25th percentile.",
    "NORMAL": "Typical Treasury volatility (middle of the historical range).",
    "ELEVATED": "Elevated Treasury uncertainty — 75th–90th percentile.",
    "HIGH": "Treasury market stress — above the 90th percentile.",
}


def build_treasury_stress() -> dict:
    db = get_db()
    hist = _asc(db.get_move_history(days=365))  # method returns DESC
    if hist is None or hist.empty:
        return {"as_of": None, "regime": None, "regime_note": "No MOVE data available",
                "metrics": [], "charts": {"move_history": [], "percentile_history": []}}

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
        "regime_note": _MOVE_NOTES.get((stress or "").upper(), "Treasury volatility regime"),
        "metrics": metrics,
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
                "metrics": [], "charts": {"sofr_history": [], "rrp_history": []}}

    latest = hist.iloc[-1]
    sofr = _num(latest.get("sofr"))
    rrp = _num(latest.get("rrp_on"))
    zscore = _num(latest.get("sofr_z_score"))
    stress = latest.get("stress_level")

    metrics = [
        {"key": "sofr", "label": "SOFR", "value": sofr, "unit": "%",
         "state": "neutral", "source": "FRED (SOFR)"},
        {"key": "rrp", "label": "RRP (Overnight)", "value": rrp, "unit": "",
         "state": "neutral", "source": "FRED (RRPONTSYD)"},
        {"key": "sofr_z", "label": "SOFR Z-Score", "value": zscore, "unit": "",
         "state": "neutral", "source": "Derived (252d)"},
    ]
    return {
        "as_of": str(latest.get("date")),
        "regime": stress,
        "regime_note": (
            "Ample reserves — no funding stress." if _stress_state(stress) == "good"
            else "Funding conditions tightening." if _stress_state(stress) == "warn"
            else "Funding stress." if _stress_state(stress) == "crit"
            else "Repo market status"
        ),
        "metrics": metrics,
        "charts": {
            "sofr_history": _series(hist, value_col="sofr"),
            "rrp_history": _series(hist, value_col="rrp_on"),
        },
    }
