"""Pydantic response models for the API.

These give the endpoints typed contracts: request/response validation, a
populated OpenAPI schema at /openapi.json, and a basis for client generation.
"""
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict

State = Literal["good", "warn", "crit", "neutral"]


class Freshness(BaseModel):
    status: str
    age: str
    is_fresh: bool


class FreshnessDetail(Freshness):
    as_of: Optional[str] = None
    age_hours: Optional[float] = None


class Metric(BaseModel):
    key: str
    label: str
    value: Optional[float] = None
    unit: str = ""
    state: State
    source: str


class RegimeComponent(BaseModel):
    key: str
    label: str
    state: State
    value: str
    note: str


class Regime(BaseModel):
    composite_risk: Optional[int] = None
    components: List[RegimeComponent]


class Point(BaseModel):
    date: str
    value: float


class CreditSpreads(BaseModel):
    hy: List[Point]
    ig: List[Point]


class Charts(BaseModel):
    vrp_history: List[Point]
    credit_spreads: CreditSpreads


class DetailRow(BaseModel):
    indicator: str
    value: Optional[float] = None
    unit: Optional[str] = None
    state: State
    source: str


class OverviewResponse(BaseModel):
    as_of: Optional[str] = None
    freshness: Freshness
    left_signal: Optional[str] = None
    regime: Regime
    metrics: List[Metric]
    charts: Charts
    detail: List[DetailRow]


class HealthResponse(BaseModel):
    # The health summary is a nested structure; pin the headline field and pass
    # the rest through rather than mirror the whole tree.
    model_config = ConfigDict(extra="allow")
    overall_status: str


# --- Phase 1 pages ---

class VolatilityCharts(BaseModel):
    vrp_history: List[Point]
    vix: List[Point]
    realized_vol: List[Point]


class VolatilityResponse(BaseModel):
    as_of: Optional[str] = None
    regime: Optional[str] = None
    regime_note: str
    metrics: List[Metric]
    charts: VolatilityCharts


class BreadthCharts(BaseModel):
    ad_line: List[Point]
    mcclellan: List[Point]
    breadth_pct: List[Point]


class BreadthResponse(BaseModel):
    as_of: Optional[str] = None
    metrics: List[Metric]
    charts: BreadthCharts


class RefreshResponse(BaseModel):
    status: Literal["started", "already_running", "unauthorized"]
    detail: Optional[str] = None


class RefreshStatus(BaseModel):
    running: bool


# --- Phase 2 pages ---

class SectorRow(BaseModel):
    ticker: str
    name: Optional[str] = None
    category: Optional[str] = None
    change_pct: Optional[float] = None
    price: Optional[float] = None
    state: State


class Rotation(BaseModel):
    signal: Optional[str] = None
    state: State
    interpretation: Optional[str] = None
    leading_sectors: List[str]


class VixTenor(BaseModel):
    maturity: str
    days: int
    value: float


class SectorsResponse(BaseModel):
    as_of: Optional[str] = None
    sectors: List[SectorRow]
    rotation: Rotation
    vix_term: List[VixTenor]
    vix_structure: Optional[str] = None
    warnings: List[str] = []


class CreditCharts(BaseModel):
    credit_spreads: CreditSpreads
    fed_assets: List[Point]
    qt_cumulative: List[Point]


class CreditLiquidityResponse(BaseModel):
    as_of: Optional[str] = None
    metrics: List[Metric]
    charts: CreditCharts
    notes: Dict[str, str]
