"""Pydantic response models for the API.

These give the endpoints typed contracts: request/response validation, a
populated OpenAPI schema at /openapi.json, and a basis for client generation.
"""
from typing import List, Literal, Optional

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
