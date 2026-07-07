"""Pydantic response models for the API.

These give the endpoints typed contracts: request/response validation, a
populated OpenAPI schema at /openapi.json, and a basis for client generation.
"""
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

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
    warnings: List[str] = Field(default_factory=list)
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


class VolatilityStats(BaseModel):
    avg_vrp: Optional[float] = None
    std_dev: Optional[float] = None
    current_percentile: Optional[float] = None
    max_vrp: Optional[float] = None
    min_vrp: Optional[float] = None
    observations: int = 0


class VolatilityResponse(BaseModel):
    as_of: Optional[str] = None
    regime: Optional[str] = None
    regime_note: str
    metrics: List[Metric]
    stats: VolatilityStats
    warnings: List[str] = Field(default_factory=list)
    charts: VolatilityCharts


class BreadthCharts(BaseModel):
    ad_line: List[Point]
    mcclellan: List[Point]
    breadth_pct: List[Point]


class BreadthResponse(BaseModel):
    as_of: Optional[str] = None
    metrics: List[Metric]
    warnings: List[str] = Field(default_factory=list)
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
    warnings: List[str] = Field(default_factory=list)


class CreditCharts(BaseModel):
    credit_spreads: CreditSpreads
    fed_assets: List[Point]
    qt_cumulative: List[Point]


class CreditLiquidityResponse(BaseModel):
    as_of: Optional[str] = None
    metrics: List[Metric]
    warnings: List[str] = Field(default_factory=list)
    charts: CreditCharts
    notes: Dict[str, str]


# --- Phase 3 pages ---

class TreasuryCharts(BaseModel):
    move_history: List[Point]
    percentile_history: List[Point]


class TreasuryResponse(BaseModel):
    as_of: Optional[str] = None
    regime: Optional[str] = None
    state: State = "neutral"
    regime_note: str
    metrics: List[Metric]
    warnings: List[str] = Field(default_factory=list)
    charts: TreasuryCharts


class RepoCharts(BaseModel):
    sofr_history: List[Point]
    rrp_history: List[Point]


class RepoResponse(BaseModel):
    as_of: Optional[str] = None
    regime: Optional[str] = None
    state: State = "neutral"
    regime_note: str
    metrics: List[Metric]
    warnings: List[str] = Field(default_factory=list)
    charts: RepoCharts


# --- Phase 4 pages (macro, live) ---

class RateProbability(BaseModel):
    outcome: str
    pct: Optional[float] = None


class NextMeeting(BaseModel):
    date: Optional[str] = None
    days_until: Optional[int] = None


class MostLikely(BaseModel):
    outcome: Optional[str] = None
    pct: Optional[float] = None


class FedWatchResponse(BaseModel):
    as_of: Optional[str] = None
    current_rate: Optional[str] = None
    degraded: bool = False
    next_meeting: NextMeeting
    most_likely: MostLikely
    market_bias: Optional[str] = None
    bias_state: State = "neutral"
    probabilities: List[RateProbability]
    metrics: List[Metric]
    warnings: List[str] = Field(default_factory=list)


class AssetPerf(BaseModel):
    ticker: str
    name: Optional[str] = None
    change_pct: Optional[float] = None
    state: State


class Correlation(BaseModel):
    pair: Optional[str] = None
    correlation: Optional[float] = None
    strength: Optional[str] = None
    interpretation: Optional[str] = None


class CrossAssetRegime(BaseModel):
    signal: Optional[str] = None
    state: State = "neutral"
    description: Optional[str] = None
    confidence: Optional[float] = None


class CrossAssetResponse(BaseModel):
    as_of: Optional[str] = None
    regime: CrossAssetRegime
    assets: List[AssetPerf]
    correlations: List[Correlation]
    warnings: List[str] = Field(default_factory=list)


# --- Phase 5 pages (positioning & flows, live) ---

class COTPosition(BaseModel):
    symbol: str
    name: Optional[str] = None
    category: Optional[str] = None
    date: Optional[str] = None
    spec_net: Optional[float] = None
    spec_net_change: Optional[float] = None
    comm_net: Optional[float] = None
    open_interest: Optional[float] = None


class COTResponse(BaseModel):
    as_of: Optional[str] = None
    positions: List[COTPosition]
    warnings: List[str] = Field(default_factory=list)


class OptionsETF(BaseModel):
    ticker: str
    price: Optional[float] = None
    expiry: Optional[str] = None
    dte: Optional[int] = None
    put_call_ratio: Optional[float] = None
    call_volume: Optional[float] = None
    put_volume: Optional[float] = None
    sentiment: Optional[str] = None
    state: State


class OptionsFlowResponse(BaseModel):
    as_of: Optional[str] = None
    etfs: List[OptionsETF]
    warnings: List[str] = Field(default_factory=list)


# --- Phase 6 pages ---

class DarkPool(BaseModel):
    avg_pct: Optional[float] = None
    etf_pct: Optional[float] = None
    stock_pct: Optional[float] = None
    sentiment: Optional[str] = None
    state: State = "neutral"
    interpretation: Optional[str] = None
    week_ending: Optional[str] = None


class Insider(BaseModel):
    total_transactions: Optional[float] = None
    buy_count: Optional[float] = None
    sell_count: Optional[float] = None
    buy_sell_ratio: Optional[float] = None
    sentiment: Optional[str] = None
    state: State = "neutral"
    period_days: Optional[float] = None


class Auctions(BaseModel):
    avg_bid_to_cover: Optional[float] = None
    avg_indirect_pct: Optional[float] = None
    avg_direct_pct: Optional[float] = None
    auction_count: Optional[float] = None
    weak_auctions: Optional[float] = None
    strong_auctions: Optional[float] = None
    health: Optional[str] = None
    state: State = "neutral"


class InstitutionalResponse(BaseModel):
    as_of: Optional[str] = None
    dark_pool: Optional[DarkPool] = None
    insider: Optional[Insider] = None
    auctions: Optional[Auctions] = None
    warnings: List[str] = Field(default_factory=list)


class EconomicEvent(BaseModel):
    name: Optional[str] = None
    date: Optional[str] = None
    days_until: Optional[int] = None
    importance: Optional[str] = None
    category: Optional[str] = None
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None
    yoy_change: Optional[float] = None
    unit: Optional[str] = None


class EconomicCalendarResponse(BaseModel):
    as_of: Optional[str] = None
    events: List[EconomicEvent]
    warnings: List[str] = Field(default_factory=list)


# --- Final pages: Sentiment, LEFT Strategy, CTA Flow ---

class FearGreed(BaseModel):
    score: Optional[float] = None
    rating: Optional[str] = None
    state: State = "neutral"


class SentimentCharts(BaseModel):
    fear_greed_history: List[Point] = Field(default_factory=list)
    put_call_history: List[Point] = Field(default_factory=list)


class SentimentResponse(BaseModel):
    as_of: Optional[str] = None
    fear_greed: FearGreed
    put_call_ratio: Optional[float] = None
    put_call_source: Optional[str] = None
    charts: SentimentCharts = Field(default_factory=SentimentCharts)
    warnings: List[str] = Field(default_factory=list)


class LeftCharts(BaseModel):
    spread: List[Point]
    ema: List[Point]


class LeftResponse(BaseModel):
    as_of: Optional[str] = None
    signal: Optional[str] = None
    state: State = "neutral"
    metrics: List[Metric]
    charts: LeftCharts
    warnings: List[str] = Field(default_factory=list)


class CTAPosition(BaseModel):
    symbol: str
    position: Optional[str] = None
    exposure: Optional[float] = None
    state: State


class CTAResponse(BaseModel):
    as_of: Optional[str] = None
    positions: List[CTAPosition]
    long_count: int = 0
    short_count: int = 0
    flat_count: int = 0
    warnings: List[str] = Field(default_factory=list)


# --- Admin pages: System Health, Settings ---

class HealthSource(BaseModel):
    key: str
    name: str
    status: str  # raw backend status: healthy/stale/degraded/down/unknown
    state: State  # mapped to the UI's semantic dot
    last_update: Optional[str] = None
    age_hours: Optional[float] = None
    message: str = ""


class SystemHealthResponse(BaseModel):
    overall_status: str
    overall_state: State = "neutral"
    as_of: Optional[str] = None
    sources: List[HealthSource]
    summary: Dict[str, int] = Field(default_factory=dict)
    total_sources: int = 0


class CredentialStatus(BaseModel):
    name: str
    configured: bool
    source: Optional[str] = None  # "environment" | "streamlit_secrets" | None


class ConfigItem(BaseModel):
    label: str
    value: str


class ConfigGroup(BaseModel):
    title: str
    items: List[ConfigItem]


class SettingsResponse(BaseModel):
    # `protected` tells the UI whether the backend actually enforces a token on
    # this endpoint, so it never implies auth that isn't configured.
    protected: bool = False
    credentials: List[CredentialStatus]
    config: List[ConfigGroup]
    warnings: List[str] = Field(default_factory=list)
