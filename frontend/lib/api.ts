// Typed client for the Market Risk Dashboard API (server-side).
//
// These run in Server Components / Route Handlers, so the base URL is a
// server-only env var (MARKET_API_URL) — not NEXT_PUBLIC_*, which would be
// inlined into the browser bundle and frozen at build time. Client components
// call the same-origin Next route handlers (/api/refresh, /api/freshness)
// which proxy to this base, so the backend URL never reaches the browser.

export const API_BASE = process.env.MARKET_API_URL ?? "http://localhost:8000";

export type State = "good" | "warn" | "crit" | "neutral";

export interface Freshness {
  status: string;
  age: string;
  is_fresh: boolean;
  as_of?: string | null;
  age_hours?: number | null;
}

export interface Metric {
  key: string;
  label: string;
  value: number | null;
  unit: string;
  state: State;
  source: string;
}

export interface RegimeComponent {
  key: string;
  label: string;
  state: State;
  value: string;
  note: string;
}

export interface Point {
  date: string;
  value: number;
}

export interface DetailRow {
  indicator: string;
  value: number | null;
  unit?: string;
  state: State;
  source: string;
}

export interface Overview {
  as_of: string | null;
  freshness: Freshness;
  left_signal: string | null;
  regime: { composite_risk: number | null; components: RegimeComponent[] };
  metrics: Metric[];
  charts: {
    vrp_history: Point[];
    credit_spreads: { hy: Point[]; ig: Point[] };
  };
  detail: DetailRow[];
}

export interface Volatility {
  as_of: string | null;
  regime: string | null;
  regime_note: string;
  metrics: Metric[];
  charts: { vrp_history: Point[]; vix: Point[]; realized_vol: Point[] };
}

export interface Breadth {
  as_of: string | null;
  metrics: Metric[];
  charts: { ad_line: Point[]; mcclellan: Point[]; breadth_pct: Point[] };
}

export interface SectorRow {
  ticker: string;
  name: string | null;
  category: string | null;
  change_pct: number | null;
  price: number | null;
  state: State;
}

export interface VixTenor {
  maturity: string;
  days: number;
  value: number;
}

export interface Sectors {
  as_of: string | null;
  sectors: SectorRow[];
  rotation: {
    signal: string | null;
    state: State;
    interpretation: string | null;
    leading_sectors: string[];
  };
  vix_term: VixTenor[];
  vix_structure: string | null;
  warnings: string[];
}

export interface CreditLiquidity {
  as_of: string | null;
  metrics: Metric[];
  charts: {
    credit_spreads: { hy: Point[]; ig: Point[] };
    fed_assets: Point[];
    qt_cumulative: Point[];
  };
  notes: Record<string, string>;
}

export interface TreasuryStress {
  as_of: string | null;
  regime: string | null;
  state: State;
  regime_note: string;
  metrics: Metric[];
  charts: { move_history: Point[]; percentile_history: Point[] };
}

export interface Repo {
  as_of: string | null;
  regime: string | null;
  state: State;
  regime_note: string;
  metrics: Metric[];
  charts: { sofr_history: Point[]; rrp_history: Point[] };
}

export interface FedWatch {
  as_of: string | null;
  current_rate: string | null;
  degraded: boolean;
  next_meeting: { date: string | null; days_until: number | null };
  most_likely: { outcome: string | null; pct: number | null };
  market_bias: string | null;
  bias_state: State;
  probabilities: { outcome: string; pct: number | null }[];
  metrics: Metric[];
  warnings: string[];
}

export interface AssetPerf {
  ticker: string;
  name: string | null;
  change_pct: number | null;
  state: State;
}

export interface Correlation {
  pair: string | null;
  correlation: number | null;
  strength: string | null;
  interpretation: string | null;
}

export interface CrossAsset {
  as_of: string | null;
  regime: { signal: string | null; state: State; description: string | null; confidence: number | null };
  assets: AssetPerf[];
  correlations: Correlation[];
  warnings: string[];
}

export interface COTPosition {
  symbol: string;
  name: string | null;
  category: string | null;
  date: string | null;
  spec_net: number | null;
  spec_net_change: number | null;
  comm_net: number | null;
  open_interest: number | null;
}

export interface COT {
  as_of: string | null;
  positions: COTPosition[];
  warnings: string[];
}

export interface OptionsETF {
  ticker: string;
  price: number | null;
  expiry: string | null;
  dte: number | null;
  put_call_ratio: number | null;
  call_volume: number | null;
  put_volume: number | null;
  sentiment: string | null;
  state: State;
}

export interface OptionsFlow {
  as_of: string | null;
  etfs: OptionsETF[];
  warnings: string[];
}

export interface Institutional {
  as_of: string | null;
  dark_pool: {
    avg_pct: number | null; etf_pct: number | null; stock_pct: number | null;
    sentiment: string | null; state: State; interpretation: string | null; week_ending: string | null;
  } | null;
  insider: {
    total_transactions: number | null; buy_count: number | null; sell_count: number | null;
    buy_sell_ratio: number | null; sentiment: string | null; state: State; period_days: number | null;
  } | null;
  auctions: {
    avg_bid_to_cover: number | null; avg_indirect_pct: number | null; avg_direct_pct: number | null;
    auction_count: number | null; weak_auctions: number | null; strong_auctions: number | null;
    health: string | null; state: State;
  } | null;
  warnings: string[];
}

export interface EconomicEvent {
  name: string | null;
  date: string | null;
  days_until: number | null;
  importance: string | null;
  category: string | null;
  actual: number | null;
  forecast: number | null;
  previous: number | null;
  yoy_change: number | null;
  unit: string | null;
}

export interface EconomicCalendar {
  as_of: string | null;
  events: EconomicEvent[];
  warnings: string[];
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`API ${res.status}: ${path}`);
  return res.json() as Promise<T>;
}

export const getOverview = () => getJson<Overview>("/api/overview");
export const getVolatility = () => getJson<Volatility>("/api/volatility");
export const getBreadth = () => getJson<Breadth>("/api/breadth");
export const getSectors = () => getJson<Sectors>("/api/sectors");
export const getCreditLiquidity = () => getJson<CreditLiquidity>("/api/credit-liquidity");
export const getTreasuryStress = () => getJson<TreasuryStress>("/api/treasury-stress");
export const getRepo = () => getJson<Repo>("/api/repo");
export const getFedWatch = () => getJson<FedWatch>("/api/fed-watch");
export const getCrossAsset = () => getJson<CrossAsset>("/api/cross-asset");
export const getCot = () => getJson<COT>("/api/cot");
export const getOptionsFlow = () => getJson<OptionsFlow>("/api/options-flow");
export const getInstitutional = () => getJson<Institutional>("/api/institutional");
export const getEconomicCalendar = () => getJson<EconomicCalendar>("/api/economic-calendar");
export const getFreshness = () => getJson<Freshness>("/api/freshness");
