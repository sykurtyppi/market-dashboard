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
  warnings: string[];
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
  warnings: string[];
  regime: string | null;
  regime_note: string;
  metrics: Metric[];
  stats: {
    avg_vrp: number | null;
    std_dev: number | null;
    current_percentile: number | null;
    max_vrp: number | null;
    min_vrp: number | null;
    observations: number;
  };
  charts: { vrp_history: Point[]; vix: Point[]; realized_vol: Point[] };
}

export interface Breadth {
  as_of: string | null;
  warnings: string[];
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
  warnings: string[];
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
  warnings: string[];
  regime: string | null;
  state: State;
  regime_note: string;
  metrics: Metric[];
  charts: { move_history: Point[]; percentile_history: Point[] };
}

export interface Repo {
  as_of: string | null;
  warnings: string[];
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
  change_1d_pct: number | null;
  change_1m_pct: number | null;
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

export interface Sentiment {
  as_of: string | null;
  fear_greed: { score: number | null; rating: string | null; state: State };
  put_call_ratio: number | null;
  put_call_source: string | null;
  charts: { fear_greed_history: Point[]; put_call_history: Point[] };
  warnings: string[];
}

export interface Left {
  as_of: string | null;
  signal: string | null;
  state: State;
  metrics: Metric[];
  charts: { spread: Point[]; ema: Point[] };
  warnings: string[];
}

export interface CTAPosition {
  symbol: string;
  position: string | null;
  exposure: number | null;
  state: State;
}

export interface CTA {
  as_of: string | null;
  positions: CTAPosition[];
  long_count: number;
  short_count: number;
  flat_count: number;
  warnings: string[];
}

export interface HealthSource {
  key: string;
  name: string;
  status: string;
  state: State;
  last_update: string | null;
  age_hours: number | null;
  message: string;
}

export interface SystemHealth {
  overall_status: string;
  overall_state: State;
  as_of: string | null;
  sources: HealthSource[];
  summary: Record<string, number>;
  total_sources: number;
}

export interface CredentialStatus {
  name: string;
  configured: boolean;
  source: string | null;
}

export interface ConfigItem {
  label: string;
  value: string;
}

export interface ConfigGroup {
  title: string;
  items: ConfigItem[];
}

export interface Settings {
  protected: boolean;
  credentials: CredentialStatus[];
  config: ConfigGroup[];
  warnings: string[];
}

// Every backend request times out rather than hanging a page render
// indefinitely — a wedged backend (accepts the connection, never responds)
// must surface as a clear error, not a blank page until some infra-level
// timeout fires.
const API_TIMEOUT_MS = 10_000;

export function backendFetch(path: string, init: RequestInit = {}): Promise<Response> {
  return fetch(`${API_BASE}${path}`, { cache: "no-store", ...init, signal: AbortSignal.timeout(API_TIMEOUT_MS) });
}

// Translate fetch's raw failures into actionable messages. Verified shapes:
// AbortSignal.timeout rejects with DOMException name "TimeoutError";
// connection-refused rejects with TypeError "fetch failed".
export function describeApiError(error: unknown, path: string): Error {
  if (error instanceof DOMException && error.name === "TimeoutError") {
    return new Error(`API timed out after ${API_TIMEOUT_MS / 1000}s: ${path} — the backend is not responding`);
  }
  const message = error instanceof Error ? error.message : "Unknown error";
  return new Error(`Cannot reach API (${message}): ${path}`);
}

async function getJson<T>(path: string): Promise<T> {
  let res: Response;
  try {
    res = await backendFetch(path);
  } catch (error: unknown) {
    throw describeApiError(error, path);
  }
  if (!res.ok) throw new Error(`API ${res.status}: ${path}`);
  try {
    return (await res.json()) as T;
  } catch {
    // 200 with a non-JSON body (proxy error page, truncated stream) is an
    // infra problem, not an app bug — say so instead of leaking a SyntaxError.
    throw new Error(`API returned invalid JSON (status ${res.status}): ${path}`);
  }
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
export const getSentiment = () => getJson<Sentiment>("/api/sentiment");
export const getLeft = () => getJson<Left>("/api/left");
export const getCta = () => getJson<CTA>("/api/cta");
export const getFreshness = () => getJson<Freshness>("/api/freshness");
export const getSystemHealth = () => getJson<SystemHealth>("/api/system-health");

// Settings can be token-gated. These fetchers run server-side, so the
// server-only MARKET_API_TOKEN is forwarded as X-API-Token and never reaches
// the browser — matching the refresh route-handler pattern.
export async function getSettings(): Promise<Settings> {
  const headers: Record<string, string> = {};
  const token = process.env.MARKET_API_TOKEN;
  if (token) headers["X-API-Token"] = token;
  let res: Response;
  try {
    res = await backendFetch("/api/settings", { headers });
  } catch (error: unknown) {
    throw describeApiError(error, "/api/settings");
  }
  if (!res.ok) throw new Error(`API ${res.status}: /api/settings`);
  try {
    return (await res.json()) as Settings;
  } catch {
    throw new Error(`API returned invalid JSON (status ${res.status}): /api/settings`);
  }
}
