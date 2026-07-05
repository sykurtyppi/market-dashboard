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

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`API ${res.status}: ${path}`);
  return res.json() as Promise<T>;
}

export const getOverview = () => getJson<Overview>("/api/overview");
export const getVolatility = () => getJson<Volatility>("/api/volatility");
export const getBreadth = () => getJson<Breadth>("/api/breadth");
export const getFreshness = () => getJson<Freshness>("/api/freshness");
