// Typed client for the Market Risk Dashboard API.

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type State = "good" | "warn" | "crit" | "neutral";

export interface Freshness {
  status: string;
  age: string;
  is_fresh: boolean;
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
  regime: {
    composite_risk: number | null;
    components: RegimeComponent[];
  };
  metrics: Metric[];
  charts: {
    vrp_history: Point[];
    credit_spreads: { hy: Point[]; ig: Point[] };
  };
  detail: DetailRow[];
}

export async function getOverview(): Promise<Overview> {
  const res = await fetch(`${API_BASE}/api/overview`, { cache: "no-store" });
  if (!res.ok) throw new Error(`API ${res.status}: /api/overview`);
  return res.json();
}
