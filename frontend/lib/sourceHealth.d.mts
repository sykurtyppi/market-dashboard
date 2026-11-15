export interface HealthSource {
  key: string;
  name: string;
  status: string;
  state: "good" | "warn" | "crit" | "neutral";
  last_update: string | null;
  age_hours: number | null;
  message: string;
}
export interface SourceHealth {
  overall_status: string;
  timestamp: string | null;
  total_sources: number;
  sources: HealthSource[];
}
export function healthStatus(status: string): { label: string; state: HealthSource["state"]; order: number };
export function normalizeHealth(payload: unknown): SourceHealth;
export function filterSources(sources: HealthSource[], status: string, query: string): HealthSource[];
export function formatHealthDate(value: string | null): string;
export function formatHealthAge(value: number | null): string;
