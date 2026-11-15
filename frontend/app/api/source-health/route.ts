import { NextResponse } from "next/server";
import { normalizeHealth } from "@/lib/sourceHealth.mjs";

const BASE = process.env.MARKET_API_URL ?? "http://localhost:8000";
export const dynamic = "force-dynamic";

// Public health only: never invoke collectors, snapshot writes or admin endpoints.
export async function GET() {
  const headers = { "Cache-Control": "no-store" };
  try {
    const response = await fetch(`${BASE}/api/health`, {
      cache: "no-store",
      signal: AbortSignal.timeout(10_000),
    });
    if (!response.ok) throw new Error("Source health unavailable");
    const payload: unknown = await response.json();
    const health = normalizeHealth(payload); // Invalid is unavailable, never healthy.
    // Whitelist the public contract, not future backend metadata or configuration.
    return NextResponse.json({
      overall_status: health.overall_status,
      timestamp: health.timestamp,
      sources: Object.fromEntries(health.sources.map((source) => [source.key, {
        name: source.name,
        status: source.status,
        last_update: source.last_update,
        age_hours: source.age_hours,
        message: source.message,
      }])),
    }, { headers });
  } catch {
    return NextResponse.json({ detail: "Source health unavailable" }, { status: 502, headers });
  }
}
