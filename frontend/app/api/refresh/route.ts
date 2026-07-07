import { NextResponse } from "next/server";
import { backendFetch, describeApiError } from "@/lib/api";

// Same-origin proxy so the client can trigger a refresh without knowing the
// backend URL (kept server-only) or dealing with CORS. Forwards the server-only
// MARKET_API_TOKEN as X-API-Token so refresh keeps working when the backend is
// token-gated — the token never reaches the browser.
export async function POST() {
  try {
    const headers: Record<string, string> = {};
    const token = process.env.MARKET_API_TOKEN;
    if (token) headers["X-API-Token"] = token;

    const res = await backendFetch("/api/refresh", { method: "POST", headers });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (error: unknown) {
    return NextResponse.json({ status: "error", detail: describeApiError(error, "/api/refresh").message }, { status: 502 });
  }
}
