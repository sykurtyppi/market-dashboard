import { NextResponse } from "next/server";
import { API_BASE } from "@/lib/api";

// Same-origin proxy for the client-side freshness poll during a refresh.
export async function GET() {
  try {
    const res = await fetch(`${API_BASE}/api/freshness`, { cache: "no-store" });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ status: "error", detail: message }, { status: 502 });
  }
}
