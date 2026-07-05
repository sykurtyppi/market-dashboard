import { NextResponse } from "next/server";
import { API_BASE } from "@/lib/api";

// Same-origin proxy so the client can trigger a refresh without knowing the
// backend URL (kept server-only) or dealing with CORS.
export async function POST() {
  try {
    const res = await fetch(`${API_BASE}/api/refresh`, { method: "POST" });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ status: "error", detail: message }, { status: 502 });
  }
}
