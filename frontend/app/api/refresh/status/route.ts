import { NextResponse } from "next/server";
import { API_BASE } from "@/lib/api";

// Same-origin proxy for the client-side refresh-completion poll. On failure it
// must NOT report running:false (that would read as "refresh finished" and show
// a false success) — it returns an error status the client treats as unknown.
export async function GET() {
  try {
    const res = await fetch(`${API_BASE}/api/refresh/status`, { cache: "no-store" });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ status: "error", detail: message }, { status: 502 });
  }
}
