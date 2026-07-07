import { NextResponse } from "next/server";
import { backendFetch, describeApiError } from "@/lib/api";

// Same-origin proxy for the client-side refresh-completion poll. On failure it
// must NOT report running:false (that would read as "refresh finished" and show
// a false success) — it returns an error status the client treats as unknown.
export async function GET() {
  try {
    const res = await backendFetch("/api/refresh/status");
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (error: unknown) {
    return NextResponse.json({ status: "error", detail: describeApiError(error, "/api/refresh/status").message }, { status: 502 });
  }
}
