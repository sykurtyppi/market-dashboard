import { NextResponse } from "next/server";
import { backendFetch, describeApiError } from "@/lib/api";

// Same-origin proxy for the client-side freshness poll during a refresh.
export async function GET() {
  try {
    const res = await backendFetch("/api/freshness");
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (error: unknown) {
    return NextResponse.json({ status: "error", detail: describeApiError(error, "/api/freshness").message }, { status: 502 });
  }
}
