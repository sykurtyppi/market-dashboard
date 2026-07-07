"use client";

import { useMutation } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

type RefreshResult = { status: string; detail?: string };
type StatusResult = { running: boolean };

const POLL_MS = 3_000;
const MAX_POLLS = 100; // 5-minute cap so a wedged backend run can't spin forever
const POLL_RETRIES = 2; // tolerate transient poll failures before giving up

const RefreshIcon = (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M3 12a9 9 0 019-9 9 9 0 016.7 3M21 12a9 9 0 01-9 9 9 9 0 01-6.7-3" />
    <path d="M21 3v5h-5M3 21v-5h5" />
  </svg>
);

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

// Poll /api/refresh/status until the run completes. A failed poll must NOT
// read as running:false ("finished" → false success), so bad responses throw
// after a couple of retries. Runs inside the mutation, which makes the whole
// trigger→poll→done lifecycle one async event: state updates happen only in
// the mutation callbacks, never in effects (the old two-effect design tripped
// react-hooks/set-state-in-effect).
async function pollUntilDone(): Promise<void> {
  let failures = 0;
  for (let i = 0; i < MAX_POLLS; i += 1) {
    await sleep(POLL_MS);
    try {
      const res = await fetch("/api/refresh/status", { cache: "no-store" });
      if (!res.ok) throw new Error(`status ${res.status}`);
      const body: StatusResult = await res.json();
      if (typeof body.running !== "boolean") throw new Error("bad status payload");
      if (!body.running) return;
      failures = 0;
    } catch {
      failures += 1;
      if (failures > POLL_RETRIES) {
        throw new Error("Couldn't confirm refresh — check data freshness");
      }
    }
  }
  throw new Error("Refresh is taking unusually long — check data freshness");
}

export default function RefreshButton() {
  const router = useRouter();
  const [note, setNote] = useState<string | null>(null);
  const noteTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Only a cleanup — no state writes — so unmount can't leak the note timer.
  useEffect(() => () => {
    if (noteTimer.current) clearTimeout(noteTimer.current);
  }, []);

  const showNote = (text: string, clearAfterMs: number) => {
    if (noteTimer.current) clearTimeout(noteTimer.current);
    setNote(text);
    noteTimer.current = setTimeout(() => setNote(null), clearAfterMs);
  };

  const refresh = useMutation({
    mutationFn: async (): Promise<void> => {
      const res = await fetch("/api/refresh", { method: "POST" });
      const data: RefreshResult = await res.json();
      if (data.status !== "started" && data.status !== "already_running") {
        throw new Error(data.detail ?? "Could not start refresh");
      }
      await pollUntilDone();
    },
    onSuccess: () => {
      showNote("Data updated", 4000);
      router.refresh();
    },
    onError: (error: Error) => {
      showNote(error.message || "Could not reach the API", 5000);
    },
  });

  const busy = refresh.isPending;

  return (
    <div className="refresh-wrap">
      <button
        className="btn ghost"
        type="button"
        onClick={() => refresh.mutate()}
        disabled={busy}
        aria-busy={busy}
      >
        <span className={busy ? "spin" : undefined}>{RefreshIcon}</span>
        {busy ? "Refreshing…" : "Refresh"}
      </button>
      {note ? <span className="refresh-note">{note}</span> : null}
    </div>
  );
}
