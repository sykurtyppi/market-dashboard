"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

type RefreshResult = { status: string; detail?: string };
type StatusResult = { running: boolean };

const RefreshIcon = (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M3 12a9 9 0 019-9 9 9 0 016.7 3M21 12a9 9 0 01-9 9 9 9 0 01-6.7-3" />
    <path d="M21 3v5h-5M3 21v-5h5" />
  </svg>
);

export default function RefreshButton() {
  const router = useRouter();
  const [active, setActive] = useState(false);
  const [note, setNote] = useState<string | null>(null);

  const trigger = useMutation({
    mutationFn: async (): Promise<RefreshResult> => {
      const res = await fetch("/api/refresh", { method: "POST" });
      return res.json();
    },
    onSuccess: (data) => {
      if (data.status === "started" || data.status === "already_running") {
        setActive(true);
        setNote(data.status === "already_running" ? "Refresh already running…" : "Refreshing…");
      } else {
        setNote(data.detail ?? "Could not start refresh");
      }
    },
    onError: () => setNote("Could not reach the API"),
  });

  // Poll for completion only while a refresh is active.
  const status = useQuery({
    queryKey: ["refresh-status"],
    queryFn: async (): Promise<StatusResult> => {
      const res = await fetch("/api/refresh/status", { cache: "no-store" });
      return res.json();
    },
    enabled: active,
    refetchInterval: active ? 3000 : false,
  });

  useEffect(() => {
    if (active && status.data && status.data.running === false) {
      setActive(false);
      setNote("Data updated");
      router.refresh();
      const t = setTimeout(() => setNote(null), 4000);
      return () => clearTimeout(t);
    }
  }, [active, status.data, router]);

  const busy = active || trigger.isPending;

  return (
    <div className="refresh-wrap">
      <button
        className="btn ghost"
        type="button"
        onClick={() => trigger.mutate()}
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
