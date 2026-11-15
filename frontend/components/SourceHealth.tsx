"use client";

import { useQuery } from "@tanstack/react-query";
import { useEffect, useId, useRef, useState } from "react";
import Link from "next/link";
import type { Freshness } from "@/lib/api";
import { filterSources, formatHealthAge, formatHealthDate, healthStatus, normalizeHealth } from "@/lib/sourceHealth.mjs";

async function readHealth({ signal }: { signal: AbortSignal }) {
  const response = await fetch("/api/source-health", { cache: "no-store", signal });
  if (!response.ok) throw new Error("Source health unavailable");
  return normalizeHealth(await response.json());
}

export default function SourceHealth({ expanded = false, snapshot }: { expanded?: boolean; snapshot?: Freshness }) {
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState("all");
  const [search, setSearch] = useState("");
  const id = useId();
  const toggle = useRef<HTMLButtonElement>(null);
  const health = useQuery({
    queryKey: ["source-health"],
    queryFn: readHealth,
    staleTime: 60_000,
    refetchInterval: 60_000,
    retry: false,
  });
  useEffect(() => {
    if (!open) return;
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false);
        toggle.current?.focus();
      }
    };
    document.addEventListener("keydown", close);
    return () => document.removeEventListener("keydown", close);
  }, [open]);

  const data = health.data;
  const verdict = healthStatus(data?.overall_status ?? "unknown");
  const label = health.isError ? "Unavailable" : health.isPending ? "Checking" : verdict.label;
  const state = health.isError || health.isPending ? "neutral" : verdict.state;
  const sources = data ? filterSources(data.sources, status, search) : [];
  const clearFilters = () => { setStatus("all"); setSearch(""); };
  const visible = expanded || open;

  return (
    <div className={expanded ? "source-health-full" : "source-health-control"}>
      {!expanded && <button
        ref={toggle}
        type="button"
        className="freshness health-toggle"
        aria-label={`Source health: ${label}`}
        aria-expanded={open}
        aria-controls={`${id}-panel`}
        onClick={() => setOpen(!open)}
      >
        <span className={`dot ${state}`} aria-hidden="true" />
        <span>Sources: {label}</span><span aria-hidden="true">{open ? "▴" : "▾"}</span>
      </button>}
      {visible && <section
        id={`${id}-panel`}
        className={expanded ? "panel source-health-panel" : "panel source-health-panel source-health-popover"}
        aria-label={expanded ? "Data source health" : "Source health details"}
        aria-busy={health.isFetching}
      >
        <div className="health-heading">
          <h2>Source health <span className="tag"><span className={`dot ${state}`} aria-hidden="true" />{label}</span></h2>
          {!expanded && <button type="button" className="btn ghost" onClick={() => { setOpen(false); toggle.current?.focus(); }}>Close</button>}
        </div>
        <p className="health-explanation">Status follows each source’s backend calendar and reporting cadence, not a four-hour cutoff. Elapsed age alone does not mean data is stale.</p>
        {health.isError && <p className="health-error" role="alert">Could not check source health. {data ? "Last successful check shown below; current health is unconfirmed." : "No health results available."}</p>}
        <div className="health-check-row">
          <span className="src">{health.isError && data ? "Last successful check" : "Backend checked"}: {formatHealthDate(data?.timestamp ?? null)}</span>
          <button type="button" className="btn ghost" disabled={health.isFetching} onClick={() => void health.refetch()}>
            {health.isFetching ? "Checking…" : health.isError ? "Retry health" : "Recheck health"}
          </button>
        </div>
        <p className="src">Read-only check · does not collect data or write snapshots. Checks automatically every minute while this page is active.</p>
        {snapshot && <p className="health-snapshot src">Daily snapshot: {snapshot.as_of ?? "date not reported"} · elapsed age: {snapshot.age || "not reported"}. Snapshot age is separate from source health.</p>}
        {health.isPending && <p role="status">Checking source health…</p>}
        {data && <>
          <div className="health-counts" aria-label="Source status counts">
            <button type="button" className="btn ghost" aria-pressed={status === "all"} onClick={() => setStatus("all")}>All {data.total_sources}</button>
            {["healthy", "stale", "degraded", "down", "unknown"].map((key) => {
              const count = data.sources.filter((source) => key === "unknown" ? healthStatus(source.status).label === "Unknown" : source.status === key).length;
              return <button type="button" key={key} className="btn ghost" aria-pressed={status === key} onClick={() => setStatus(key)}>{healthStatus(key).label} {count}</button>;
            })}
          </div>
          <div className="health-filters">
            <div className="health-filter-field"><label htmlFor={`${id}-search`}>Find a source</label><input id={`${id}-search`} type="search" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Name or reason" /></div>
            <div className="health-filter-field"><label htmlFor={`${id}-status`}>Source status</label><select id={`${id}-status`} value={status} onChange={(event) => setStatus(event.target.value)}>
              <option value="all">All sources</option><option value="attention">Needs attention</option>
              {["healthy", "stale", "degraded", "down", "unknown"].map((key) => <option value={key} key={key}>{healthStatus(key).label}</option>)}
            </select></div>
            <button type="button" className="btn ghost" onClick={clearFilters}>Clear filters</button>
          </div>
          <p className="src" role="status">Showing {sources.length} of {data.total_sources} sources{health.isError ? " · last-known results" : ""}</p>
          {data.total_sources === 0 ? <p className="empty-state">No sources reported by the backend.</p> : sources.length === 0 ? <p className="empty-state">No sources match these filters.</p> : <ul className="health-source-list">
            {sources.map((source) => <li key={source.key}>
              <div className="health-source-heading"><strong>{source.name}</strong><span className="tag"><span className={`dot ${source.state}`} aria-hidden="true" />{healthStatus(source.status).label}</span></div>
              <p>{source.message}</p>
              <dl><div><dt>Last update</dt><dd>{formatHealthDate(source.last_update)}</dd></div><div><dt>Age</dt><dd>{formatHealthAge(source.age_hours)}</dd></div></dl>
            </li>)}
          </ul>}
        </>}
        {!expanded && <Link className="health-page-link" href="/system-health">Open System Health →</Link>}
      </section>}
    </div>
  );
}
