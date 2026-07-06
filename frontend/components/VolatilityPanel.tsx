"use client";

import { useState } from "react";
import { Point } from "@/lib/api";
import { VRPCompositeChart } from "@/components/Charts";

type Range = "1M" | "3M" | "6M" | "ALL";

const RANGES: { key: Range; label: string; days: number }[] = [
  { key: "1M", label: "1M", days: 30 },
  { key: "3M", label: "3M", days: 90 },
  { key: "6M", label: "6M", days: 180 },
  { key: "ALL", label: "All", days: Infinity },
];

const DAY_MS = 86_400_000;

// Filter to a trailing window anchored on the latest available date (robust to
// stale data — we measure back from the newest point, not "today").
function filterByRange(points: Point[], anchor: number, days: number): Point[] {
  if (!Number.isFinite(days)) return points;
  const cutoff = anchor - days * DAY_MS;
  return points.filter((p) => new Date(p.date).getTime() >= cutoff);
}

function fmt(value: number | null, unit = ""): string {
  if (value === null || value === undefined) return "—";
  return `${value.toFixed(2)}${unit}`;
}

type Stats = { avg: number | null; std: number | null; percentile: number | null; max: number | null; min: number | null; count: number };

// Range-aware VRP stats so the cards always match what's on screen. Mirrors the
// backend _vrp_stats (population std, percentile rank of the latest value).
function computeStats(vrp: Point[]): Stats {
  const values = vrp.map((p) => p.value).filter((v): v is number => v !== null && v !== undefined);
  const n = values.length;
  if (n === 0) return { avg: null, std: null, percentile: null, max: null, min: null, count: 0 };
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
  const current = values[values.length - 1];
  return {
    avg: mean,
    std: n > 1 ? Math.sqrt(variance) : 0,
    percentile: (values.filter((v) => v < current).length / n) * 100,
    max: Math.max(...values),
    min: Math.min(...values),
    count: n,
  };
}

function StatCard({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div className="stat-card">
      <div className="k">{label}</div>
      <div className="v mono">{value}</div>
      <div className="note">{note}</div>
    </div>
  );
}

interface VolatilityPanelProps {
  vix: Point[];
  realizedVol: Point[];
  vrp: Point[];
}

export default function VolatilityPanel({ vix, realizedVol, vrp }: VolatilityPanelProps) {
  const [range, setRange] = useState<Range>("ALL");
  const days = RANGES.find((r) => r.key === range)?.days ?? Infinity;

  const anchor = Math.max(0, ...[...vix, ...realizedVol, ...vrp].map((p) => new Date(p.date).getTime()));
  const fVix = filterByRange(vix, anchor, days);
  const fRv = filterByRange(realizedVol, anchor, days);
  const fVrp = filterByRange(vrp, anchor, days);
  const stats = computeStats(fVrp);

  // A range needs >=2 points to draw a line. With sparse/gappy history a short
  // window can be empty — disable those so the control never lands on a blank
  // chart. Ranges re-enable automatically as fresh data fills in.
  const rangeCounts = new Map(
    RANGES.map((r) => [r.key, filterByRange(vrp, anchor, r.days).length]),
  );

  return (
    <div className="panel">
      <div className="panel-head">
        <span className="t">VIX vs Realized Vol &amp; VRP Spread</span>
        <div className="range-select" role="group" aria-label="Chart time range">
          {RANGES.map((r) => {
            const disabled = (rangeCounts.get(r.key) ?? 0) < 2;
            return (
              <button
                key={r.key}
                type="button"
                className={`range-btn${range === r.key ? " active" : ""}`}
                aria-pressed={range === r.key}
                disabled={disabled}
                title={disabled ? "Not enough history in this window" : undefined}
                onClick={() => setRange(r.key)}
              >
                {r.label}
              </button>
            );
          })}
        </div>
      </div>

      <VRPCompositeChart vix={fVix} realizedVol={fRv} vrp={fVrp} />

      <div className="legend">
        <span><i style={{ background: "var(--crit)" }} />VIX implied vol</span>
        <span><i style={{ background: "var(--accent)" }} />Realized vol 21d</span>
        <span><i style={{ background: "var(--good)" }} />VRP spread</span>
        <span style={{ color: "var(--ink-faint)" }}>{stats.count} obs · zero line = implied equals realized</span>
      </div>

      <div className="stat-grid">
        <StatCard label="Avg VRP" value={fmt(stats.avg)} note="Mean over selected range" />
        <StatCard label="VRP Std Dev" value={fmt(stats.std)} note="Dispersion of the spread" />
        <StatCard label="Current Percentile" value={fmt(stats.percentile, "%")} note="Low = realized vol is rich" />
        <StatCard label="Max / Min VRP" value={fmt(stats.max)} note={`Min ${fmt(stats.min)}`} />
      </div>
    </div>
  );
}
