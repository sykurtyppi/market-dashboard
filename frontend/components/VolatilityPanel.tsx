"use client";

import { useState } from "react";
import type { Point } from "@/lib/api";
import VolatilityChart from "@/components/VolatilityChart";
import { maxGapDays } from "@/components/GapNotice";

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
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
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

// Inline-styled so the panel renders correctly even when a stale dev-server
// stylesheet is missing the newer class rules.
function StatCard({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div style={{ border: "1px solid var(--line)", borderRadius: 9, padding: "11px 12px", background: "#fafbfc" }}>
      <div style={{ fontSize: 10.5, textTransform: "uppercase", letterSpacing: ".07em", color: "var(--ink-faint)", fontWeight: 650 }}>{label}</div>
      <div className="mono" style={{ marginTop: 6, fontSize: 22, lineHeight: 1, fontWeight: 500, letterSpacing: "-.02em" }}>{value}</div>
      <div style={{ marginTop: 6, fontSize: 11, color: "var(--ink-faint)" }}>{note}</div>
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

  // Only finite timestamps — one unparseable date must not poison the anchor
  // (NaN would silently blank every finite range while "All" still renders).
  const times = [...vix, ...realizedVol, ...vrp].map((p) => new Date(p.date).getTime()).filter(Number.isFinite);
  const anchor = times.length ? Math.max(...times) : 0;
  const fVix = filterByRange(vix, anchor, days);
  const fRv = filterByRange(realizedVol, anchor, days);
  const fVrp = filterByRange(vrp, anchor, days);
  const stats = computeStats(fVrp);
  const gapDays = maxGapDays(fVrp);

  // A range needs >=2 points to draw a line. With sparse/gappy history a short
  // window can be empty — disable those so the control never lands on a blank
  // chart. Ranges re-enable automatically as fresh data fills in.
  const rangeCounts = new Map(
    RANGES.map((r) => [r.key, filterByRange(vrp, anchor, r.days).length]),
  );

  const legendItem: React.CSSProperties = { display: "inline-flex", alignItems: "center", gap: 6 };
  const swatch = (color: string, dashed = false): React.CSSProperties =>
    dashed
      ? { width: 12, height: 0, borderTop: `2px dashed ${color}`, display: "inline-block" }
      : { width: 12, height: 2, borderRadius: 2, background: color, display: "inline-block" };

  return (
    <div className="panel">
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, marginBottom: 4 }}>
        <span style={{ fontSize: 14, fontWeight: 600, letterSpacing: "-.01em" }}>VIX vs Realized Vol &amp; VRP Spread</span>
        <div style={{ display: "inline-flex", gap: 2, padding: 2, background: "var(--ground)", border: "1px solid var(--line)", borderRadius: 8 }} role="group" aria-label="Chart time range">
          {RANGES.map((r) => {
            const disabled = (rangeCounts.get(r.key) ?? 0) < 2;
            const active = range === r.key;
            return (
              <button
                key={r.key}
                type="button"
                aria-pressed={active}
                disabled={disabled}
                title={disabled ? "Not enough history in this window" : undefined}
                onClick={() => setRange(r.key)}
                style={{
                  fontSize: 11.5, padding: "3px 10px", borderRadius: 6, border: "none",
                  background: active ? "var(--surface)" : "transparent",
                  color: disabled ? "var(--line-strong)" : active ? "var(--accent-ink)" : "var(--ink-muted)",
                  fontWeight: 550, cursor: disabled ? "not-allowed" : "pointer",
                  fontVariantNumeric: "tabular-nums",
                  boxShadow: active ? "0 1px 2px rgba(22,24,29,.08)" : "none",
                }}
              >
                {r.label}
              </button>
            );
          })}
        </div>
      </div>

      <VolatilityChart vix={fVix} realizedVol={fRv} vrp={fVrp} />

      {gapDays > 30 ? (
        <div style={{ marginTop: 10, padding: "7px 11px", borderRadius: 7, border: "1px solid var(--warn)", background: "var(--warn-soft)", color: "var(--warn)", fontSize: 11.5, display: "inline-flex", alignItems: "center", gap: 7 }}>
          <span style={{ width: 7, height: 7, borderRadius: "50%", background: "var(--warn)", flex: "none" }} />
          Series has a {Math.round(gapDays)}-day gap — the line interpolates across missing data. Stats reflect only the points shown.
        </div>
      ) : null}

      <div style={{ display: "flex", gap: 14, marginTop: 12, fontSize: 11.5, color: "var(--ink-muted)", flexWrap: "wrap" }}>
        <span style={legendItem}><i style={swatch("var(--accent)")} />VIX implied vol</span>
        <span style={legendItem}><i style={swatch("var(--warn)", true)} />Realized vol 21d</span>
        <span style={legendItem}><i style={swatch("var(--good)")} />VRP spread (lower panel)</span>
        <span style={{ color: "var(--ink-faint)" }}>{stats.count} obs</span>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginTop: 14 }}>
        <StatCard label="Avg VRP" value={fmt(stats.avg)} note="Mean over plotted range" />
        <StatCard label="VRP Std Dev" value={fmt(stats.std)} note="Dispersion of the spread" />
        <StatCard label="Current Percentile" value={fmt(stats.percentile, "%")} note="Low = realized vol is rich" />
        <StatCard label="Max / Min VRP" value={fmt(stats.max)} note={`Min ${fmt(stats.min)}`} />
      </div>

      <VrpExplainer />
    </div>
  );
}

// Collapsible explainer for non-experts — the new-frontend equivalent of the
// old Streamlit "How to Use VRP" expander. Native <details> so it's accessible
// and needs no JS; inline-styled so it survives a stale dev-server stylesheet.
function VrpExplainer() {
  const term: React.CSSProperties = { color: "var(--ink)", fontWeight: 600 };
  return (
    <details style={{ marginTop: 14, border: "1px solid var(--line)", borderRadius: 9, background: "var(--surface)" }}>
      <summary style={{ cursor: "pointer", padding: "10px 14px", fontSize: 12.5, fontWeight: 600, color: "var(--ink)" }}>
        How to read this — what is the VRP?
      </summary>
      <div style={{ padding: "2px 14px 14px", fontSize: 12.5, color: "var(--ink-muted)", lineHeight: 1.6 }}>
        <p style={{ margin: "0 0 10px" }}>
          The <span style={term}>Volatility Risk Premium</span> is the gap between <em>implied</em> volatility
          (what the VIX prices options at) and <em>realized</em> volatility (how much the S&amp;P 500 actually moved over
          the last 21 days): <span style={term}>VRP = VIX − realized</span>.
        </p>
        <p style={{ margin: "0 0 10px" }}>
          <span style={term}>Reading the chart.</span> The top panel plots VIX and realized vol on the same scale —
          when the VIX line sits above realized, options are pricing in more movement than actually occurred (a premium).
          The lower panel shows that spread directly, against a zero line.
        </p>
        <ul style={{ margin: "0 0 8px", paddingLeft: 18, display: "flex", flexDirection: "column", gap: 5 }}>
          <li><span style={term}>High (above ~4):</span> options expensive relative to actual moves — historically a supportive, risk-on backdrop (favors selling volatility).</li>
          <li><span style={term}>Neutral (~0 to 4):</span> options roughly fairly priced; balanced risk/reward.</li>
          <li><span style={term}>Negative (below 0):</span> realized volatility is outrunning implied — the market may be underpricing risk (favors buying protection, trimming exposure).</li>
        </ul>
        <p style={{ margin: 0, color: "var(--ink-faint)", fontSize: 11.5 }}>
          Bands are the dashboard&apos;s model heuristics, not investment advice.
        </p>
      </div>
    </details>
  );
}
