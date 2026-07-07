"use client";

import { useState } from "react";
import type { CSSProperties, MouseEvent } from "react";
import type { Point } from "@/lib/api";

// Shared chart primitives (AreaChart, MultiLineChart) built on the same
// hardened pattern as VolatilityChart:
//  - x is positioned by *time*, not by array index, so series of different
//    lengths align honestly and gaps show as real gaps in spacing.
//  - Real y-axis ticks and x-axis date labels (HTML overlays, crisp at any
//    scale), vector-effect strokes so lines never warp under stretch.
//  - Non-finite points are filtered before scaling (one NaN must not blank
//    the whole chart), hover is guarded against zero-width rects and re-derived
//    from current data every render (no stale-index crash).
// Inline-styled so the charts survive a stale dev-server stylesheet.

const H = 210;
const PAD_LEFT = 40;
const PAD_BOTTOM = 20;

type CleanPoint = { t: number; date: string; value: number };
type SeriesInput = { points: Point[]; color: string; label?: string; dashed?: boolean; area?: boolean };
type CleanSeries = { pts: CleanPoint[]; color: string; label: string; dashed: boolean; area: boolean };

// Parse YYYY-MM-DD as a *local* date. `new Date("2026-07-02")` is UTC midnight,
// which renders as the previous day in negative-UTC-offset zones.
function parseLocal(date: string): Date {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(date);
  return m ? new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3])) : new Date(date);
}

function fmtDateShort(date: string): string {
  const d = parseLocal(date);
  return Number.isNaN(d.getTime()) ? date : d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function fmtDateFull(date: string): string {
  const d = parseLocal(date);
  return Number.isNaN(d.getTime()) ? date : d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

function niceTicks(min: number, max: number, count: number): number[] {
  const span = max - min || 1;
  const step0 = span / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const norm = step0 / mag;
  const step = (norm >= 5 ? 5 : norm >= 2 ? 2 : norm >= 1 ? 1 : 0.5) * mag;
  const first = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let v = first; v <= max + step * 0.001; v += step) ticks.push(Number(v.toFixed(4)));
  // Flat data can yield zero ticks; fall back so the axis isn't bare.
  return ticks.length ? ticks : [Number(min.toFixed(2))];
}

// Thousands get grouped, everything else keeps two decimals so small
// spreads/ratios read precisely.
function fmtVal(v: number, unit: string): string {
  const n = Math.abs(v) >= 1000 ? Math.round(v).toLocaleString() : v.toFixed(2);
  return unit ? `${n}${unit}` : n;
}

// Decimals follow the tick *step*, not the magnitude — a 0.05 step needs two
// decimals or adjacent labels collapse into duplicates ("6.7, 6.7, 6.6").
function fmtTick(t: number, step: number): string {
  if (Math.abs(t) >= 1000) return Math.round(t).toLocaleString();
  const decimals = Math.max(0, Math.min(2, -Math.floor(Math.log10(step || 1))));
  return t.toFixed(decimals);
}

// Drop non-finite values/dates and sort by time so one bad point can't poison
// the scale or the path.
function clean(se: SeriesInput, idx: number): CleanSeries {
  const pts = se.points
    .map((p) => ({ t: parseLocal(p.date).getTime(), date: p.date, value: p.value }))
    .filter((p) => Number.isFinite(p.t) && Number.isFinite(p.value))
    .sort((a, b) => a.t - b.t);
  return { pts, color: se.color, label: se.label ?? `Series ${idx + 1}`, dashed: se.dashed ?? false, area: se.area ?? false };
}

function nearestIdx(pts: CleanPoint[], t: number): number {
  let best = 0;
  let bestD = Infinity;
  for (let i = 0; i < pts.length; i += 1) {
    const d = Math.abs(pts[i].t - t);
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}

// Stack endpoint tags that would overlap, then clamp into the plot so a pile
// of near-equal series can't push labels outside the panel.
function labelOffsets(yFracs: number[]): number[] {
  const ordered = yFracs.map((yFrac, idx) => ({ yFrac, idx })).sort((a, b) => a.yFrac - b.yFrac);
  const offsets = new Array(yFracs.length).fill(0);
  for (let i = 1; i < ordered.length; i += 1) {
    const prev = ordered[i - 1];
    const curr = ordered[i];
    const prevPx = prev.yFrac * H + offsets[prev.idx];
    const currPx = curr.yFrac * H + offsets[curr.idx];
    if (currPx - prevPx < 15) offsets[curr.idx] = prevPx + 15 - curr.yFrac * H;
  }
  return offsets.map((off, i) => {
    const px = yFracs[i] * H + off;
    return Math.max(9, Math.min(H - 9, px)) - yFracs[i] * H;
  });
}

function ChartBase({ series, unit = "" }: { series: SeriesInput[]; unit?: string }) {
  const [hover, setHover] = useState<number | null>(null); // 0..1 ratio across the time domain

  const all = series.map(clean).filter((se) => se.pts.length >= 2);
  if (all.length === 0) return <Empty />;

  const times = all.flatMap((se) => [se.pts[0].t, se.pts[se.pts.length - 1].t]);
  const t0 = Math.min(...times);
  const t1 = Math.max(...times);
  const xOf = (t: number) => (t1 > t0 ? ((t - t0) / (t1 - t0)) * 100 : 50);

  const values = all.flatMap((se) => se.pts.map((p) => p.value));
  const dMin = Math.min(...values);
  const dMax = Math.max(...values);
  const pad = (dMax - dMin || 1) * 0.12;
  const vMin = dMin - pad;
  const vMax = dMax + pad;
  const yOf = (v: number) => 100 - ((v - vMin) / (vMax - vMin)) * 100;

  const ticks = niceTicks(dMin, dMax, 4);
  const tickStep = ticks.length > 1 ? ticks[1] - ticks[0] : Math.abs(ticks[0]) || 1;
  const zeroInRange = dMin < 0 && dMax > 0;

  const linePath = (pts: CleanPoint[]) =>
    pts.map((p, i) => `${i ? "L" : "M"}${xOf(p.t).toFixed(2)},${yOf(p.value).toFixed(2)}`).join(" ");
  const areaPath = (pts: CleanPoint[]) => {
    const bottom = 100;
    return `${linePath(pts)} L${xOf(pts[pts.length - 1].t).toFixed(2)},${bottom} L${xOf(pts[0].t).toFixed(2)},${bottom} Z`;
  };

  // Longest series anchors the x-date labels and the tooltip's headline date.
  // Endpoints get priority; interior labels only render if they keep >=10% x
  // clearance — gappy series can snap two candidates onto near-identical x.
  const base = all.reduce((longest, se) => (se.pts.length > longest.pts.length ? se : longest), all[0]);
  const xIdx: number[] = [];
  for (const f of [0, 1, 0.5, 0.25, 0.75]) {
    const i = nearestIdx(base.pts, t0 + f * (t1 - t0));
    const x = xOf(base.pts[i].t);
    if (!xIdx.includes(i) && xIdx.every((j) => Math.abs(xOf(base.pts[j].t) - x) >= 10)) xIdx.push(i);
  }
  xIdx.sort((a, b) => a - b);

  const offsets = labelOffsets(all.map((se) => yOf(se.pts[se.pts.length - 1].value) / 100));

  // Hover values are re-derived from *current* data each render, so a stale
  // ratio can never index out of bounds after the data shrinks.
  const onMove = (e: MouseEvent<HTMLDivElement>) => {
    const b = e.currentTarget.getBoundingClientRect();
    if (b.width <= 0) return; // zero-width (hidden/collapsed) → NaN ratio
    setHover(Math.max(0, Math.min(1, (e.clientX - b.left) / b.width)));
  };
  const targetT = hover !== null ? t0 + hover * (t1 - t0) : null;
  const basePt = targetT !== null ? base.pts[nearestIdx(base.pts, targetT)] : null;
  const hoverRows = basePt !== null
    ? all.map((se) => {
        const p = se.pts[nearestIdx(se.pts, basePt.t)];
        return { label: se.label, color: se.color, value: p.value, date: p.date, offDate: p.date !== basePt.date };
      })
    : [];
  const hoverX = basePt !== null ? xOf(basePt.t) : 0;
  const tooltipLeft = Math.max(9, Math.min(78, hoverX));

  const summary = `${all.map((se) => {
    const last = se.pts[se.pts.length - 1];
    return `${se.label} ${fmtVal(last.value, unit)} as of ${last.date}`;
  }).join("; ")}. ${base.pts.length} points from ${base.pts[0].date} to ${base.pts[base.pts.length - 1].date}.`;

  const yLabel: CSSProperties = { position: "absolute", left: -36, transform: "translateY(-50%)", fontSize: 10.5, color: "var(--ink-faint)", fontVariantNumeric: "tabular-nums", pointerEvents: "none", whiteSpace: "nowrap" };
  const xLabel: CSSProperties = { position: "absolute", bottom: -19, fontSize: 10.5, color: "var(--ink-faint)", fontVariantNumeric: "tabular-nums", pointerEvents: "none", whiteSpace: "nowrap" };
  const tag = (color: string, topPct: number, offset: number): CSSProperties => ({
    position: "absolute", right: 2, top: `calc(${topPct}% + ${offset}px)`, transform: "translateY(-50%)",
    fontSize: 11, fontWeight: 600, padding: "1px 5px", borderRadius: 5, background: "var(--surface)",
    border: "1px solid var(--line)", color, fontVariantNumeric: "tabular-nums", whiteSpace: "nowrap",
    pointerEvents: "none", boxShadow: "0 1px 2px rgba(22,24,29,.05)",
  });

  return (
    <div role="img" aria-label={summary} style={{ position: "relative", paddingLeft: PAD_LEFT, paddingBottom: PAD_BOTTOM, paddingTop: 4 }}>
      <div style={{ position: "relative", height: H }}>
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={{ position: "absolute", inset: 0, width: "100%", height: "100%", display: "block", overflow: "visible" }} aria-hidden>
          {ticks.map((t) => (
            <line key={t} x1="0" y1={yOf(t)} x2="100" y2={yOf(t)} stroke="var(--line)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
          ))}
          {zeroInRange ? (
            <line x1="0" y1={yOf(0)} x2="100" y2={yOf(0)} stroke="var(--line-strong)" strokeWidth="1" strokeDasharray="3 3" vectorEffect="non-scaling-stroke" />
          ) : null}
          {all.map((se, idx) => (
            <g key={idx}>
              {se.area ? <path d={areaPath(se.pts)} fill={se.color} fillOpacity="0.08" /> : null}
              <path d={linePath(se.pts)} fill="none" stroke={se.color} strokeWidth="1.5" strokeDasharray={se.dashed ? "4 4" : undefined} strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
            </g>
          ))}
          {basePt !== null ? (
            <line x1={hoverX} y1="0" x2={hoverX} y2="100" stroke="var(--ink-faint)" strokeWidth="1" strokeDasharray="3 3" vectorEffect="non-scaling-stroke" />
          ) : null}
        </svg>

        {/* Y-axis ticks */}
        {ticks.map((t) => (
          <span key={t} style={{ ...yLabel, top: `${yOf(t)}%` }}>{fmtTick(t, tickStep)}</span>
        ))}

        {/* X-axis date labels (from real data points, dates positioned by time) */}
        {xIdx.map((i) => (
          <span key={i} style={{ ...xLabel, left: `${xOf(base.pts[i].t)}%`, transform: `translateX(${i === 0 ? "0" : i === base.pts.length - 1 ? "-100%" : "-50%"})` }}>
            {fmtDateShort(base.pts[i].date)}
          </span>
        ))}

        {/* Endpoint dots + latest-value tags */}
        {all.map((se, idx) => {
          const last = se.pts[se.pts.length - 1];
          return (
            <span key={`d${idx}`} style={{ position: "absolute", left: `${xOf(last.t)}%`, top: `${yOf(last.value)}%`, transform: "translate(-50%, -50%)", width: 7, height: 7, borderRadius: "50%", background: se.color, border: "1.5px solid var(--surface)", pointerEvents: "none" }} />
          );
        })}
        {all.map((se, idx) => {
          const last = se.pts[se.pts.length - 1];
          return <span key={`t${idx}`} style={tag(se.color, yOf(last.value), offsets[idx])}>{fmtVal(last.value, unit)}</span>;
        })}

        {/* Hover dots */}
        {hoverRows.map((r) => (
          <span key={r.label} style={{ position: "absolute", left: `${hoverX}%`, top: `${yOf(r.value)}%`, transform: "translate(-50%, -50%)", width: 8, height: 8, borderRadius: "50%", background: r.color, border: "1.5px solid var(--surface)", pointerEvents: "none" }} />
        ))}

        {/* Hover capture */}
        <div style={{ position: "absolute", inset: 0, cursor: "crosshair" }} onMouseMove={onMove} onMouseLeave={() => setHover(null)} />

        {/* Tooltip — a row whose nearest point is from a different date than the
            headline says so explicitly instead of silently borrowing the date. */}
        {basePt !== null ? (
          <div style={{ position: "absolute", top: 6, left: `${tooltipLeft}%`, transform: "translateX(-50%)", minWidth: 160, padding: "8px 10px", borderRadius: 8, border: "1px solid var(--line)", background: "rgba(255,255,255,.97)", boxShadow: "0 10px 26px rgba(22,24,29,.12)", pointerEvents: "none", zIndex: 2, fontVariantNumeric: "tabular-nums" }}>
            <div style={{ fontSize: 10.5, color: "var(--ink-faint)", marginBottom: 6 }}>{fmtDateFull(basePt.date)}</div>
            {hoverRows.map((r) => (
              <div key={r.label} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 14, fontSize: 11.5, color: "var(--ink-muted)", marginTop: 3 }}>
                <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                  <i style={{ width: 8, height: 8, borderRadius: "50%", background: r.color, display: "inline-block" }} />
                  {r.label}{r.offDate ? ` (${fmtDateShort(r.date)})` : ""}
                </span>
                <b style={{ color: "var(--ink)", fontWeight: 600 }}>{fmtVal(r.value, unit)}</b>
              </div>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}

export function AreaChart({ points, color, unit = "", label = "Value" }: { points: Point[]; color: string; unit?: string; label?: string }) {
  return <ChartBase series={[{ points, color, label, area: true }]} unit={unit} />;
}

export function MultiLineChart({ series, unit = "" }: { series: { points: Point[]; color: string; label?: string; dashed?: boolean }[]; unit?: string }) {
  return <ChartBase series={series} unit={unit} />;
}

function Empty() {
  return (
    <div style={{ height: 120, display: "grid", placeItems: "center", color: "var(--ink-faint)", fontSize: 12 }}>
      No history available
    </div>
  );
}
