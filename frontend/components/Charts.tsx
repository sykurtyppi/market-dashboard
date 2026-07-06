"use client";

import { useState } from "react";
import type { MouseEvent } from "react";
import type { Point } from "@/lib/api";

const W = 480;
const H = 190;

type Scale = { x: (i: number, n: number) => number; y: (v: number) => number };
type ChartSeries = { points: Point[]; color: string; label?: string; dashed?: boolean };
type HoverPoint = { label: string; value: number; color: string; unit: string };
type HoverState = { x: number; date: string; values: HoverPoint[] } | null;

function makeScale(all: number[]): Scale {
  let min = Math.min(...all);
  let max = Math.max(...all);
  if (min === max) { min -= 1; max += 1; }
  const pad = (max - min) * 0.12;
  min -= pad; max += pad;
  return {
    x: (i, n) => (n <= 1 ? 0 : (i / (n - 1)) * W),
    y: (v) => H - ((v - min) / (max - min)) * H,
  };
}

function linePath(points: Point[], s: Scale): string {
  return points
    .map((p, i) => `${i === 0 ? "M" : "L"}${s.x(i, points.length).toFixed(1)},${s.y(p.value).toFixed(1)}`)
    .join(" ");
}

function areaPath(points: Point[], s: Scale): string {
  return `${linePath(points, s)} L${W},${H} L0,${H} Z`;
}

function chartIndexFromMouse(event: MouseEvent<SVGRectElement>, n: number): number {
  const bounds = event.currentTarget.getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width));
  return Math.round(ratio * (n - 1));
}

function proportionalIndex(baseIdx: number, baseLen: number, seriesLen: number): number {
  if (seriesLen <= 1 || baseLen <= 1) return 0;
  return Math.max(0, Math.min(seriesLen - 1, Math.round((baseIdx / (baseLen - 1)) * (seriesLen - 1))));
}

function fmtDate(date: string): string {
  // Parse YYYY-MM-DD as a *local* date. `new Date("2026-07-02")` is treated as
  // UTC midnight, which toLocaleDateString then renders as the previous day in
  // any negative-UTC-offset zone (e.g. "Jul 1" in the US) — an off-by-one.
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(date);
  const parsed = m ? new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3])) : new Date(date);
  if (Number.isNaN(parsed.getTime())) return date;
  return parsed.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

// Format the current-value tag: thousands get grouped, everything else keeps
// two decimals so small spreads/ratios read precisely.
function fmtTag(v: number, unit: string): string {
  const n = Math.abs(v) >= 1000 ? Math.round(v).toLocaleString() : v.toFixed(2);
  return unit ? `${n}${unit}` : n;
}

// A crisp HTML label pinned to a line's latest point. Rendered outside the SVG
// so it stays at true pixel size regardless of how the chart is scaled.
function ValueTag({ value, color, unit, yFrac, offset = 0 }: { value: number; color: string; unit: string; yFrac: number; offset?: number }) {
  return (
    <span
      className="chart-tag mono"
      style={{ top: `calc(${(yFrac * 100).toFixed(2)}% + ${offset}px)`, color }}
    >
      {fmtTag(value, unit)}
    </span>
  );
}

function labelOffsets(items: { yFrac: number }[]): number[] {
  const ordered = items.map((item, idx) => ({ ...item, idx })).sort((a, b) => a.yFrac - b.yFrac);
  const offsets = new Array(items.length).fill(0);
  for (let i = 1; i < ordered.length; i += 1) {
    const prev = ordered[i - 1];
    const curr = ordered[i];
    if (Math.abs(curr.yFrac - prev.yFrac) < 0.055) {
      offsets[curr.idx] = offsets[prev.idx] + 16;
    }
  }
  return offsets;
}

function GridLines() {
  return (
    <>
      <line x1="0" y1={H * 0.25} x2={W} y2={H * 0.25} stroke="var(--line)" strokeWidth="1" />
      <line x1="0" y1={H * 0.5} x2={W} y2={H * 0.5} stroke="var(--line)" strokeWidth="1" />
      <line x1="0" y1={H * 0.75} x2={W} y2={H * 0.75} stroke="var(--line)" strokeWidth="1" />
    </>
  );
}

function HoverOverlay({ hover }: { hover: HoverState }) {
  if (!hover) return null;
  const left = Math.max(8, Math.min(78, (hover.x / W) * 100));
  return (
    <div className="chart-tooltip mono" style={{ left: `${left}%` }}>
      <div className="chart-tooltip-date">{fmtDate(hover.date)}</div>
      {hover.values.map((v) => (
        <div className="chart-tooltip-row" key={v.label}>
          <span><i style={{ background: v.color }} />{v.label}</span>
          <b>{fmtTag(v.value, v.unit)}</b>
        </div>
      ))}
    </div>
  );
}

export function AreaChart({ points, color, unit = "", label = "Value" }: { points: Point[]; color: string; unit?: string; label?: string }) {
  const [hover, setHover] = useState<HoverState>(null);
  if (points.length < 2) return <Empty />;
  const s = makeScale(points.map((p) => p.value));
  const path = linePath(points, s);
  const last = points[points.length - 1];
  const zeroInRange = points.some((p) => p.value > 0) && points.some((p) => p.value < 0);
  const hoverY = hover ? s.y(hover.values[0].value) : null;
  return (
    <div className="chart-wrap interactive">
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden>
        <GridLines />
        {zeroInRange && (
          <line x1="0" y1={s.y(0)} x2={W} y2={s.y(0)} stroke="var(--line-strong)" strokeWidth="1" strokeDasharray="3 3" />
        )}
        <path d={areaPath(points, s)} fill={color} fillOpacity="0.1" />
        <path d={path} fill="none" stroke={color} strokeWidth="2" />
        <circle cx={W} cy={s.y(last.value)} r="3.2" fill={color} />
        {hover && hoverY !== null ? (
          <g className="chart-focus">
            <line x1={hover.x} y1="0" x2={hover.x} y2={H} />
            <circle cx={hover.x} cy={hoverY} r="4" fill={color} />
          </g>
        ) : null}
        <rect
          width={W}
          height={H}
          fill="transparent"
          onMouseMove={(event) => {
            const idx = chartIndexFromMouse(event, points.length);
            const point = points[idx];
            setHover({ x: s.x(idx, points.length), date: point.date, values: [{ label, value: point.value, color, unit }] });
          }}
          onMouseLeave={() => setHover(null)}
        />
      </svg>
      <ValueTag value={last.value} color={color} unit={unit} yFrac={s.y(last.value) / H} />
      <HoverOverlay hover={hover} />
    </div>
  );
}

export function MultiLineChart({ series, unit = "" }: { series: ChartSeries[]; unit?: string }) {
  const [hover, setHover] = useState<HoverState>(null);
  const all = series.flatMap((se) => se.points.map((p) => p.value));
  if (all.length < 2) return <Empty />;
  const s = makeScale(all);
  const base = series.reduce((longest, se) => (se.points.length > longest.points.length ? se : longest), series[0]);
  const labels = series
    .filter((se) => se.points.length >= 2)
    .map((se) => ({ yFrac: s.y(se.points[se.points.length - 1].value) / H }));
  const offsets = labelOffsets(labels);
  return (
    <div className="chart-wrap interactive">
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden>
        <GridLines />
        {series.map((se, idx) =>
          se.points.length >= 2 ? (
            <g key={idx}>
              <path d={linePath(se.points, s)} fill="none" stroke={se.color} strokeWidth="2" strokeDasharray={se.dashed ? "5 4" : undefined} />
              <circle cx={W} cy={s.y(se.points[se.points.length - 1].value)} r="3.2" fill={se.color} />
            </g>
          ) : null
        )}
        {hover ? (
          <g className="chart-focus">
            <line x1={hover.x} y1="0" x2={hover.x} y2={H} />
            {hover.values.map((v) => <circle key={v.label} cx={hover.x} cy={s.y(v.value)} r="4" fill={v.color} />)}
          </g>
        ) : null}
        <rect
          width={W}
          height={H}
          fill="transparent"
          onMouseMove={(event) => {
            const idx = chartIndexFromMouse(event, base.points.length);
            const values = series
              .filter((se) => se.points.length >= 2)
              .map((se) => {
                const point = se.points[proportionalIndex(idx, base.points.length, se.points.length)];
                return { label: se.label ?? "Series", value: point.value, color: se.color, unit };
              });
            setHover({ x: s.x(idx, base.points.length), date: base.points[idx].date, values });
          }}
          onMouseLeave={() => setHover(null)}
        />
      </svg>
      {series.filter((se) => se.points.length >= 2).map((se, idx) => {
        const last = se.points[se.points.length - 1];
        return <ValueTag key={idx} value={last.value} color={se.color} unit={unit} yFrac={s.y(last.value) / H} offset={offsets[idx]} />;
      })}
      <HoverOverlay hover={hover} />
    </div>
  );
}

export function VRPCompositeChart({
  vix,
  realizedVol,
  vrp,
}: {
  vix: Point[];
  realizedVol: Point[];
  vrp: Point[];
}) {
  const [hover, setHover] = useState<HoverState>(null);
  const volValues = [...vix, ...realizedVol].map((p) => p.value);
  const vrpValues = vrp.map((p) => p.value);
  if (vix.length < 2 || realizedVol.length < 2 || vrp.length < 2) return <Empty />;

  const volScale = makeScale(volValues);
  const vrpScale = makeScale([...vrpValues, 0]);
  const base = vix.length >= realizedVol.length ? vix : realizedVol;
  const lastVix = vix[vix.length - 1];
  const lastRv = realizedVol[realizedVol.length - 1];
  const lastVrp = vrp[vrp.length - 1];
  const labelItems = [
    { yFrac: volScale.y(lastVix.value) / H },
    { yFrac: volScale.y(lastRv.value) / H },
    { yFrac: vrpScale.y(lastVrp.value) / H },
  ];
  const offsets = labelOffsets(labelItems);

  // Plain computation, not useMemo: a hook here would sit *after* the early
  // return above, violating the Rules of Hooks (inconsistent hook count if the
  // data ever crosses the 2-point threshold on the same mounted instance). It's
  // cheap enough to recompute each render.
  const hoverDots = hover
    ? hover.values.map((v) => ({ ...v, y: v.label === "VRP" ? vrpScale.y(v.value) : volScale.y(v.value) }))
    : [];

  return (
    <div className="chart-wrap interactive vrp-composite">
      <svg className="chart tall" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden>
        <GridLines />
        <line x1="0" y1={vrpScale.y(0)} x2={W} y2={vrpScale.y(0)} stroke="var(--line-strong)" strokeWidth="1" strokeDasharray="3 3" />
        <path d={areaPath(vrp, vrpScale)} fill="var(--good)" fillOpacity="0.10" />
        <path d={linePath(vrp, vrpScale)} fill="none" stroke="var(--good)" strokeWidth="2.4" />
        <path d={linePath(vix, volScale)} fill="none" stroke="var(--crit)" strokeWidth="2" />
        <path d={linePath(realizedVol, volScale)} fill="none" stroke="var(--accent)" strokeWidth="2" strokeDasharray="5 4" />
        <circle cx={W} cy={volScale.y(lastVix.value)} r="3.2" fill="var(--crit)" />
        <circle cx={W} cy={volScale.y(lastRv.value)} r="3.2" fill="var(--accent)" />
        <circle cx={W} cy={vrpScale.y(lastVrp.value)} r="4" fill="var(--good)" stroke="var(--surface)" strokeWidth="1" />
        {hover ? (
          <g className="chart-focus">
            <line x1={hover.x} y1="0" x2={hover.x} y2={H} />
            {hoverDots.map((v) => <circle key={v.label} cx={hover.x} cy={v.y} r="4" fill={v.color} />)}
          </g>
        ) : null}
        <rect
          width={W}
          height={H}
          fill="transparent"
          onMouseMove={(event) => {
            const idx = chartIndexFromMouse(event, base.length);
            const vixPoint = vix[proportionalIndex(idx, base.length, vix.length)];
            const rvPoint = realizedVol[proportionalIndex(idx, base.length, realizedVol.length)];
            const vrpPoint = vrp[proportionalIndex(idx, base.length, vrp.length)];
            setHover({
              x: volScale.x(idx, base.length),
              date: base[idx].date,
              values: [
                { label: "VIX", value: vixPoint.value, color: "var(--crit)", unit: "%" },
                { label: "Realized 21d", value: rvPoint.value, color: "var(--accent)", unit: "%" },
                { label: "VRP", value: vrpPoint.value, color: "var(--good)", unit: " pts" },
              ],
            });
          }}
          onMouseLeave={() => setHover(null)}
        />
      </svg>
      <ValueTag value={lastVix.value} color="var(--crit)" unit="%" yFrac={volScale.y(lastVix.value) / H} offset={offsets[0]} />
      <ValueTag value={lastRv.value} color="var(--accent)" unit="%" yFrac={volScale.y(lastRv.value) / H} offset={offsets[1]} />
      <ValueTag value={lastVrp.value} color="var(--good)" unit=" pts" yFrac={vrpScale.y(lastVrp.value) / H} offset={offsets[2]} />
      <HoverOverlay hover={hover} />
    </div>
  );
}

function Empty() {
  return (
    <div style={{ height: 120, display: "grid", placeItems: "center", color: "var(--ink-faint)", fontSize: 12 }}>
      No history available
    </div>
  );
}
