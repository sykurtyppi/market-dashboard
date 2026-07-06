"use client";

import { useState } from "react";
import type { CSSProperties, MouseEvent } from "react";
import type { Point } from "@/lib/api";

// Single shared vol axis: VIX (implied) and realized vol are the same unit, so
// they share one scale and the gap between them *is* the VRP. Overlays (axis
// labels, tags, tooltip) are inline-styled so the chart renders correctly even
// if a stale dev-server stylesheet is missing the class rules.

const IMPLIED = "var(--accent)";
const REALIZED = "var(--warn)";
const BAND = "var(--accent)";

type Hover = { i: number; xPct: number } | null;

function fmtDateShort(date: string): string {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(date);
  const d = m ? new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3])) : new Date(date);
  return Number.isNaN(d.getTime()) ? date : d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function fmtDateFull(date: string): string {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(date);
  const d = m ? new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3])) : new Date(date);
  return Number.isNaN(d.getTime()) ? date : d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

// "Nice" round axis ticks spanning [min, max].
function niceTicks(min: number, max: number, count: number): number[] {
  const span = max - min || 1;
  const step0 = span / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const norm = step0 / mag;
  const step = (norm >= 5 ? 5 : norm >= 2 ? 2 : norm >= 1 ? 1 : 0.5) * mag;
  const first = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let v = first; v <= max + step * 0.001; v += step) ticks.push(Number(v.toFixed(4)));
  return ticks;
}

interface VolatilityChartProps {
  vix: Point[];
  realizedVol: Point[];
}

export default function VolatilityChart({ vix, realizedVol }: VolatilityChartProps) {
  const [hover, setHover] = useState<Hover>(null);
  const n = Math.min(vix.length, realizedVol.length);
  if (n < 2) {
    return (
      <div style={{ height: 250, display: "grid", placeItems: "center", color: "var(--ink-faint)", fontSize: 12 }}>
        Not enough history in this window
      </div>
    );
  }

  const volAll = [...vix.slice(0, n), ...realizedVol.slice(0, n)].map((p) => p.value);
  const dataMin = Math.min(...volAll);
  const dataMax = Math.max(...volAll);
  const pad = (dataMax - dataMin || 1) * 0.1;
  const min = dataMin - pad;
  const max = dataMax + pad;

  const xi = (i: number) => (n <= 1 ? 0 : (i / (n - 1)) * 100);
  const yv = (v: number) => 100 - ((v - min) / (max - min)) * 100;

  const impliedPts = vix.slice(0, n);
  const realizedPts = realizedVol.slice(0, n);
  const line = (pts: Point[]) => pts.map((p, i) => `${i ? "L" : "M"}${xi(i).toFixed(2)},${yv(p.value).toFixed(2)}`).join(" ");
  const impliedPath = line(impliedPts);
  const realizedPath = line(realizedPts);
  // Band between the two lines = the VRP zone.
  const bandPath = `${impliedPath} L${xi(n - 1).toFixed(2)},${yv(realizedPts[n - 1].value).toFixed(2)} ` +
    realizedPts.slice().reverse().map((p, k) => `L${xi(n - 1 - k).toFixed(2)},${yv(p.value).toFixed(2)}`).join(" ") + " Z";

  const yTicks = niceTicks(dataMin, dataMax, 4);
  const xIdx = Array.from(new Set([0, Math.round((n - 1) * 0.25), Math.round((n - 1) * 0.5), Math.round((n - 1) * 0.75), n - 1]));

  const lastImplied = impliedPts[n - 1].value;
  const lastRealized = realizedPts[n - 1].value;

  const onMove = (e: MouseEvent<SVGRectElement>) => {
    const b = e.currentTarget.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (e.clientX - b.left) / b.width));
    const i = Math.round(ratio * (n - 1));
    setHover({ i, xPct: xi(i) });
  };

  const gutterLabel: CSSProperties = { position: "absolute", fontSize: 10.5, color: "var(--ink-faint)", fontVariantNumeric: "tabular-nums", pointerEvents: "none", whiteSpace: "nowrap" };
  const dotStyle = (leftPct: number, topPct: number, color: string): CSSProperties => ({
    position: "absolute", left: `${leftPct}%`, top: `${topPct}%`, transform: "translate(-50%, -50%)",
    width: 8, height: 8, borderRadius: "50%", background: color, border: "1.5px solid var(--surface)",
    pointerEvents: "none",
  });
  const tag = (color: string): CSSProperties => ({
    position: "absolute", right: 2, transform: "translateY(-50%)", fontSize: 11, fontWeight: 600,
    padding: "1px 5px", borderRadius: 5, background: "var(--surface)", border: "1px solid var(--line)",
    color, fontVariantNumeric: "tabular-nums", whiteSpace: "nowrap", pointerEvents: "none",
    boxShadow: "0 1px 2px rgba(22,24,29,.05)",
  });

  const hi = hover?.i ?? null;
  const hVix = hi !== null ? impliedPts[hi].value : null;
  const hRv = hi !== null ? realizedPts[hi].value : null;
  const hVrp = hVix !== null && hRv !== null ? hVix - hRv : null;
  const tooltipLeft = hover ? Math.max(9, Math.min(78, hover.xPct)) : 0;

  return (
    <div style={{ position: "relative", paddingLeft: 40, paddingBottom: 22, paddingTop: 6 }}>
      <div style={{ position: "relative", height: 250 }}>
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={{ position: "absolute", inset: 0, width: "100%", height: "100%", display: "block", overflow: "visible" }} aria-hidden>
          {yTicks.map((t) => (
            <line key={t} x1="0" y1={yv(t)} x2="100" y2={yv(t)} stroke="var(--line)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
          ))}
          <path d={bandPath} fill={BAND} fillOpacity="0.07" />
          <path d={impliedPath} fill="none" stroke={IMPLIED} strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
          <path d={realizedPath} fill="none" stroke={REALIZED} strokeWidth="1.5" strokeDasharray="4 4" strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
          {hover ? (
            <line x1={hover.xPct} y1="0" x2={hover.xPct} y2="100" stroke="var(--ink-faint)" strokeWidth="1" strokeDasharray="3 3" vectorEffect="non-scaling-stroke" />
          ) : null}
          <rect x="0" y="0" width="100" height="100" fill="transparent" style={{ cursor: "crosshair" }} onMouseMove={onMove} onMouseLeave={() => setHover(null)} />
        </svg>

        {/* Y-axis labels (left gutter) */}
        {yTicks.map((t) => (
          <span key={t} style={{ ...gutterLabel, left: -38, top: `${yv(t)}%`, transform: "translateY(-50%)" }}>{t.toFixed(t < 10 ? 1 : 0)}</span>
        ))}

        {/* X-axis date labels (bottom gutter) */}
        {xIdx.map((i) => (
          <span key={i} style={{ ...gutterLabel, bottom: -20, left: `${xi(i)}%`, transform: `translateX(${i === 0 ? "0" : i === n - 1 ? "-100%" : "-50%"})` }}>
            {fmtDateShort(impliedPts[i].date)}
          </span>
        ))}

        {/* Endpoint value tags */}
        <span style={{ ...tag(IMPLIED), top: `${yv(lastImplied)}%` }}>{lastImplied.toFixed(2)}%</span>
        <span style={{ ...tag(REALIZED), top: `${yv(lastRealized)}%` }}>{lastRealized.toFixed(2)}%</span>

        {/* Round hover dots (HTML so they aren't stretched into ovals by the SVG) */}
        {hover && hVix !== null ? <span style={dotStyle(hover.xPct, yv(hVix), IMPLIED)} /> : null}
        {hover && hRv !== null ? <span style={dotStyle(hover.xPct, yv(hRv), REALIZED)} /> : null}

        {/* Tooltip */}
        {hover && hVix !== null && hRv !== null ? (
          <div style={{ position: "absolute", top: 6, left: `${tooltipLeft}%`, transform: "translateX(-50%)", minWidth: 150, padding: "8px 10px", borderRadius: 8, border: "1px solid var(--line)", background: "rgba(255,255,255,.97)", boxShadow: "0 10px 26px rgba(22,24,29,.12)", pointerEvents: "none", zIndex: 2, fontVariantNumeric: "tabular-nums" }}>
            <div style={{ fontSize: 10.5, color: "var(--ink-faint)", marginBottom: 6 }}>{fmtDateFull(impliedPts[hi as number].date)}</div>
            <Row color={IMPLIED} label="VIX" value={`${hVix.toFixed(2)}%`} />
            <Row color={REALIZED} label="Realized 21d" value={`${hRv.toFixed(2)}%`} />
            <Row color="var(--ink)" label="VRP" value={`${(hVrp as number).toFixed(2)} pts`} strong />
          </div>
        ) : null}
      </div>
    </div>
  );
}

function Row({ color, label, value, strong }: { color: string; label: string; value: string; strong?: boolean }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 14, fontSize: 11.5, color: "var(--ink-muted)", marginTop: 3 }}>
      <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
        <i style={{ width: 8, height: 8, borderRadius: "50%", background: color, display: "inline-block" }} />{label}
      </span>
      <b style={{ color: strong ? "var(--ink)" : "var(--ink)", fontWeight: strong ? 700 : 600 }}>{value}</b>
    </div>
  );
}
