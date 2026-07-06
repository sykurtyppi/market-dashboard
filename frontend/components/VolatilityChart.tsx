"use client";

import { useState } from "react";
import type { CSSProperties, MouseEvent } from "react";
import type { Point } from "@/lib/api";

// Two stacked plots on a shared x-axis:
//  - Top: VIX (implied) vs 21d realized vol on one shared vol axis; the shaded
//    gap between them is the VRP.
//  - Bottom: the VRP spread itself, on its own zero-based axis, so the sign and
//    magnitude read directly (no dual-axis-on-one-plot confusion).
// Overlays are inline-styled so the chart survives a stale dev-server stylesheet.

const IMPLIED = "var(--accent)";
const REALIZED = "var(--warn)";
const VRP_COLOR = "var(--good)";
const MAIN_H = 196;
const SUB_H = 78;

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

function niceTicks(min: number, max: number, count: number): number[] {
  const span = max - min || 1;
  const step0 = span / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const norm = step0 / mag;
  const step = (norm >= 5 ? 5 : norm >= 2 ? 2 : norm >= 1 ? 1 : 0.5) * mag;
  const first = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let v = first; v <= max + step * 0.001; v += step) ticks.push(Number(v.toFixed(4)));
  // Flat data (min == max) can yield zero ticks; fall back to the value itself
  // so the axis still shows a gridline/label rather than rendering bare.
  return ticks.length ? ticks : [Number(min.toFixed(2))];
}

interface VolatilityChartProps {
  vix: Point[];
  realizedVol: Point[];
  vrp: Point[];
}

export default function VolatilityChart({ vix, realizedVol, vrp }: VolatilityChartProps) {
  const [hover, setHover] = useState<Hover>(null);
  const n = Math.min(vix.length, realizedVol.length, vrp.length);
  if (n < 2) {
    return (
      <div style={{ height: MAIN_H + SUB_H, display: "grid", placeItems: "center", color: "var(--ink-faint)", fontSize: 12 }}>
        Not enough history in this window
      </div>
    );
  }

  const vixPts = vix.slice(0, n);
  const rvPts = realizedVol.slice(0, n);
  const vrpPts = vrp.slice(0, n);
  const xi = (i: number) => (n <= 1 ? 0 : (i / (n - 1)) * 100);

  // --- vol scale (shared, top plot) ---
  const volAll = [...vixPts, ...rvPts].map((p) => p.value);
  const volMinD = Math.min(...volAll), volMaxD = Math.max(...volAll);
  const volPad = (volMaxD - volMinD || 1) * 0.1;
  const vMin = volMinD - volPad, vMax = volMaxD + volPad;
  const yVol = (v: number) => 100 - ((v - vMin) / (vMax - vMin)) * 100;

  // --- VRP scale (zero-anchored, bottom plot) ---
  const vrpVals = vrpPts.map((p) => p.value);
  const vrpMinD = Math.min(0, ...vrpVals), vrpMaxD = Math.max(0, ...vrpVals);
  const vrpPad = (vrpMaxD - vrpMinD || 1) * 0.12;
  const rMin = vrpMinD - vrpPad, rMax = vrpMaxD + vrpPad;
  const yVrp = (v: number) => 100 - ((v - rMin) / (rMax - rMin)) * 100;

  const line = (pts: Point[], y: (v: number) => number) =>
    pts.map((p, i) => `${i ? "L" : "M"}${xi(i).toFixed(2)},${y(p.value).toFixed(2)}`).join(" ");
  const vixPath = line(vixPts, yVol);
  const rvPath = line(rvPts, yVol);
  const vrpPath = line(vrpPts, yVrp);
  const bandPath = `${vixPath} L${xi(n - 1).toFixed(2)},${yVol(rvPts[n - 1].value).toFixed(2)} ` +
    rvPts.slice().reverse().map((p, k) => `L${xi(n - 1 - k).toFixed(2)},${yVol(p.value).toFixed(2)}`).join(" ") + " Z";
  const zeroY = yVrp(0);
  const vrpArea = `${vrpPath} L${xi(n - 1).toFixed(2)},${zeroY.toFixed(2)} L0,${zeroY.toFixed(2)} Z`;

  const volTicks = niceTicks(volMinD, volMaxD, 4);
  const vrpTicks = niceTicks(vrpMinD, vrpMaxD, 2);
  const xIdx = Array.from(new Set([0, Math.round((n - 1) * 0.25), Math.round((n - 1) * 0.5), Math.round((n - 1) * 0.75), n - 1]));

  const lastVix = vixPts[n - 1].value, lastRv = rvPts[n - 1].value, lastVrp = vrpPts[n - 1].value;

  const onMove = (e: MouseEvent<HTMLDivElement>) => {
    const b = e.currentTarget.getBoundingClientRect();
    if (b.width <= 0) return;  // zero-width (hidden/collapsed) → NaN ratio
    const ratio = Math.max(0, Math.min(1, (e.clientX - b.left) / b.width));
    const i = Math.round(ratio * (n - 1));
    setHover({ i, xPct: xi(i) });
  };

  // Clamp the stored hover index against the *current* series length. Clicking a
  // range button shrinks the arrays without a mousemove, so a stale hover.i can
  // exceed n — deref'ing vixPts[hi] would otherwise crash the tooltip render.
  const hi = hover && Number.isInteger(hover.i) && hover.i >= 0 && hover.i < n ? hover.i : null;
  const hVix = hi !== null ? vixPts[hi].value : null;
  const hRv = hi !== null ? rvPts[hi].value : null;
  const hVrp = hi !== null ? vrpPts[hi].value : null;
  const tooltipLeft = hover ? Math.max(9, Math.min(78, hover.xPct)) : 0;

  const svgStyle: CSSProperties = { position: "absolute", inset: 0, width: "100%", height: "100%", display: "block", overflow: "visible" };
  const yLabel = (topPct: number, text: string, key: string) => (
    <span key={key} style={{ position: "absolute", left: -38, top: `${topPct}%`, transform: "translateY(-50%)", fontSize: 10.5, color: "var(--ink-faint)", fontVariantNumeric: "tabular-nums", pointerEvents: "none", whiteSpace: "nowrap" }}>{text}</span>
  );
  const dot = (leftPct: number, topPct: number, color: string): CSSProperties => ({
    position: "absolute", left: `${leftPct}%`, top: `${topPct}%`, transform: "translate(-50%, -50%)",
    width: 8, height: 8, borderRadius: "50%", background: color, border: "1.5px solid var(--surface)", pointerEvents: "none",
  });
  const tag = (color: string, topPct: number): CSSProperties => ({
    position: "absolute", right: 2, top: `${topPct}%`, transform: "translateY(-50%)", fontSize: 11, fontWeight: 600,
    padding: "1px 5px", borderRadius: 5, background: "var(--surface)", border: "1px solid var(--line)",
    color, fontVariantNumeric: "tabular-nums", whiteSpace: "nowrap", pointerEvents: "none", boxShadow: "0 1px 2px rgba(22,24,29,.05)",
  });

  const summary = `VIX vs realized volatility with VRP spread, ${n} points. ` +
    `Latest: VIX ${lastVix.toFixed(2)} percent, realized ${lastRv.toFixed(2)} percent, VRP ${lastVrp.toFixed(2)} points.`;

  return (
    <div role="img" aria-label={summary} style={{ position: "relative", paddingLeft: 40, paddingBottom: 22, paddingTop: 6 }}>
      <div style={{ position: "relative" }}>
        {/* ---- Top plot: VIX vs realized ---- */}
        <div style={{ position: "relative", height: MAIN_H }}>
          <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={svgStyle} aria-hidden>
            {volTicks.map((t) => (
              <line key={t} x1="0" y1={yVol(t)} x2="100" y2={yVol(t)} stroke="var(--line)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
            ))}
            <path d={bandPath} fill={VRP_COLOR} fillOpacity="0.08" />
            <path d={vixPath} fill="none" stroke={IMPLIED} strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
            <path d={rvPath} fill="none" stroke={REALIZED} strokeWidth="1.5" strokeDasharray="4 4" strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
            {hover ? <line x1={hover.xPct} y1="0" x2={hover.xPct} y2="100" stroke="var(--ink-faint)" strokeWidth="1" strokeDasharray="3 3" vectorEffect="non-scaling-stroke" /> : null}
          </svg>
          {volTicks.map((t) => yLabel(yVol(t), t.toFixed(t < 10 ? 1 : 0), `v${t}`))}
          <span style={tag(IMPLIED, yVol(lastVix))}>{lastVix.toFixed(2)}%</span>
          <span style={tag(REALIZED, yVol(lastRv))}>{lastRv.toFixed(2)}%</span>
          {hover && hVix !== null ? <span style={dot(hover.xPct, yVol(hVix), IMPLIED)} /> : null}
          {hover && hRv !== null ? <span style={dot(hover.xPct, yVol(hRv), REALIZED)} /> : null}
        </div>

        {/* ---- Bottom plot: VRP spread ---- */}
        <div style={{ position: "relative", height: SUB_H, marginTop: 10 }}>
          <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={svgStyle} aria-hidden>
            <line x1="0" y1={zeroY} x2="100" y2={zeroY} stroke="var(--line-strong)" strokeWidth="1" strokeDasharray="3 3" vectorEffect="non-scaling-stroke" />
            <path d={vrpArea} fill={VRP_COLOR} fillOpacity="0.12" />
            <path d={vrpPath} fill="none" stroke={VRP_COLOR} strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
            {hover ? <line x1={hover.xPct} y1="0" x2={hover.xPct} y2="100" stroke="var(--ink-faint)" strokeWidth="1" strokeDasharray="3 3" vectorEffect="non-scaling-stroke" /> : null}
          </svg>
          {vrpTicks.map((t) => yLabel(yVrp(t), t.toFixed(t < 10 && t > -10 ? 1 : 0), `r${t}`))}
          <span style={{ position: "absolute", left: 4, top: 2, fontSize: 10, fontWeight: 650, letterSpacing: ".05em", color: "var(--ink-faint)", textTransform: "uppercase", pointerEvents: "none" }}>VRP spread</span>
          <span style={tag(VRP_COLOR, yVrp(lastVrp))}>{lastVrp.toFixed(2)}</span>
          {hover && hVrp !== null ? <span style={dot(hover.xPct, yVrp(hVrp), VRP_COLOR)} /> : null}
        </div>

        {/* Shared x-axis labels */}
        {xIdx.map((i) => (
          <span key={i} style={{ position: "absolute", bottom: -20, left: `${xi(i)}%`, transform: `translateX(${i === 0 ? "0" : i === n - 1 ? "-100%" : "-50%"})`, fontSize: 10.5, color: "var(--ink-faint)", fontVariantNumeric: "tabular-nums", pointerEvents: "none", whiteSpace: "nowrap" }}>
            {fmtDateShort(vixPts[i].date)}
          </span>
        ))}

        {/* Single hover-capture layer over both plots */}
        <div style={{ position: "absolute", inset: 0, cursor: "crosshair" }} onMouseMove={onMove} onMouseLeave={() => setHover(null)} />

        {/* Tooltip */}
        {hover && hVix !== null && hRv !== null && hVrp !== null ? (
          <div style={{ position: "absolute", top: 6, left: `${tooltipLeft}%`, transform: "translateX(-50%)", minWidth: 150, padding: "8px 10px", borderRadius: 8, border: "1px solid var(--line)", background: "rgba(255,255,255,.97)", boxShadow: "0 10px 26px rgba(22,24,29,.12)", pointerEvents: "none", zIndex: 2, fontVariantNumeric: "tabular-nums" }}>
            <div style={{ fontSize: 10.5, color: "var(--ink-faint)", marginBottom: 6 }}>{fmtDateFull(vixPts[hi as number].date)}</div>
            <Row color={IMPLIED} label="VIX" value={`${hVix.toFixed(2)}%`} />
            <Row color={REALIZED} label="Realized 21d" value={`${hRv.toFixed(2)}%`} />
            <Row color={VRP_COLOR} label="VRP" value={`${hVrp.toFixed(2)} pts`} strong />
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
      <b style={{ color: "var(--ink)", fontWeight: strong ? 700 : 600 }}>{value}</b>
    </div>
  );
}
