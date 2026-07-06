"use client";

import type { CSSProperties } from "react";
import type { VixTenor } from "@/lib/api";

// VIX term structure: a small curve across the tenors (9D / 30D / 3M) with a
// labelled point at each maturity and real axes — the shape shows contango
// (upward) or backwardation (downward). Tenors are spaced evenly by index; the
// x-axis labels carry the actual maturities. Inline-styled to survive a stale
// dev-server stylesheet.

const LINE = "var(--accent)";

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

export default function VixTermChart({ term }: { term: VixTenor[] }) {
  const pts = term.filter((t) => typeof t.value === "number");
  if (pts.length < 2) {
    return (
      <div style={{ height: 220, display: "grid", placeItems: "center", color: "var(--ink-faint)", fontSize: 12 }}>
        Term structure unavailable
      </div>
    );
  }

  const n = pts.length;
  const values = pts.map((t) => t.value);
  const dMin = Math.min(...values), dMax = Math.max(...values);
  const pad = (dMax - dMin || 1) * 0.25;
  const min = dMin - pad, max = dMax + pad;

  const xi = (i: number) => (n <= 1 ? 50 : (i / (n - 1)) * 100);
  const yv = (v: number) => 100 - ((v - min) / (max - min)) * 100;
  const linePath = pts.map((t, i) => `${i ? "L" : "M"}${xi(i).toFixed(2)},${yv(t.value).toFixed(2)}`).join(" ");
  const areaPath = `${linePath} L${xi(n - 1).toFixed(2)},100 L0,100 Z`;
  const ticks = niceTicks(dMin, dMax, 3);

  const yLabel: CSSProperties = { position: "absolute", left: -34, fontSize: 10.5, color: "var(--ink-faint)", fontVariantNumeric: "tabular-nums", pointerEvents: "none", transform: "translateY(-50%)", whiteSpace: "nowrap" };
  const valueLabel: CSSProperties = { position: "absolute", fontSize: 12, fontWeight: 600, color: "var(--ink)", fontVariantNumeric: "tabular-nums", pointerEvents: "none", whiteSpace: "nowrap", transform: "translate(-50%, -100%)" };
  const tenorLabel: CSSProperties = { position: "absolute", bottom: -20, fontSize: 11, color: "var(--ink-muted)", fontWeight: 550, pointerEvents: "none", whiteSpace: "nowrap", transform: "translateX(-50%)" };

  return (
    <div style={{ position: "relative", paddingLeft: 36, paddingBottom: 22, paddingTop: 20 }}>
      <div style={{ position: "relative", height: 200 }}>
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={{ position: "absolute", inset: 0, width: "100%", height: "100%", display: "block", overflow: "visible" }} aria-hidden>
          {ticks.map((t) => (
            <line key={t} x1="0" y1={yv(t)} x2="100" y2={yv(t)} stroke="var(--line)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
          ))}
          <path d={areaPath} fill={LINE} fillOpacity="0.07" />
          <path d={linePath} fill="none" stroke={LINE} strokeWidth="1.75" strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
        </svg>

        {/* Y-axis ticks */}
        {ticks.map((t) => (
          <span key={t} style={{ ...yLabel, top: `${yv(t)}%` }}>{t.toFixed(t < 100 ? 1 : 0)}</span>
        ))}

        {/* Point dots + value labels */}
        {pts.map((t, i) => (
          <span key={`d${t.maturity}`} style={{ position: "absolute", left: `${xi(i)}%`, top: `${yv(t.value)}%`, transform: "translate(-50%, -50%)", width: 9, height: 9, borderRadius: "50%", background: LINE, border: "2px solid var(--surface)", boxShadow: "0 0 0 1px var(--line)", pointerEvents: "none" }} />
        ))}
        {pts.map((t, i) => (
          <span key={`v${t.maturity}`} style={{ ...valueLabel, left: `${xi(i)}%`, top: `calc(${yv(t.value)}% - 9px)` }}>{t.value.toFixed(2)}</span>
        ))}

        {/* X-axis tenor labels */}
        {pts.map((t, i) => (
          <span key={`t${t.maturity}`} style={{ ...tenorLabel, left: `${xi(i)}%` }}>{t.maturity}</span>
        ))}
      </div>
    </div>
  );
}
