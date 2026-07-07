import { ReactNode } from "react";
import { Metric } from "@/lib/api";

export function fmt(value: number | null, unit: string, key?: string): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  let s: string;
  if (key === "fear_greed") s = value.toFixed(1);
  // Percent metrics always get two decimals, integer or not — "4%" next to
  // "3.63%" reads as a different kind of number than it is.
  else if (Number.isInteger(value) && unit !== "%") s = value.toLocaleString();
  else s = value.toFixed(2);
  if (key === "contango" && value > 0) s = `+${s}`;
  return s;
}

// Section-header timestamps: "2026-07-01 00:00:00" is collector noise — a
// midnight time means the source is daily, so show just the date; a real
// intraday time is meaningful, so keep it to the minute.
export function fmtAsOf(asOf: string | null | undefined): string {
  if (!asOf) return "—";
  const s = String(asOf);
  const m = /^(\d{4}-\d{2}-\d{2})[T ](\d{2}):(\d{2})/.exec(s);
  if (!m) return s;
  return m[2] === "00" && m[3] === "00" ? m[1] : `${m[1]} ${m[2]}:${m[3]}`;
}

export function MetricCard({ m }: { m: Metric }) {
  return (
    <div className="card">
      <div className="k">{m.label}</div>
      <div className="v mono">
        {fmt(m.value, m.unit, m.key)}
        {m.unit && m.value !== null ? <span className="u">{m.unit}</span> : null}
      </div>
      <div className="caption"><span className={`dot ${m.state}`} />{m.source}</div>
    </div>
  );
}

export function Section({
  title,
  aside,
  children,
}: {
  title: string;
  aside?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section>
      <div className="section-head">
        <h2>{title}</h2>
        {aside ? <span className="aside">{aside}</span> : null}
      </div>
      {children}
    </section>
  );
}

export function Panel({
  title,
  sub,
  children,
}: {
  title: string;
  sub?: string;
  children: ReactNode;
}) {
  return (
    <div className="panel">
      <div className="panel-head">
        <span className="t">{title}</span>
        {sub ? <span className="s mono">{sub}</span> : null}
      </div>
      {children}
    </div>
  );
}
