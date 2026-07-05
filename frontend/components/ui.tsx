import { ReactNode } from "react";
import { Metric } from "@/lib/api";

export function fmt(value: number | null, unit: string, key?: string): string {
  if (value === null || value === undefined) return "—";
  let s: string;
  if (key === "fear_greed") s = value.toFixed(1);
  else if (Number.isInteger(value)) s = value.toLocaleString();
  else s = value.toFixed(2);
  if (key === "contango" && value > 0) s = `+${s}`;
  return s;
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
