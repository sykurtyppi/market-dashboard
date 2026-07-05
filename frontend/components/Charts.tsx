import { Point } from "@/lib/api";

const W = 480;
const H = 190;

type Scale = { x: (i: number, n: number) => number; y: (v: number) => number };

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

function GridLines() {
  return (
    <>
      <line x1="0" y1={H * 0.25} x2={W} y2={H * 0.25} stroke="var(--line)" strokeWidth="1" />
      <line x1="0" y1={H * 0.5} x2={W} y2={H * 0.5} stroke="var(--line)" strokeWidth="1" />
      <line x1="0" y1={H * 0.75} x2={W} y2={H * 0.75} stroke="var(--line)" strokeWidth="1" />
    </>
  );
}

export function AreaChart({ points, color }: { points: Point[]; color: string }) {
  if (points.length < 2) return <Empty />;
  const s = makeScale(points.map((p) => p.value));
  const path = linePath(points, s);
  const last = points[points.length - 1];
  const zeroInRange = points.some((p) => p.value > 0) && points.some((p) => p.value < 0);
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden>
      <GridLines />
      {zeroInRange && (
        <line x1="0" y1={s.y(0)} x2={W} y2={s.y(0)} stroke="var(--line-strong)" strokeWidth="1" strokeDasharray="3 3" />
      )}
      <path d={`${path} L${W},${H} L0,${H} Z`} fill={color} fillOpacity="0.1" />
      <path d={path} fill="none" stroke={color} strokeWidth="2" />
      <circle cx={W} cy={s.y(last.value)} r="3.2" fill={color} />
    </svg>
  );
}

export function MultiLineChart({ series }: { series: { points: Point[]; color: string }[] }) {
  const all = series.flatMap((se) => se.points.map((p) => p.value));
  if (all.length < 2) return <Empty />;
  const s = makeScale(all);
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden>
      <GridLines />
      {series.map((se, idx) =>
        se.points.length >= 2 ? (
          <g key={idx}>
            <path d={linePath(se.points, s)} fill="none" stroke={se.color} strokeWidth="2" />
            <circle cx={W} cy={s.y(se.points[se.points.length - 1].value)} r="3.2" fill={se.color} />
          </g>
        ) : null
      )}
    </svg>
  );
}

function Empty() {
  return (
    <div style={{ height: 120, display: "grid", placeItems: "center", color: "var(--ink-faint)", fontSize: 12 }}>
      No history available
    </div>
  );
}
