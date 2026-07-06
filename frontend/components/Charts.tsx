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

// Format the current-value tag: thousands get grouped, everything else keeps
// two decimals so small spreads/ratios read precisely.
function fmtTag(v: number, unit: string): string {
  const n = Math.abs(v) >= 1000 ? Math.round(v).toLocaleString() : v.toFixed(2);
  return unit ? `${n}${unit}` : n;
}

// A crisp HTML label pinned to a line's latest point. Rendered outside the SVG
// so it stays at true pixel size regardless of how the chart is scaled.
function ValueTag({ value, color, unit, yFrac }: { value: number; color: string; unit: string; yFrac: number }) {
  return (
    <span
      className="chart-tag mono"
      style={{ top: `${(yFrac * 100).toFixed(2)}%`, color }}
    >
      {fmtTag(value, unit)}
    </span>
  );
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

export function AreaChart({ points, color, unit = "" }: { points: Point[]; color: string; unit?: string }) {
  if (points.length < 2) return <Empty />;
  const s = makeScale(points.map((p) => p.value));
  const path = linePath(points, s);
  const last = points[points.length - 1];
  const zeroInRange = points.some((p) => p.value > 0) && points.some((p) => p.value < 0);
  return (
    <div className="chart-wrap">
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden>
        <GridLines />
        {zeroInRange && (
          <line x1="0" y1={s.y(0)} x2={W} y2={s.y(0)} stroke="var(--line-strong)" strokeWidth="1" strokeDasharray="3 3" />
        )}
        <path d={`${path} L${W},${H} L0,${H} Z`} fill={color} fillOpacity="0.1" />
        <path d={path} fill="none" stroke={color} strokeWidth="2" />
        <circle cx={W} cy={s.y(last.value)} r="3.2" fill={color} />
      </svg>
      <ValueTag value={last.value} color={color} unit={unit} yFrac={s.y(last.value) / H} />
    </div>
  );
}

export function MultiLineChart({ series, unit = "" }: { series: { points: Point[]; color: string }[]; unit?: string }) {
  const all = series.flatMap((se) => se.points.map((p) => p.value));
  if (all.length < 2) return <Empty />;
  const s = makeScale(all);
  return (
    <div className="chart-wrap">
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
      {series.map((se, idx) =>
        se.points.length >= 2 ? (
          <ValueTag
            key={idx}
            value={se.points[se.points.length - 1].value}
            color={se.color}
            unit={unit}
            yFrac={s.y(se.points[se.points.length - 1].value) / H}
          />
        ) : null
      )}
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
