import type { Point } from "@/lib/api";

// Data-honesty guard for history charts: a chart draws a smooth line straight
// across a collection outage, which reads as "nothing happened" when the truth
// is "we weren't looking". Server-safe (no hooks) so any page can render it
// under a chart panel. Inline-styled to survive a stale dev-server stylesheet.

const DAY_MS = 86_400_000;

// Largest gap (in days) between consecutive points. UTC-parse offsets cancel
// in differences, so plain Date parsing is safe here.
export function maxGapDays(points: Point[]): number {
  let max = 0;
  for (let i = 1; i < points.length; i += 1) {
    const prev = new Date(points[i - 1].date).getTime();
    const cur = new Date(points[i].date).getTime();
    if (Number.isFinite(prev) && Number.isFinite(cur)) {
      max = Math.max(max, (cur - prev) / DAY_MS);
    }
  }
  return max;
}

export default function GapNotice({ points, thresholdDays = 30 }: { points: Point[]; thresholdDays?: number }) {
  const gap = maxGapDays(points);
  if (gap <= thresholdDays) return null;
  return (
    <div style={{ marginTop: 10, padding: "7px 11px", borderRadius: 7, border: "1px solid var(--warn)", background: "var(--warn-soft)", color: "var(--warn)", fontSize: 11.5, display: "inline-flex", alignItems: "center", gap: 7 }}>
      <span style={{ width: 7, height: 7, borderRadius: "50%", background: "var(--warn)", flex: "none" }} />
      Series has a {Math.round(gap)}-day gap — the line interpolates across missing data.
    </div>
  );
}
