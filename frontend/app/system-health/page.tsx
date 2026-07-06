import { getSystemHealth, getFreshness, SystemHealth, Freshness } from "@/lib/api";
import Topbar from "@/components/Topbar";
import { Section } from "@/components/ui";

export const dynamic = "force-dynamic";

const STATUS_LABEL: Record<string, string> = {
  healthy: "Healthy",
  stale: "Stale",
  degraded: "Degraded",
  down: "Down",
  unknown: "Unknown",
};

function fmtAge(hours: number | null): string {
  if (hours === null || hours === undefined) return "—";
  if (hours < 1) return `${Math.round(hours * 60)}m`;
  if (hours < 48) return `${hours.toFixed(1)}h`;
  return `${(hours / 24).toFixed(1)}d`;
}

function fmtUpdated(iso: string | null): string {
  if (!iso) return "—";
  return iso.replace("T", " ").slice(0, 16);
}

const COUNT_META: { key: string; label: string; state: string }[] = [
  { key: "healthy", label: "Healthy", state: "good" },
  { key: "stale", label: "Stale", state: "warn" },
  { key: "degraded", label: "Degraded", state: "warn" },
  { key: "down", label: "Down", state: "crit" },
  { key: "unknown", label: "Unknown", state: "neutral" },
];

function Content({ data, freshness }: { data: SystemHealth; freshness: Freshness }) {
  return (
    <>
      <Topbar title="System Health" subtitle="Data source status and freshness" freshness={freshness} />
      <div className="content">
        <Section
          title="Overall Status"
          aside={<span className="mono">{data.as_of ? fmtUpdated(data.as_of) : "—"}</span>}
        >
          <div className="regime" style={{ gridTemplateColumns: "1.4fr repeat(5, 1fr)" }}>
            <div className="regime-cell lead">
              <span className="k">System</span>
              <span className="regime-score" style={{ fontSize: 22, display: "flex", alignItems: "center", gap: 10 }}>
                <span className={`dot ${data.overall_state}`} style={{ width: 12, height: 12 }} />
                {STATUS_LABEL[data.overall_status] ?? data.overall_status}
              </span>
              <span className="note">{data.total_sources} sources monitored</span>
            </div>
            {COUNT_META.map((c) => (
              <div className="regime-cell" key={c.key}>
                <span className="k">{c.label}</span>
                <span className="v mono" style={{ fontSize: 20 }}>
                  <span className={`dot ${c.state}`} />{data.summary[c.key] ?? 0}
                </span>
              </div>
            ))}
          </div>
        </Section>

        <Section title="Data Sources">
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Source</th>
                  <th>Status</th>
                  <th>Last update</th>
                  <th className="num">Age</th>
                  <th>Detail</th>
                </tr>
              </thead>
              <tbody>
                {data.sources.map((s) => (
                  <tr key={s.key}>
                    <td>{s.name}</td>
                    <td>
                      <span className="tag"><span className={`dot ${s.state}`} />{STATUS_LABEL[s.status] ?? s.status}</span>
                    </td>
                    <td className="mono src">{fmtUpdated(s.last_update)}</td>
                    <td className="num mono">{fmtAge(s.age_hours)}</td>
                    <td className="src">{s.message || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      </div>
    </>
  );
}

export default async function SystemHealthPage() {
  try {
    const [data, freshness] = await Promise.all([getSystemHealth(), getFreshness()]);
    return <Content data={data} freshness={freshness} />;
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return (
      <div className="content">
        <div className="panel" style={{ maxWidth: 560 }}>
          <div className="panel-head"><span className="t">Cannot reach the API</span></div>
          <p style={{ color: "var(--ink-muted)", fontSize: 13 }}>{message}</p>
        </div>
      </div>
    );
  }
}
