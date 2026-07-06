import { getTreasuryStress, getFreshness, TreasuryStress, Freshness } from "@/lib/api";
import Topbar from "@/components/Topbar";
import { AreaChart } from "@/components/Charts";
import { MetricCard, Section, Panel } from "@/components/ui";

export const dynamic = "force-dynamic";

function Content({ data, freshness }: { data: TreasuryStress; freshness: Freshness }) {
  const state = data.state;
  return (
    <>
      <Topbar title="Treasury Stress" subtitle="MOVE index — Treasury market volatility" freshness={freshness} />
      <div className="content">
        <Section title="Stress Regime" aside={<span className="mono">{data.as_of ?? "—"}</span>}>
          <div className="regime" style={{ gridTemplateColumns: "1.2fr 2fr" }}>
            <div className="regime-cell lead">
              <span className="k">Regime</span>
              <span className="v" style={{ fontSize: 20, marginTop: 4 }}>
                <span className={`dot ${state}`} />{data.regime ?? "—"}
              </span>
              <span className="note">{data.regime_note}</span>
            </div>
            <div className="regime-cell" style={{ display: "flex", gap: 24, alignItems: "center" }}>
              {data.metrics.map((m) => (
                <div key={m.key}>
                  <span className="k">{m.label}</span>
                  <div className="v mono" style={{ fontSize: 22, marginTop: 6 }}>
                    {m.value === null ? "—" : m.unit === "%" ? `${m.value.toFixed(1)}%` : m.value.toFixed(1)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </Section>

        <Section title="Key Indicators">
          <div className="metrics">
            {data.metrics.map((m) => <MetricCard key={m.key} m={m} />)}
          </div>
        </Section>

        <section className="grid-2">
          <Panel title="MOVE Index" sub="1Y">
            <AreaChart points={data.charts.move_history} color="var(--accent)" label="MOVE" />
            <div className="legend">
              <span><i style={{ background: "var(--accent)" }} />MOVE</span>
              <span style={{ color: "var(--ink-faint)" }}>&lt;80 calm · 80–120 normal · 120–150 elevated · &gt;150 stress</span>
            </div>
          </Panel>
          <Panel title="Historical Percentile" sub="1Y">
            <AreaChart points={data.charts.percentile_history} color="var(--warn)" label="Percentile" unit="%" />
            <div className="legend">
              <span><i style={{ background: "var(--warn)" }} />Percentile</span>
            </div>
          </Panel>
        </section>
      </div>
    </>
  );
}

export default async function TreasuryPage() {
  try {
    const [data, freshness] = await Promise.all([getTreasuryStress(), getFreshness()]);
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
