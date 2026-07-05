import { getVolatility, getFreshness, Volatility, Freshness } from "@/lib/api";
import Topbar from "@/components/Topbar";
import { AreaChart, MultiLineChart } from "@/components/Charts";
import { MetricCard, Section, Panel } from "@/components/ui";

export const dynamic = "force-dynamic";

function Content({ data, freshness }: { data: Volatility; freshness: Freshness }) {
  return (
    <>
      <Topbar title="Volatility & VRP" subtitle="Volatility risk premium & regime" freshness={freshness} />
      <div className="content">
        <Section title="VRP Regime" aside={<span className="mono">{data.as_of ?? "—"}</span>}>
          <div className="regime" style={{ gridTemplateColumns: "1.4fr repeat(4, 1fr)" }}>
            <div className="regime-cell lead">
              <span className="k">Regime</span>
              <span className="regime-score" style={{ fontSize: 22 }}>{data.regime ?? "—"}</span>
              <span className="note">{data.regime_note}</span>
            </div>
            {data.metrics.map((m) => (
              <div className="regime-cell" key={m.key}>
                <span className="k">{m.label}</span>
                <span className="v mono" style={{ fontSize: 18 }}>
                  <span className={`dot ${m.state}`} />
                  {m.value === null ? "—" : m.unit === "%" ? `${m.value.toFixed(2)}%` : m.value.toFixed(2)}
                </span>
                <span className="note">{m.source}</span>
              </div>
            ))}
          </div>
        </Section>

        <Section title="Key Indicators">
          <div className="metrics">
            {data.metrics.map((m) => <MetricCard key={m.key} m={m} />)}
          </div>
        </Section>

        <section className="grid-2">
          <Panel title="Implied vs Realized Volatility" sub="180d">
            <MultiLineChart series={[
              { points: data.charts.vix, color: "var(--accent)" },
              { points: data.charts.realized_vol, color: "var(--warn)" },
            ]} />
            <div className="legend">
              <span><i style={{ background: "var(--accent)" }} />Implied (VIX)</span>
              <span><i style={{ background: "var(--warn)" }} />Realized (21d)</span>
            </div>
          </Panel>
          <Panel title="Volatility Risk Premium" sub="180d">
            <AreaChart points={data.charts.vrp_history} color="var(--accent)" />
            <div className="legend">
              <span><i style={{ background: "var(--accent)" }} />VRP</span>
              <span style={{ color: "var(--ink-faint)" }}>Positive = implied above realized (premium rich)</span>
            </div>
          </Panel>
        </section>
      </div>
    </>
  );
}

export default async function VolatilityPage() {
  try {
    const [data, freshness] = await Promise.all([getVolatility(), getFreshness()]);
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
