import { getVolatility, getFreshness, Volatility, Freshness } from "@/lib/api";
import Topbar from "@/components/Topbar";
import { VRPCompositeChart } from "@/components/Charts";
import { MetricCard, Section, Panel } from "@/components/ui";

export const dynamic = "force-dynamic";

function stat(value: number | null, unit = "") {
  if (value === null || value === undefined) return "—";
  return `${value.toFixed(2)}${unit}`;
}

function VrpStat({ label, value, unit, note }: { label: string; value: number | null; unit?: string; note: string }) {
  return (
    <div className="stat-card">
      <div className="k">{label}</div>
      <div className="v mono">{stat(value, unit)}</div>
      <div className="note">{note}</div>
    </div>
  );
}

function Content({ data, freshness }: { data: Volatility; freshness: Freshness }) {
  const stats = data.stats ?? {
    avg_vrp: null,
    std_dev: null,
    current_percentile: null,
    max_vrp: null,
    min_vrp: null,
    observations: 0,
  };

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

        <Panel title="VIX vs Realized Vol & VRP Spread" sub={`${stats.observations || 180} observations · hover for history`}>
          <VRPCompositeChart
            vix={data.charts.vix}
            realizedVol={data.charts.realized_vol}
            vrp={data.charts.vrp_history}
          />
          <div className="legend">
            <span><i style={{ background: "var(--crit)" }} />VIX implied vol</span>
            <span><i style={{ background: "var(--accent)" }} />Realized vol 21d</span>
            <span><i style={{ background: "var(--good)" }} />VRP spread</span>
            <span style={{ color: "var(--ink-faint)" }}>Zero line = implied equals realized</span>
          </div>
          <div className="stat-grid">
            <VrpStat label="Avg VRP" value={stats.avg_vrp} note="Mean over visible history" />
            <VrpStat label="VRP Std Dev" value={stats.std_dev} note="Dispersion of the spread" />
            <VrpStat label="Current Percentile" value={stats.current_percentile} unit="%" note="Low = realized vol is rich" />
            <VrpStat label="Max / Min VRP" value={stats.max_vrp} note={`Min ${stat(stats.min_vrp)}`} />
          </div>
        </Panel>
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
