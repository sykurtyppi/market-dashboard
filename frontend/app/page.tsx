import { getOverview, RegimeComponent, DetailRow, Overview } from "@/lib/api";
import Topbar from "@/components/Topbar";
import { AreaChart, MultiLineChart } from "@/components/Charts";
import { MetricCard, Section, Panel, fmtAsOf } from "@/components/ui";

export const dynamic = "force-dynamic";

function stateLabel(state: string): string {
  switch (state) {
    case "good": return "Supportive";
    case "warn": return "Elevated";
    case "crit": return "Stress";
    default: return "Neutral";
  }
}

function RegimeCell({ c }: { c: RegimeComponent }) {
  return (
    <div className="regime-cell">
      <span className="k">{c.label}</span>
      <span className="v"><span className={`dot ${c.state}`} />{c.value}</span>
      <span className="note">{c.note}</span>
    </div>
  );
}

function DetailTableRow({ r }: { r: DetailRow }) {
  const val = r.value === null ? "—" : r.unit === "%" ? `${r.value.toFixed(1)}%` : r.value.toFixed(2);
  return (
    <tr>
      <td>{r.indicator}</td>
      <td className="num mono">{val}</td>
      <td><span className="tag"><span className={`dot ${r.state}`} />{stateLabel(r.state)}</span></td>
      <td className="src">{r.source}</td>
    </tr>
  );
}

function OverviewContent({ data }: { data: Overview }) {
  const composite = data.regime.composite_risk;
  return (
    <>
      <Topbar title="Overview" subtitle="Market regime & key risk indicators" freshness={data.freshness} />
      <div className="content">
        <Section title="Market Regime" aside={<span className="mono">{fmtAsOf(data.as_of)}</span>}>
          <div className="regime">
            <div className="regime-cell lead">
              <span className="k">Composite Risk</span>
              <span className="regime-score mono">{composite ?? "—"}<span className="u"> / 100</span></span>
              <div className="regime-scale">
                {composite !== null ? <i style={{ left: `${Math.min(Math.max(composite, 0), 100)}%` }} /> : null}
              </div>
              <span className="note">
                {data.left_signal ? `LEFT signal: ${data.left_signal}` : "Aggregated risk indicators"}
              </span>
            </div>
            {data.regime.components.map((c) => <RegimeCell key={c.key} c={c} />)}
          </div>
        </Section>

        <Section title="Key Indicators" aside="delayed · sourced">
          <div className="metrics">
            {data.metrics.map((m) => <MetricCard key={m.key} m={m} />)}
          </div>
        </Section>

        <section className="grid-2">
          <Panel title="Volatility Risk Premium" sub="180d">
            <AreaChart points={data.charts.vrp_history} color="var(--accent)" label="VRP" />
            <div className="legend">
              <span><i style={{ background: "var(--accent)" }} />VRP</span>
              <span style={{ color: "var(--ink-faint)" }}>Zero line = fair value · negative = rich realized vol</span>
            </div>
          </Panel>
          <Panel title="Credit Spreads" sub="1Y · %">
            <MultiLineChart series={[
              { points: data.charts.credit_spreads.hy, color: "var(--crit)", label: "HY spread" },
              { points: data.charts.credit_spreads.ig, color: "var(--accent)", label: "IG spread" },
            ]} />
            <div className="legend">
              <span><i style={{ background: "var(--crit)" }} />HY (BAMLH0A0HYM2)</span>
              <span><i style={{ background: "var(--accent)" }} />IG (BAMLC0A0CM)</span>
            </div>
          </Panel>
        </section>

        <Section title="Options & Volatility Detail">
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th>Indicator</th><th className="num">Value</th><th>State</th><th>Source</th></tr>
              </thead>
              <tbody>
                {data.detail.map((r) => <DetailTableRow key={r.indicator} r={r} />)}
              </tbody>
            </table>
          </div>
        </Section>
      </div>
    </>
  );
}

function ApiError({ message }: { message: string }) {
  return (
    <div className="content">
      <div className="panel" style={{ maxWidth: 560 }}>
        <div className="panel-head"><span className="t">Cannot reach the API</span></div>
        <p style={{ color: "var(--ink-muted)", fontSize: 13 }}>{message}</p>
        <p style={{ color: "var(--ink-faint)", fontSize: 12.5, marginTop: 10 }}>
          Start the backend: <span className="mono">uvicorn api.main:app --port 8000</span>
        </p>
      </div>
    </div>
  );
}

export default async function OverviewPage() {
  try {
    const data = await getOverview();
    return <OverviewContent data={data} />;
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return <ApiError message={message} />;
  }
}
