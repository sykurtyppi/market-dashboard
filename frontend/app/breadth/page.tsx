import { getBreadth, getFreshness, Breadth, Freshness } from "@/lib/api";
import Topbar from "@/components/Topbar";
import { AreaChart } from "@/components/Charts";
import { MetricCard, Section, Panel, fmtAsOf } from "@/components/ui";

export const dynamic = "force-dynamic";

function Content({ data, freshness }: { data: Breadth; freshness: Freshness }) {
  return (
    <>
      <Topbar title="Market Breadth" subtitle="S&P 500 advance-decline internals" freshness={freshness} />
      <div className="content">
        <Section title="Current Breadth" aside={<span className="mono">{fmtAsOf(data.as_of)}</span>}>
          <div className="metrics">
            {data.metrics.map((m) => <MetricCard key={m.key} m={m} />)}
          </div>
        </Section>

        <section className="grid-2">
          <Panel title="Advance–Decline Line" sub="120d">
            <AreaChart points={data.charts.ad_line} color="var(--accent)" label="A/D line" />
            <div className="legend">
              <span><i style={{ background: "var(--accent)" }} />Cumulative A/D</span>
              <span style={{ color: "var(--ink-faint)" }}>Rising = broad participation</span>
            </div>
          </Panel>
          <Panel title="McClellan Oscillator" sub="120d">
            <AreaChart points={data.charts.mcclellan} color="var(--good)" label="McClellan" />
            <div className="legend">
              <span><i style={{ background: "var(--good)" }} />McClellan</span>
              <span style={{ color: "var(--ink-faint)" }}>Above zero = positive breadth momentum</span>
            </div>
          </Panel>
        </section>

        <Section title="Breadth Percentage" aside="120d">
          <Panel title="Advancers as % of active issues">
            <AreaChart points={data.charts.breadth_pct} color="var(--accent)" label="Advancers" unit="%" />
          </Panel>
        </Section>
      </div>
    </>
  );
}

export default async function BreadthPage() {
  try {
    const [data, freshness] = await Promise.all([getBreadth(), getFreshness()]);
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
