import { getBreadth, getFreshness, Breadth, Freshness } from "@/lib/api";
import Topbar from "@/components/Topbar";
import Explainer from "@/components/Explainer";
import GapNotice from "@/components/GapNotice";
import { AreaChart } from "@/components/Charts";
import { MetricCard, Section, Panel, fmtAsOf } from "@/components/ui";

export const dynamic = "force-dynamic";

function Content({ data, freshness }: { data: Breadth; freshness: Freshness }) {
  return (
    <>
      <Topbar title="Market Breadth" subtitle="S&P 500 advance-decline internals" freshness={freshness} />
      <div className="content">
        {data.warnings.length > 0 ? (
          <div className="notice"><span className="dot warn" />{data.warnings.join(" ")}</div>
        ) : null}

        <Section title="Current Breadth" aside={<span className="mono">{fmtAsOf(data.as_of)}</span>}>
          <div className="metrics">
            {data.metrics.map((m) => <MetricCard key={m.key} m={m} />)}
          </div>
        </Section>

        <section className="grid-2">
          <Panel title="Advance–Decline Line" sub="120d">
            <AreaChart points={data.charts.ad_line} color="var(--accent)" label="A/D line" />
            <GapNotice points={data.charts.ad_line} />
            <div className="legend">
              <span><i style={{ background: "var(--accent)" }} />Cumulative A/D</span>
              <span style={{ color: "var(--ink-faint)" }}>Rising = broad participation</span>
            </div>
          </Panel>
          <Panel title="McClellan Oscillator" sub="120d">
            <AreaChart points={data.charts.mcclellan} color="var(--good)" label="McClellan" />
            <GapNotice points={data.charts.mcclellan} />
            <div className="legend">
              <span><i style={{ background: "var(--good)" }} />McClellan</span>
              <span style={{ color: "var(--ink-faint)" }}>Above zero = positive breadth momentum</span>
            </div>
          </Panel>
        </section>

        <Section title="Breadth Percentage" aside="120d">
          <Panel title="Advancers as % of active issues">
            <AreaChart points={data.charts.breadth_pct} color="var(--accent)" label="Advancers" unit="%" />
            <GapNotice points={data.charts.breadth_pct} />
          </Panel>
        </Section>
        <Explainer
          title="How to read this — market breadth"
          intro="Breadth measures how many stocks participate in a move — a rally on narrow leadership is more fragile than one with broad participation."
          points={[
            { label: "Breadth %:", text: "share of sampled stocks advancing on the day. Above ~55% is healthy participation; below ~45% is weak." },
            { label: "A/D line:", text: "cumulative advancers minus decliners. Rising with the index confirms the trend; falling while the index rises is a divergence that often precedes weakness." },
            { label: "McClellan oscillator:", text: "EMA(19) − EMA(39) of net advances. Above zero = improving momentum; below −50 = washout conditions." },
          ]}
          caveat="Computed from a ~100-stock S&P 500 sample. Not investment advice."
        />
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
