import { getCreditLiquidity, getFreshness, CreditLiquidity, Freshness } from "@/lib/api";
import Topbar from "@/components/Topbar";
import { AreaChart, MultiLineChart } from "@/components/Charts";
import { MetricCard, Section, Panel, fmtAsOf } from "@/components/ui";

export const dynamic = "force-dynamic";

function Content({ data, freshness }: { data: CreditLiquidity; freshness: Freshness }) {
  return (
    <>
      <Topbar title="Credit & Liquidity" subtitle="Credit spreads & Fed balance sheet" freshness={freshness} />
      <div className="content">
        <Section title="Current Levels" aside={<span className="mono">{fmtAsOf(data.as_of)}</span>}>
          <div className="metrics">
            {data.metrics.map((m) => <MetricCard key={m.key} m={m} />)}
          </div>
        </Section>

        <section className="grid-2">
          <Panel title="Credit Spreads" sub="1Y · %">
            <MultiLineChart unit="%" series={[
              { points: data.charts.credit_spreads.hy, color: "var(--crit)", label: "HY spread" },
              { points: data.charts.credit_spreads.ig, color: "var(--accent)", label: "IG spread" },
            ]} />
            <div className="legend">
              <span><i style={{ background: "var(--crit)" }} />HY (BAMLH0A0HYM2)</span>
              <span><i style={{ background: "var(--accent)" }} />IG (BAMLC0A0CM)</span>
            </div>
          </Panel>
          <Panel title="Fed Total Assets" sub="$T">
            <AreaChart points={data.charts.fed_assets} color="var(--accent)" label="Fed assets" />
            <div className="legend">
              <span><i style={{ background: "var(--accent)" }} />Balance sheet ($T)</span>
              <span style={{ color: "var(--ink-faint)" }}>Falling = quantitative tightening</span>
            </div>
          </Panel>
        </section>

        <Section title="Quantitative Tightening" aside="cumulative $B">
          <Panel title="QT Cumulative Runoff">
            <AreaChart points={data.charts.qt_cumulative} color="var(--warn)" label="QT runoff" />
          </Panel>
        </Section>

        {data.notes?.net_liquidity ? (
          <p style={{ color: "var(--ink-faint)", fontSize: 12, marginTop: -4 }}>
            Note: {data.notes.net_liquidity}
          </p>
        ) : null}
      </div>
    </>
  );
}

export default async function CreditPage() {
  try {
    const [data, freshness] = await Promise.all([getCreditLiquidity(), getFreshness()]);
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
