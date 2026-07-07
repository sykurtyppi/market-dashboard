import { getLeft, getFreshness, Left, Freshness } from "@/lib/api";
import Topbar from "@/components/Topbar";
import GapNotice from "@/components/GapNotice";
import { MultiLineChart } from "@/components/Charts";
import { MetricCard, Section, Panel, fmtAsOf } from "@/components/ui";

export const dynamic = "force-dynamic";

function Content({ data, freshness }: { data: Left; freshness: Freshness }) {
  return (
    <>
      <Topbar title="LEFT Strategy" subtitle="Credit-spread trend signal (HYG OAS vs 330d EMA)" freshness={freshness} />
      <div className="content">
        {data.warnings.length > 0 ? (
          <div className="notice"><span className="dot warn" />{data.warnings.join(" ")}</div>
        ) : null}

        <Section title="Signal" aside={<span className="mono">{fmtAsOf(data.as_of)}</span>}>
          <div className="regime" style={{ gridTemplateColumns: "1fr 3fr" }}>
            <div className="regime-cell lead">
              <span className="k">LEFT Signal</span>
              <span className="v" style={{ fontSize: 24, marginTop: 6 }}>
                <span className={`dot ${data.state}`} />{data.signal ?? "—"}
              </span>
              <span className="note">Buy when spread crosses below its 330-day EMA</span>
            </div>
            <div className="regime-cell" style={{ display: "flex", alignItems: "center" }}>
              <span className="note" style={{ fontSize: 12.5 }}>
                The LEFT model tracks high-yield credit spreads (HYG OAS) against a long-term trend.
                Tightening spreads below trend signal risk-on; widening above trend signals risk-off.
              </span>
            </div>
          </div>
        </Section>

        <Section title="Key Indicators">
          <div className="metrics">
            {data.metrics.map((m) => <MetricCard key={m.key} m={m} />)}
          </div>
        </Section>

        <Section title="Spread vs Trend" aside="2Y · %">
          <Panel title="HYG OAS vs 330-day EMA">
            <MultiLineChart series={[
              { points: data.charts.spread, color: "var(--crit)", label: "HYG OAS" },
              { points: data.charts.ema, color: "var(--accent)", label: "330d EMA" },
            ]} />
            <GapNotice points={data.charts.spread} />
            <div className="legend">
              <span><i style={{ background: "var(--crit)" }} />HYG OAS</span>
              <span><i style={{ background: "var(--accent)" }} />330d EMA</span>
            </div>
          </Panel>
        </Section>
      </div>
    </>
  );
}

export default async function LeftPage() {
  try {
    const [data, freshness] = await Promise.all([getLeft(), getFreshness()]);
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
