import { getRepo, getFreshness, Repo, Freshness } from "@/lib/api";
import Topbar from "@/components/Topbar";
import GapNotice from "@/components/GapNotice";
import { AreaChart } from "@/components/Charts";
import { Section, Panel, fmtAsOf } from "@/components/ui";

export const dynamic = "force-dynamic";

function Content({ data, freshness }: { data: Repo; freshness: Freshness }) {
  const state = data.state;
  return (
    <>
      <Topbar title="Repo Market" subtitle="SOFR & overnight funding" freshness={freshness} />
      <div className="content">
        {data.warnings.length > 0 ? (
          <div className="notice"><span className="dot warn" />{data.warnings.join(" ")}</div>
        ) : null}

        <Section title="Funding Regime" aside={<span className="mono">{fmtAsOf(data.as_of)}</span>}>
          <div className="regime" style={{ gridTemplateColumns: "1.2fr 2fr" }}>
            <div className="regime-cell lead">
              <span className="k">Liquidity</span>
              <span className="v" style={{ fontSize: 20, marginTop: 4 }}>
                <span className={`dot ${state}`} />{data.regime ?? "—"}
              </span>
              <span className="note">{data.regime_note}</span>
            </div>
            {/* Strip cells carry the state dot and source so the old duplicate
                "Key Indicators" cards (same numbers, same sources) could go. */}
            <div className="regime-cell" style={{ display: "flex", gap: 32, alignItems: "center" }}>
              {data.metrics.map((m) => (
                <div key={m.key}>
                  <span className="k">{m.label}</span>
                  <div className="v mono" style={{ fontSize: 20, marginTop: 6 }}>
                    <span className={`dot ${m.state}`} />
                    {m.value === null ? "—" : `${m.value.toFixed(2)}${m.unit}`}
                  </div>
                  <span className="note">{m.source}</span>
                </div>
              ))}
            </div>
          </div>
        </Section>

        <section className="grid-2">
          <Panel title="SOFR" sub="1Y · %">
            <AreaChart points={data.charts.sofr_history} color="var(--accent)" label="SOFR" />
            <GapNotice points={data.charts.sofr_history} />
            <div className="legend">
              <span><i style={{ background: "var(--accent)" }} />SOFR</span>
              <span style={{ color: "var(--ink-faint)" }}>Secured Overnight Financing Rate</span>
            </div>
          </Panel>
          <Panel title="Overnight RRP Volume" sub="1Y">
            <AreaChart points={data.charts.rrp_history} color="var(--good)" label="Overnight RRP" />
            <GapNotice points={data.charts.rrp_history} />
            <div className="legend">
              <span><i style={{ background: "var(--good)" }} />RRP</span>
              <span style={{ color: "var(--ink-faint)" }}>Reverse-repo take-up (liquidity buffer)</span>
            </div>
          </Panel>
        </section>
      </div>
    </>
  );
}

export default async function RepoPage() {
  try {
    const [data, freshness] = await Promise.all([getRepo(), getFreshness()]);
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
