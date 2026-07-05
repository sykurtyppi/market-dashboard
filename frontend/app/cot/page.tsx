import { getCot, getFreshness, COT, COTPosition, Freshness } from "@/lib/api";
import Topbar from "@/components/Topbar";
import { Section, Panel } from "@/components/ui";

export const dynamic = "force-dynamic";

function signColor(v: number | null): string {
  if (v === null) return "var(--ink-muted)";
  return v > 0 ? "var(--accent)" : v < 0 ? "var(--crit)" : "var(--ink-muted)";
}

function signed(v: number | null): string {
  if (v === null) return "—";
  return `${v > 0 ? "+" : ""}${v.toLocaleString()}`;
}

function PositionRow({ p }: { p: COTPosition }) {
  return (
    <tr>
      <td className="mono" style={{ fontWeight: 600 }}>{p.symbol}</td>
      <td>{p.name}</td>
      <td className="num mono" style={{ color: signColor(p.spec_net) }}>{p.spec_net === null ? "—" : p.spec_net.toLocaleString()}</td>
      <td className="num mono" style={{ color: signColor(p.spec_net_change) }}>{signed(p.spec_net_change)}</td>
      <td className="num mono" style={{ color: signColor(p.comm_net) }}>{p.comm_net === null ? "—" : p.comm_net.toLocaleString()}</td>
      <td className="num mono">{p.open_interest === null ? "—" : p.open_interest.toLocaleString()}</td>
    </tr>
  );
}

function Content({ data, freshness }: { data: COT; freshness: Freshness }) {
  return (
    <>
      <Topbar title="COT Positioning" subtitle="CFTC Commitments of Traders" freshness={freshness} />
      <div className="content">
        {data.warnings.length > 0 ? (
          <div className="notice"><span className="dot warn" />{data.warnings.join(" ")}</div>
        ) : null}

        <Section title="Net Positioning" aside={<span className="mono">{data.as_of ? String(data.as_of).slice(0, 10) : "—"}</span>}>
          <Panel title="Speculators vs Commercials" sub="net contracts">
            <div className="table-wrap" style={{ border: "none" }}>
              <table>
                <thead>
                  <tr>
                    <th>Contract</th><th>Name</th>
                    <th className="num">Spec Net</th><th className="num">Δ Spec</th>
                    <th className="num">Comm Net</th><th className="num">Open Int.</th>
                  </tr>
                </thead>
                <tbody>
                  {data.positions.length > 0 ? (
                    data.positions.map((p) => <PositionRow key={p.symbol} p={p} />)
                  ) : (
                    <tr><td colSpan={6}><div className="empty-state">COT data unavailable.</div></td></tr>
                  )}
                </tbody>
              </table>
            </div>
            <div className="legend">
              <span style={{ color: "var(--ink-faint)" }}>
                Speculators (funds) are trend-followers; Commercials (hedgers) take the other side. Extreme net positioning often precedes reversals.
              </span>
            </div>
          </Panel>
        </Section>
      </div>
    </>
  );
}

export default async function COTPage() {
  try {
    const [data, freshness] = await Promise.all([getCot(), getFreshness()]);
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
