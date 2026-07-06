import { getCta, getFreshness, CTA, CTAPosition, Freshness } from "@/lib/api";
import Topbar from "@/components/Topbar";
import { Section, Panel } from "@/components/ui";

export const dynamic = "force-dynamic";

function PositionRow({ p }: { p: CTAPosition }) {
  const color = p.state === "good" ? "var(--good)" : p.state === "crit" ? "var(--crit)" : "var(--ink-muted)";
  return (
    <tr>
      <td className="mono" style={{ fontWeight: 600 }}>{p.symbol}</td>
      <td><span className="tag"><span className={`dot ${p.state}`} />{p.position ?? "—"}</span></td>
      <td className="num mono" style={{ color }}>
        {p.exposure === null ? "—" : `${p.exposure > 0 ? "+" : ""}${(p.exposure * 100).toFixed(0)}%`}
      </td>
    </tr>
  );
}

function Content({ data, freshness }: { data: CTA; freshness: Freshness }) {
  return (
    <>
      <Topbar title="CTA Flow" subtitle="Systematic trend-following positioning" freshness={freshness} />
      <div className="content">
        {data.warnings.length > 0 ? (
          <div className="notice"><span className="dot warn" />{data.warnings.join(" ")}</div>
        ) : null}

        <Section title="Positioning Summary">
          <div className="regime" style={{ gridTemplateColumns: "1fr 1fr 1fr" }}>
            <div className="regime-cell">
              <span className="k">Long</span>
              <span className="v mono" style={{ fontSize: 22 }}><span className="dot good" />{data.long_count}</span>
            </div>
            <div className="regime-cell">
              <span className="k">Short</span>
              <span className="v mono" style={{ fontSize: 22 }}><span className="dot crit" />{data.short_count}</span>
            </div>
            <div className="regime-cell">
              <span className="k">Flat</span>
              <span className="v mono" style={{ fontSize: 22 }}><span className="dot neutral" />{data.flat_count}</span>
            </div>
          </div>
        </Section>

        <Section title="Model Positions" aside="trend-following exposure">
          <Panel title="Estimated CTA Exposure">
            <div className="table-wrap" style={{ border: "none" }}>
              <table>
                <thead><tr><th>Asset</th><th>Position</th><th className="num">Exposure</th></tr></thead>
                <tbody>
                  {data.positions.length > 0 ? (
                    data.positions.map((p) => <PositionRow key={p.symbol} p={p} />)
                  ) : (
                    <tr><td colSpan={3}><div className="empty-state">CTA data unavailable.</div></td></tr>
                  )}
                </tbody>
              </table>
            </div>
            <div className="legend">
              <span style={{ color: "var(--ink-faint)" }}>
                CTAs follow price trends; positioning flips at technical levels can amplify momentum. Estimated, not disclosed.
              </span>
            </div>
          </Panel>
        </Section>
      </div>
    </>
  );
}

export default async function CTAPage() {
  try {
    const [data, freshness] = await Promise.all([getCta(), getFreshness()]);
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
