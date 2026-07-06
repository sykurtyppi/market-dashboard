import { getSectors, getFreshness, Sectors, Freshness, SectorRow } from "@/lib/api";
import Topbar from "@/components/Topbar";
import { AreaChart } from "@/components/Charts";
import { Section, Panel } from "@/components/ui";

export const dynamic = "force-dynamic";

function changeColor(state: string): string {
  if (state === "good") return "var(--good)";
  if (state === "crit") return "var(--crit)";
  return "var(--ink-muted)";
}

function SectorTableRow({ s }: { s: SectorRow }) {
  const change = s.change_pct;
  return (
    <tr>
      <td className="mono" style={{ fontWeight: 600 }}>{s.ticker}</td>
      <td>{s.name}</td>
      <td className="src">{s.category}</td>
      <td className="num mono" style={{ color: changeColor(s.state) }}>
        {change === null ? "—" : `${change > 0 ? "+" : ""}${change.toFixed(2)}%`}
      </td>
      <td className="num mono">{s.price === null ? "—" : s.price.toFixed(2)}</td>
    </tr>
  );
}

function Content({ data, freshness }: { data: Sectors; freshness: Freshness }) {
  const r = data.rotation;
  const curve = data.vix_term.map((t) => ({ date: t.maturity, value: t.value }));
  return (
    <>
      <Topbar title="Sectors & VIX" subtitle="Sector rotation & VIX term structure" freshness={freshness} />
      <div className="content">
        {data.warnings.length > 0 ? (
          <div className="notice">
            <span className="dot warn" />
            {data.warnings.join(" ")}
          </div>
        ) : null}
        <Section title="Sector Rotation" aside={<span className="mono">{data.as_of ?? "—"}</span>}>
          <div className="regime" style={{ gridTemplateColumns: "1.2fr 2fr" }}>
            <div className="regime-cell lead">
              <span className="k">Signal</span>
              <span className="v" style={{ fontSize: 20, marginTop: 4 }}>
                <span className={`dot ${r.state}`} />{r.signal ?? "—"}
              </span>
              <span className="note">{r.interpretation}</span>
            </div>
            <div className="regime-cell">
              <span className="k">Leading Sectors</span>
              <div style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 6 }}>
                {r.leading_sectors.map((s) => (
                  <span key={s} className="mono" style={{ fontSize: 13 }}>{s}</span>
                ))}
              </div>
            </div>
          </div>
        </Section>

        <section className="grid-2">
          <Panel title="VIX Term Structure" sub={data.vix_structure ?? undefined}>
            <AreaChart points={curve} color="var(--accent)" label="VIX tenor" />
            <div className="legend">
              {data.vix_term.map((t) => (
                <span key={t.maturity} className="mono">{t.maturity}: {t.value.toFixed(2)}</span>
              ))}
            </div>
          </Panel>
          <Panel title="Sector Performance" sub="1d">
            <div className="table-wrap" style={{ border: "none" }}>
              <table>
                <thead>
                  <tr><th>ETF</th><th>Sector</th><th>Type</th><th className="num">1d</th><th className="num">Price</th></tr>
                </thead>
                <tbody>
                  {data.sectors.length > 0 ? (
                    data.sectors.map((s) => <SectorTableRow key={s.ticker} s={s} />)
                  ) : (
                    <tr><td colSpan={5}><div className="empty-state">Sector ETF data unavailable right now.</div></td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </Panel>
        </section>
      </div>
    </>
  );
}

export default async function SectorsPage() {
  try {
    const [data, freshness] = await Promise.all([getSectors(), getFreshness()]);
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
