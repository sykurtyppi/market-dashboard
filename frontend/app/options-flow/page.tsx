import { getOptionsFlow, getFreshness, OptionsFlow, OptionsETF, Freshness } from "@/lib/api";
import Topbar from "@/components/Topbar";
import { Section, fmtAsOf } from "@/components/ui";

export const dynamic = "force-dynamic";

function EtfCard({ e }: { e: OptionsETF }) {
  return (
    <div className="card">
      <div className="k" style={{ display: "flex", justifyContent: "space-between" }}>
        <span>{e.ticker}{e.price !== null ? ` · ${e.price.toFixed(2)}` : ""}</span>
        <span className="tag"><span className={`dot ${e.state}`} />{e.sentiment ?? "—"}</span>
      </div>
      <div className="v mono">{e.put_call_ratio === null ? "—" : e.put_call_ratio.toFixed(3)}</div>
      <div className="caption" style={{ justifyContent: "space-between" }}>
        <span>Put / Call ratio</span>
        <span>{e.dte !== null ? `${e.dte}DTE` : ""}</span>
      </div>
      <div className="legend" style={{ marginTop: 10 }}>
        <span className="mono">Calls {e.call_volume === null ? "—" : e.call_volume.toLocaleString()}</span>
        <span className="mono">Puts {e.put_volume === null ? "—" : e.put_volume.toLocaleString()}</span>
      </div>
    </div>
  );
}

function Content({ data, freshness }: { data: OptionsFlow; freshness: Freshness }) {
  return (
    <>
      <Topbar title="Options Flow" subtitle="Index-ETF options positioning" freshness={freshness} />
      <div className="content">
        {data.warnings.length > 0 ? (
          <div className="notice"><span className="dot warn" />{data.warnings.join(" ")}</div>
        ) : null}

        <Section title="Nearest-Expiry Flow" aside={<span className="mono">{fmtAsOf(data.as_of)}</span>}>
          {data.etfs.length > 0 ? (
            <div className="metrics" style={{ gridTemplateColumns: "repeat(3, 1fr)" }}>
              {data.etfs.map((e) => <EtfCard key={e.ticker} e={e} />)}
            </div>
          ) : (
            <div className="panel"><div className="empty-state">Options flow data unavailable.</div></div>
          )}
          <p style={{ color: "var(--ink-faint)", fontSize: 12, marginTop: 12 }}>
            Put/Call ratio above ~1.0 leans defensive; below ~0.7 leans complacent. Sentiment blends volume and near-the-money open interest.
          </p>
        </Section>
      </div>
    </>
  );
}

export default async function OptionsFlowPage() {
  try {
    const [data, freshness] = await Promise.all([getOptionsFlow(), getFreshness()]);
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
