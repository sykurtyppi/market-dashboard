import { getOptionsFlow, getFreshness, OptionsFlow, OptionsETF, Freshness } from "@/lib/api";
import Topbar from "@/components/Topbar";
import Explainer from "@/components/Explainer";
import { Section, fmtAsOf } from "@/components/ui";

export const dynamic = "force-dynamic";

function EtfCard({ e }: { e: OptionsETF }) {
  const calls = e.call_volume;
  const puts = e.put_volume;
  const total = (calls ?? 0) + (puts ?? 0);
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
      {/* Volume split bar + fixed two-column figures — the old flex-wrap legend
          stacked Calls/Puts on some cards and inlined them on others. */}
      {calls !== null && puts !== null && total > 0 ? (
        <div style={{ display: "flex", height: 6, borderRadius: 3, overflow: "hidden", marginTop: 12, background: "var(--line)" }} aria-hidden>
          <span style={{ width: `${(calls / total) * 100}%`, background: "var(--good)" }} />
          <span style={{ flex: 1, background: "var(--crit)" }} />
        </div>
      ) : null}
      <div className="mono" style={{ display: "flex", justifyContent: "space-between", gap: 8, marginTop: 8, fontSize: 11, color: "var(--ink-muted)", whiteSpace: "nowrap" }}>
        <span>Calls {calls === null ? "—" : calls.toLocaleString()}</span>
        <span>Puts {puts === null ? "—" : puts.toLocaleString()}</span>
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
        <Explainer
          title="How to read this — options flow"
          intro="Nearest-expiry index-ETF options are where hedging and speculation show up first."
          points={[
            { label: "Put/Call ratio:", text: "above ~1.0 leans defensive (more puts bought); below ~0.7 leans complacent." },
            { label: "Volume split bar:", text: "call (green) versus put (red) share of total volume at a glance." },
            { label: "Sentiment tag:", text: "blends volume with near-the-money open interest, so it can disagree with the raw ratio — open interest carries positioning the day's volume doesn't." },
          ]}
          caveat="0–1DTE flows dominate these figures and shift fast intraday."
        />
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
