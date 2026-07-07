import { getCrossAsset, getFreshness, CrossAsset, Freshness, AssetPerf, Correlation } from "@/lib/api";
import Topbar from "@/components/Topbar";
import Explainer from "@/components/Explainer";
import { Section, Panel, fmtAsOf } from "@/components/ui";

export const dynamic = "force-dynamic";

// The backend regime signal is a raw enum (RISK_ON, RISK_OFF, MIXED) — render
// it as a human label instead of leaking the internal constant.
function fmtSignal(signal: string | null): string {
  if (!signal) return "—";
  return signal.replace(/_/g, "-").toLowerCase().replace(/(^|-)[a-z]/g, (c) => c.toUpperCase());
}

function assetColor(state: string): string {
  if (state === "good") return "var(--good)";
  if (state === "crit") return "var(--crit)";
  return "var(--ink-muted)";
}

function pct(v: number | null): string {
  return v === null ? "—" : `${v > 0 ? "+" : ""}${v.toFixed(2)}%`;
}

function AssetRow({ a }: { a: AssetPerf }) {
  return (
    <tr>
      <td className="mono" style={{ fontWeight: 600 }}>{a.ticker}</td>
      <td>{a.name}</td>
      <td className="num mono" style={{ color: assetColor(a.state) }}>{pct(a.change_1d_pct)}</td>
      <td className="num mono" style={{ color: "var(--ink-muted)" }}>{pct(a.change_1m_pct)}</td>
    </tr>
  );
}

function CorrRow({ c }: { c: Correlation }) {
  const v = c.correlation;
  const color = v === null ? "var(--ink-muted)" : v > 0 ? "var(--accent)" : "var(--crit)";
  return (
    <tr>
      <td style={{ fontWeight: 600 }}>{c.pair}</td>
      <td className="num mono" style={{ color }}>{v === null ? "—" : v.toFixed(2)}</td>
      <td className="src">{c.strength}</td>
      <td style={{ fontSize: 12, color: "var(--ink-muted)" }}>{c.interpretation}</td>
    </tr>
  );
}

function Content({ data, freshness }: { data: CrossAsset; freshness: Freshness }) {
  const r = data.regime;
  return (
    <>
      <Topbar title="Cross-Asset" subtitle="Regime & inter-market correlations" freshness={freshness} />
      <div className="content">
        {data.warnings.length > 0 ? (
          <div className="notice"><span className="dot warn" />{data.warnings.join(" ")}</div>
        ) : null}

        <Section title="Market Regime" aside={<span className="mono">{fmtAsOf(data.as_of)}</span>}>
          <div className="regime" style={{ gridTemplateColumns: "1fr 2fr" }}>
            <div className="regime-cell lead">
              <span className="k">Regime</span>
              <span className="v" style={{ fontSize: 20, marginTop: 4 }}>
                <span className={`dot ${r.state}`} />{fmtSignal(r.signal)}
              </span>
              <span className="note mono">
                {r.confidence != null ? `${r.confidence.toFixed(0)}% confidence` : ""}
              </span>
            </div>
            <div className="regime-cell" style={{ display: "flex", alignItems: "center" }}>
              <span className="note" style={{ fontSize: 13 }}>{r.description}</span>
            </div>
          </div>
        </Section>

        <section className="grid-2">
          <Panel title="Asset Performance" sub="1D · 1M">
            <div className="table-wrap" style={{ border: "none" }}>
              <table>
                <thead><tr><th>Ticker</th><th>Asset</th><th className="num">1D</th><th className="num">1M</th></tr></thead>
                <tbody>
                  {data.assets.length > 0 ? (
                    data.assets.map((a) => <AssetRow key={a.ticker} a={a} />)
                  ) : (
                    <tr><td colSpan={4}><div className="empty-state">Asset data unavailable.</div></td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </Panel>
          <Panel title="Key Correlations" sub="rolling">
            <div className="table-wrap" style={{ border: "none" }}>
              <table>
                <thead><tr><th>Pair</th><th className="num">ρ</th><th>Strength</th><th>Read</th></tr></thead>
                <tbody>
                  {data.correlations.map((c) => <CorrRow key={c.pair} c={c} />)}
                </tbody>
              </table>
            </div>
          </Panel>
        </section>
        <Explainer
          title="How to read this — cross-asset regime"
          intro="Different assets moving together (or apart) reveal the market's regime better than any single index."
          points={[
            { label: "Regime:", text: "equities up with defensives (bonds, gold, dollar) down reads risk-on; the reverse reads risk-off. Confidence reflects how cleanly the assets agree." },
            { label: "1D vs 1M:", text: "the daily move versus the one-month trend. A red day inside a green month is noise; both columns aligned is signal." },
            { label: "Correlations:", text: "stock–bond correlation negative means diversification is working; strongly positive means both are being driven by rates." },
          ]}
          caveat="Prices from Yahoo Finance. Not investment advice."
        />
      </div>
    </>
  );
}

export default async function CrossAssetPage() {
  try {
    const [data, freshness] = await Promise.all([getCrossAsset(), getFreshness()]);
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
