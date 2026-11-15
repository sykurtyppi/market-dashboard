import type { LiquidityComponent, LiquidityData } from "@/lib/api";
import { fmt, fmtAsOf } from "@/components/ui";

const COMPONENTS = [
  { key: "fed_balance", label: "Fed balance sheet" },
  { key: "tga", label: "Treasury General Account" },
  { key: "rrp_on", label: "Overnight reverse repo" },
] as const;

function EvidenceValue({ value }: { value: number | null }) {
  return <p className="liquidity-value mono">{value !== null && Number.isFinite(value) ? `${fmt(value, "")} USD billions` : "Unavailable"}</p>;
}

function ComponentObservations({ components, latest = false, proxyUnavailable = false }: { components: LiquidityComponent[]; latest?: boolean; proxyUnavailable?: boolean }) {
  const rows = COMPONENTS.map(({ key, label }) => ({ key, label, component: components.find((c) => c.key === key) }));
  const available = rows.filter(({ component: c }) => c?.value != null && Number.isFinite(c.value) && c.as_of).length;
  const title = latest ? "Latest component observations" : "Proxy component observations";
  return (
    <div className="liquidity-observations">
      <h3>{title}</h3>
      <p className="data-trust-note">{available}/3 components available with value and observation date{latest ? ". Latest stored observations are independent; not a same-date total." : proxyUnavailable ? ". Candidate observations only; no proxy was calculated. Observed coverage does not establish matching eligibility." : ". Actual inputs used for the exploratory proxy, not snapshot dates."}</p>
      <div className="table-wrap" role="region" aria-label={`${title} scroll area`} tabIndex={0}>
        <table aria-label={title}>
          <thead><tr><th scope="col">Component / series</th><th scope="col" className="num">USD billions</th><th scope="col">Observation date</th>{!latest ? <th scope="col">Age vs proxy date</th> : null}</tr></thead>
          <tbody>{rows.map(({ key, label, component: c }) => (
            <tr key={key}>
              <th scope="row">{c?.label ?? label}<span className="liquidity-series mono">{c?.series ?? "Series unknown"}</span></th>
              <td className="num mono">{c?.value != null && Number.isFinite(c.value) ? fmt(c.value, "") : "Unavailable"}</td>
              <td className="mono">{c?.as_of ? fmtAsOf(c.as_of) : "Unknown"}</td>
              {!latest ? <td>{c?.age_days != null ? `${c.age_days} ${c.age_days === 1 ? "day" : "days"}` : "Unknown"}</td> : null}
            </tr>
          ))}</tbody>
        </table>
      </div>
    </div>
  );
}

export default function LiquidityEvidence({ data }: { data?: LiquidityData | null }) {
  const proxyStatus = data?.proxy.status === "exact" ? "Exact date match" : data?.proxy.status === "backward_matched" ? "Backward-matched" : "Unavailable";
  return (
    <section className="liquidity-evidence panel" aria-labelledby="liquidity-evidence-title">
      <div className="section-head">
        <h2 id="liquidity-evidence-title">Liquidity Evidence</h2>
        <span className="aside">{data ? `${data.status[0].toUpperCase()}${data.status.slice(1)} · USD billions` : "Unknown coverage"}</span>
      </div>
      {!data ? (
        <p className="data-trust-note">Liquidity evidence unavailable: this API did not supply the evidence contract. Observation dates and component coverage are unknown; no proxy is inferred from legacy metrics.</p>
      ) : (
        <>
          <div className="grid-2">
            <div className="card" role="group" aria-label="Strict stored net liquidity">
              <h3>Strict stored net liquidity</h3>
              <EvidenceValue value={data.strict.value} />
              <p className="data-trust-note">{data.strict.as_of ? `Observation: ${fmtAsOf(data.strict.as_of)}` : "Observation date unknown"}</p>
              <p className="data-trust-note">Latest complete stored net observation. Kept separate from the exploratory proxy; no forward fill.</p>
            </div>
            <div className="card" role="group" aria-label="Exploratory liquidity proxy">
              <h3>Exploratory liquidity proxy</h3>
              <EvidenceValue value={data.proxy.value} />
              <p className="data-trust-note">{data.proxy.as_of ? `Observation: ${fmtAsOf(data.proxy.as_of)}` : "Observation date unknown"}</p>
              <p className="liquidity-match">{proxyStatus}</p>
              {data.proxy.reason ? <p className="data-trust-note">{data.proxy.reason}</p> : null}
            </div>
          </div>
          <ComponentObservations components={data.proxy.components} proxyUnavailable={data.proxy.status === "unavailable"} />
          <ComponentObservations components={data.latest_components} latest />
        </>
      )}
      <ul className="liquidity-notes data-trust-note">
        <li>WTREGEN is a weekly average, not a point-in-time Treasury cash balance.</li>
        <li>Exploratory RRP carry uses a {data?.proxy.max_rrp_age_days ?? 4} calendar-day backward tolerance; it is not calendar-aware. Exact date matches and backward-matched inputs are labeled separately.</li>
        <li>Publication/vintage eligibility is unknown. This evidence is not a historical, trading, or backtest signal.</li>
        {data?.notes.map((note, index) => <li key={index}>{note}</li>)}
      </ul>
    </section>
  );
}
