import { getInstitutional, getFreshness, Institutional, Freshness } from "@/lib/api";
import Topbar from "@/components/Topbar";
import { Section, Panel } from "@/components/ui";
import { ReactNode } from "react";

export const dynamic = "force-dynamic";

function Stat({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: "1px solid var(--line)" }}>
      <span style={{ color: "var(--ink-muted)", fontSize: 13 }}>{label}</span>
      <span className="mono" style={{ fontSize: 13 }}>{value}</span>
    </div>
  );
}

function n(v: number | null, digits = 2, suffix = ""): string {
  return v === null ? "—" : `${v.toFixed(digits)}${suffix}`;
}

function Content({ data, freshness }: { data: Institutional; freshness: Freshness }) {
  const dp = data.dark_pool;
  const ins = data.insider;
  const au = data.auctions;
  return (
    <>
      <Topbar title="Institutional Flow" subtitle="Dark pool, insider & Treasury auctions" freshness={freshness} />
      <div className="content">
        {data.warnings.length > 0 ? (
          <div className="notice"><span className="dot warn" />{data.warnings.join(" ")}</div>
        ) : null}

        <Section title="Institutional Activity" aside={<span className="mono">{data.as_of ? String(data.as_of).slice(0, 10) : "—"}</span>}>
          <section className="grid-2" style={{ gridTemplateColumns: "1fr 1fr 1fr" }}>
            <Panel title="Dark Pool" sub={dp ? undefined : "unavailable"}>
              {dp ? (
                <>
                  <div className="caption" style={{ marginTop: 0, marginBottom: 8 }}><span className={`dot ${dp.state}`} />{dp.sentiment ?? "—"}</div>
                  <Stat label="Avg off-exchange" value={n(dp.avg_pct, 1, "%")} />
                  <Stat label="ETF" value={n(dp.etf_pct, 1, "%")} />
                  <Stat label="Single stocks" value={n(dp.stock_pct, 1, "%")} />
                  <p style={{ color: "var(--ink-faint)", fontSize: 11.5, marginTop: 10 }}>{dp.interpretation}</p>
                </>
              ) : <div className="empty-state">Dark pool data unavailable.</div>}
            </Panel>

            <Panel title="Insider Activity" sub={ins ? `${n(ins.period_days, 0)}d` : "unavailable"}>
              {ins ? (
                <>
                  <div className="caption" style={{ marginTop: 0, marginBottom: 8 }}><span className={`dot ${ins.state}`} />{ins.sentiment ?? "—"}</div>
                  <Stat label="Transactions" value={n(ins.total_transactions, 0)} />
                  <Stat label="Buys" value={n(ins.buy_count, 0)} />
                  <Stat label="Sells" value={n(ins.sell_count, 0)} />
                  <Stat label="Buy/Sell ratio" value={n(ins.buy_sell_ratio, 2)} />
                </>
              ) : <div className="empty-state">Insider data unavailable.</div>}
            </Panel>

            <Panel title="Treasury Auctions" sub={au ? undefined : "unavailable"}>
              {au ? (
                <>
                  <div className="caption" style={{ marginTop: 0, marginBottom: 8 }}><span className={`dot ${au.state}`} />{au.health ?? "—"}</div>
                  <Stat label="Bid-to-cover" value={n(au.avg_bid_to_cover, 2)} />
                  <Stat label="Indirect" value={n(au.avg_indirect_pct, 1, "%")} />
                  <Stat label="Direct" value={n(au.avg_direct_pct, 1, "%")} />
                  <Stat label="Auctions (weak/strong)" value={`${n(au.auction_count, 0)} (${n(au.weak_auctions, 0)}/${n(au.strong_auctions, 0)})`} />
                </>
              ) : <div className="empty-state">Auction data unavailable.</div>}
            </Panel>
          </section>
        </Section>
      </div>
    </>
  );
}

export default async function InstitutionalPage() {
  try {
    const [data, freshness] = await Promise.all([getInstitutional(), getFreshness()]);
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
