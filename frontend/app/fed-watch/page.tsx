import { getFedWatch, getFreshness, FedWatch, Freshness } from "@/lib/api";
import Topbar from "@/components/Topbar";
import { MetricCard, Section, Panel, fmtAsOf } from "@/components/ui";

export const dynamic = "force-dynamic";

function Content({ data, freshness }: { data: FedWatch; freshness: Freshness }) {
  return (
    <>
      <Topbar title="Fed Watch" subtitle="Rate probabilities & policy path" freshness={freshness} />
      <div className="content">
        {data.warnings.length > 0 ? (
          <div className="notice"><span className="dot warn" />{data.warnings.join(" ")}</div>
        ) : null}

        <Section title="Policy Rate" aside={<span className="mono">{fmtAsOf(data.as_of)}</span>}>
          <div className="regime" style={{ gridTemplateColumns: "1.3fr 1fr 1fr" }}>
            <div className="regime-cell lead">
              <span className="k">Target Range</span>
              <span className="regime-score mono" style={{ fontSize: 24 }}>{data.current_rate ?? "—"}</span>
              <span className="note">{data.market_bias ? `Market bias: ${data.market_bias}` : "Federal funds target"}</span>
            </div>
            <div className="regime-cell">
              <span className="k">Next Meeting</span>
              <span className="v" style={{ fontSize: 15 }}>{data.next_meeting.date ?? "—"}</span>
              <span className="note">
                {data.next_meeting.days_until != null ? `in ${data.next_meeting.days_until} days` : ""}
              </span>
            </div>
            <div className="regime-cell">
              <span className="k">Most Likely</span>
              <span className="v" style={{ fontSize: 15 }}>
                <span className={`dot ${data.bias_state}`} />{data.most_likely.outcome ?? "—"}
              </span>
              <span className="note mono">
                {data.most_likely.pct != null ? `${data.most_likely.pct.toFixed(0)}% implied` : ""}
              </span>
            </div>
          </div>
        </Section>

        <section className="grid-2">
          <Panel title="Next-Meeting Probabilities" sub={data.degraded ? "fallback estimate — futures data unavailable" : "implied by fed funds futures"}>
            {data.probabilities.length > 0 ? (
              data.probabilities.map((p) => (
                <div className="prob-row" key={p.outcome}>
                  <span className="prob-label">{p.outcome}</span>
                  {/* Absolute %, not normalized to the max — a 96% probability
                      must not render as a 100% (certain-looking) bar. */}
                  <span className="prob-track">
                    <span className="prob-fill" style={{ width: `${Math.min(100, Math.max(0, p.pct ?? 0))}%` }} />
                  </span>
                  <span className="prob-pct mono">{p.pct != null ? `${p.pct.toFixed(0)}%` : "—"}</span>
                </div>
              ))
            ) : (
              <div className="empty-state">Probability data unavailable.</div>
            )}
          </Panel>
          <Panel title="Rate Detail">
            <div className="metrics" style={{ gridTemplateColumns: "1fr 1fr" }}>
              {data.metrics.map((m) => <MetricCard key={m.key} m={m} />)}
            </div>
          </Panel>
        </section>
      </div>
    </>
  );
}

export default async function FedWatchPage() {
  try {
    const [data, freshness] = await Promise.all([getFedWatch(), getFreshness()]);
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
