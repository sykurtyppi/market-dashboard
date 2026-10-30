import Link from "next/link";
import { getDailyBriefing, getOverview, DailyBriefingRecord, RegimeComponent, DetailRow, Overview } from "@/lib/api";
import { loadOverviewPageData } from "@/lib/overviewPageData.mjs";
import Topbar from "@/components/Topbar";
import GapNotice from "@/components/GapNotice";
import { AreaChart, MultiLineChart } from "@/components/Charts";
import { MetricCard, Section, Panel, fmtAsOf } from "@/components/ui";

export const dynamic = "force-dynamic";

const DRIVER_LABELS: Record<string, string> = {
  vix: "VIX", credit: "Credit", sentiment: "Sentiment", vrp: "VRP",
  breadth: "Breadth", move: "MOVE", skew: "SKEW", put_call: "Put/call",
};

// Each input's share of the headline, in points: normalized score × weight,
// over the weights present. They sum to the score, so the order is the
// true order of influence (sorting by the 0–100 scores alone put a 5%-weight
// input ahead of a 20% one).
function driversLine(b: NonNullable<Overview["regime"]["breakdown"]>): string {
  const totalWeight = Object.values(b.weights).reduce((acc, w) => acc + w, 0) || 1;
  const parts = Object.entries(b.components)
    .map(([k, v]) => [k, (v * (b.weights[k] ?? 0)) / totalWeight] as const)
    .sort((x, y) => y[1] - x[1])
    .map(([k, pts]) => {
      const label = k === "put_call" && b.put_call_source ? `${DRIVER_LABELS[k]} (${b.put_call_source})` : (DRIVER_LABELS[k] ?? k);
      return `${label} ${pts.toFixed(1)}`;
    });
  const missing = b.missing.length > 0 ? `not included: ${b.missing.map((k) => DRIVER_LABELS[k] ?? k).join(", ")}` : "";
  const hasContrarian = "sentiment" in b.components || "put_call" in b.components;
  const contrarian = hasContrarian ? "fear and heavy put buying lower the score" : "";
  return ["Points toward the score: " + (parts.join(" · ") || "none"), missing, contrarian].filter(Boolean).join(" · ");
}

function stateLabel(state: string): string {
  switch (state) {
    case "good": return "Supportive";
    case "warn": return "Elevated";
    case "crit": return "Stress";
    default: return "Neutral";
  }
}

function RegimeCell({ c }: { c: RegimeComponent }) {
  return (
    <div className="regime-cell">
      <span className="k">{c.label}</span>
      <span className="v"><span className={`dot ${c.state}`} />{c.value}</span>
      <span className="note">{c.note}</span>
    </div>
  );
}

function DetailTableRow({ r }: { r: DetailRow }) {
  const val = r.value === null ? "—" : r.unit === "%" ? `${r.value.toFixed(1)}%` : r.value.toFixed(2);
  return (
    <tr>
      <td>{r.indicator}</td>
      <td className="num mono">{val}</td>
      <td><span className="tag"><span className={`dot ${r.state}`} />{stateLabel(r.state)}</span></td>
      <td className="src">{r.source}</td>
    </tr>
  );
}

function outcome(value: number | null): string {
  return value === null ? "Pending" : `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
}

function DailyBriefingPanel({ briefing, history }: { briefing: DailyBriefingRecord | null; history: DailyBriefingRecord[] }) {
  if (!briefing) {
    return <div className="notice"><span className="dot warn" />The first briefing will be generated after a daily snapshot completes.</div>;
  }
  const score = briefing.change.current_score;
  const delta = briefing.change.score_delta;
  const postureState = briefing.posture === "normal" ? "good" : briefing.posture === "cautious" ? "warn" : "crit";
  return (
    <>
      {briefing.alert.should_alert ? (
        <div className="notice" style={{ borderColor: "var(--warn)", marginBottom: 16 }}>
          <span className="dot warn" />
          <strong>Material change:</strong>&nbsp;{briefing.alert.triggers.map((trigger) => trigger.message).join(" ")}
        </div>
      ) : null}
      <section className="panel" style={{ marginBottom: 18, padding: 0, overflow: "hidden" }}>
        <div className="briefing-hero">
          <div style={{ padding: 24, borderRight: "1px solid var(--line)", background: "var(--paper-warm)" }}>
            <div className="k" style={{ textTransform: "uppercase", letterSpacing: ".08em", color: "var(--ink-faint)", fontSize: 11 }}>Current posture</div>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 9 }}>
              <span className={`dot ${postureState}`} />
              <span style={{ fontFamily: "var(--serif)", fontSize: 34, textTransform: "capitalize" }}>{briefing.posture}</span>
            </div>
            <div className="mono" style={{ fontSize: 24, marginTop: 14 }}>
              {score ?? "—"}<span style={{ color: "var(--ink-faint)", fontSize: 13 }}> / 100</span>
            </div>
            <div style={{ color: "var(--ink-muted)", fontSize: 12.5, marginTop: 6 }}>
              {delta === null ? "First recorded briefing" : `${delta >= 0 ? "+" : ""}${delta.toFixed(1)} points vs prior snapshot`}
            </div>
            <div style={{ color: "var(--ink-faint)", fontSize: 12, marginTop: 8 }}>
              {briefing.change.previous_score === null
                ? "No prior briefing available"
                : briefing.change.inputs_comparable
                  ? `Regime ${briefing.change.regime_changed ? `changed from ${briefing.previous_posture}` : "did not change"}`
                  : "Regime comparison unavailable because model inputs changed"} · {briefing.as_of}
            </div>
          </div>
          <div style={{ padding: 24 }}>
            <div className="panel-head" style={{ padding: 0, marginBottom: 12 }}><span className="t">Why it matters</span></div>
            <p style={{ color: "var(--ink)", fontSize: 15, lineHeight: 1.6, marginBottom: 16 }}>{briefing.why_it_matters}</p>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
              <div>
                <div style={{ color: "var(--ink-faint)", fontSize: 11, textTransform: "uppercase", letterSpacing: ".06em" }}>Confirming</div>
                <div style={{ marginTop: 6, fontSize: 13 }}>{briefing.confirming.map((d) => d.label).join(" · ") || "No material confirmation"}</div>
              </div>
              <div>
                <div style={{ color: "var(--ink-faint)", fontSize: 11, textTransform: "uppercase", letterSpacing: ".06em" }}>Conflicting</div>
                <div style={{ marginTop: 6, fontSize: 13 }}>{briefing.conflicting.map((d) => d.label).join(" · ") || "No material conflict"}</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="grid-2" style={{ marginBottom: 18 }}>
        <Panel title="What changed" sub="ranked contribution delta">
          {briefing.drivers.length ? briefing.drivers.map((driver, index) => (
            <div key={driver.key} style={{ display: "grid", gridTemplateColumns: "28px 1fr auto", gap: 10, padding: "10px 0", borderBottom: "1px solid var(--line)" }}>
              <span className="mono" style={{ color: "var(--ink-faint)" }}>0{index + 1}</span>
              <div><strong>{driver.label}</strong><div style={{ fontSize: 12, color: "var(--ink-muted)", marginTop: 2 }}>{driver.direction}</div></div>
              <span className="mono" style={{ color: driver.risk_delta > 0 ? "var(--crit)" : "var(--good)" }}>{driver.risk_delta > 0 ? "+" : ""}{driver.risk_delta.toFixed(2)}</span>
            </div>
          )) : <p style={{ color: "var(--ink-muted)" }}>No material component changes.</p>}
        </Panel>
        <Panel title="What to inspect next" sub="direct evidence">
          {briefing.inspect_next.length ? briefing.inspect_next.map((item) => (
            <Link key={`${item.href}-${item.label}`} href={item.href} prefetch={false} style={{ display: "block", padding: "10px 0", borderBottom: "1px solid var(--line)", color: "var(--accent)" }}>
              <strong>{item.label} →</strong><span style={{ display: "block", color: "var(--ink-muted)", fontSize: 12, marginTop: 3 }}>{item.reason}</span>
            </Link>
          )) : <p style={{ color: "var(--ink-muted)" }}>No evidence page requires immediate inspection.</p>}
        </Panel>
      </section>

      <Section title="Accountability" aside="SPY outcome after briefing date">
        <div className="metrics">
          {(["1d", "5d", "20d"] as const).map((horizon) => (
            <div className="card" key={horizon}>
              <div className="metric-label">{horizon.toUpperCase()} outcome</div>
              <div className="metric-value mono">{outcome(briefing.outcomes[horizon])}</div>
              <div className="metric-source">
                {briefing.outcomes.anchor_status === "no_session"
                  ? "No SPY session on briefing date"
                  : "Trading-day close-to-close return"}
              </div>
            </div>
          ))}
          <div className="card">
            <div className="metric-label">Data quality</div>
            <div className="metric-value" style={{ textTransform: "capitalize" }}>{briefing.data_quality.status}</div>
            <div className="metric-source">{briefing.data_quality.issues.join(" · ") || "All model inputs available"}</div>
          </div>
        </div>
        <div className="table-wrap" style={{ marginTop: 16 }}>
          <table>
            <thead><tr><th>Date</th><th>Posture</th><th className="num">Score</th><th className="num">1D</th><th className="num">5D</th><th className="num">20D</th></tr></thead>
            <tbody>
              {history.slice(0, 10).map((record) => (
                <tr key={record.as_of}>
                  <td className="mono">{record.as_of}</td>
                  <td style={{ textTransform: "capitalize" }}>{record.posture}</td>
                  <td className="num mono">{record.change.current_score ?? "—"}</td>
                  <td className="num mono">{outcome(record.outcomes["1d"])}</td>
                  <td className="num mono">{outcome(record.outcomes["5d"])}</td>
                  <td className="num mono">{outcome(record.outcomes["20d"])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
    </>
  );
}

function OverviewContent({ data, briefing, history }: { data: Overview; briefing: DailyBriefingRecord | null; history: DailyBriefingRecord[] }) {
  const composite = data.regime.composite_risk;
  return (
    <>
      <Topbar title="Daily Risk Briefing" subtitle="What changed, why it matters, and what to inspect next" freshness={data.freshness} />
      <div className="content">
        {data.warnings.length > 0 ? (
          <div className="notice"><span className="dot warn" />{data.warnings.join(" ")}</div>
        ) : null}

        <DailyBriefingPanel briefing={briefing} history={history} />

        <Section title="Underlying Risk Model" aside={<span className="mono">{fmtAsOf(data.as_of)}</span>}>
          <div className="regime">
            <div className="regime-cell lead">
              <span className="k">Composite Risk</span>
              <span className="regime-score mono">{composite ?? "—"}<span className="u"> / 100</span></span>
              <div className="regime-scale">
                {composite !== null ? <i style={{ left: `${Math.min(Math.max(composite, 0), 100)}%` }} /> : null}
              </div>
              <span className="note">
                {data.left_signal ? `LEFT signal: ${data.left_signal}` : "Aggregated risk indicators"}
              </span>
              {data.regime.breakdown ? <span className="note">{driversLine(data.regime.breakdown)}</span> : null}
            </div>
            {data.regime.components.map((c) => <RegimeCell key={c.key} c={c} />)}
          </div>
        </Section>

        <Section title="Key Indicators" aside="delayed · sourced">
          <div className="metrics">
            {data.metrics.map((m) => <MetricCard key={m.key} m={m} />)}
          </div>
        </Section>

        <section className="grid-2">
          <Panel title="Volatility Risk Premium" sub="180d">
            <AreaChart points={data.charts.vrp_history} color="var(--accent)" label="VRP" />
            <GapNotice points={data.charts.vrp_history} />
            <div className="legend">
              <span><i style={{ background: "var(--accent)" }} />VRP</span>
              <span style={{ color: "var(--ink-faint)" }}>Zero line = fair value · negative = rich realized vol</span>
            </div>
          </Panel>
          <Panel title="Credit Spreads" sub="1Y · %">
            <MultiLineChart series={[
              { points: data.charts.credit_spreads.hy, color: "var(--crit)", label: "HY spread" },
              { points: data.charts.credit_spreads.ig, color: "var(--accent)", label: "IG spread" },
            ]} />
            <GapNotice points={data.charts.credit_spreads.hy} />
            <div className="legend">
              <span><i style={{ background: "var(--crit)" }} />HY (BAMLH0A0HYM2)</span>
              <span><i style={{ background: "var(--accent)" }} />IG (BAMLC0A0CM)</span>
            </div>
          </Panel>
        </section>

        <Section title="Options & Volatility Detail">
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th>Indicator</th><th className="num">Value</th><th>State</th><th>Source</th></tr>
              </thead>
              <tbody>
                {data.detail.map((r) => <DetailTableRow key={r.indicator} r={r} />)}
              </tbody>
            </table>
          </div>
        </Section>
      </div>
    </>
  );
}

function ApiError({ message }: { message: string }) {
  return (
    <div className="content">
      <div className="panel" style={{ maxWidth: 560 }}>
        <div className="panel-head"><span className="t">Cannot reach the API</span></div>
        <p style={{ color: "var(--ink-muted)", fontSize: 13 }}>{message}</p>
        <p style={{ color: "var(--ink-faint)", fontSize: 12.5, marginTop: 10 }}>
          Start the backend: <span className="mono">uvicorn api.main:app --port 8000</span>
        </p>
      </div>
    </div>
  );
}

export default async function OverviewPage() {
  try {
    const { overview: data, briefing } = await loadOverviewPageData(getOverview, getDailyBriefing);
    return <OverviewContent data={data} briefing={briefing.latest} history={briefing.history} />;
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return <ApiError message={message} />;
  }
}
