import { getEconomicCalendar, getFreshness, EconomicCalendar, EconomicEvent, Freshness } from "@/lib/api";
import Topbar from "@/components/Topbar";
import Explainer from "@/components/Explainer";
import { Section, Panel, fmtAsOf } from "@/components/ui";

export const dynamic = "force-dynamic";

function importanceDot(importance: string | null): string {
  const i = (importance ?? "").toLowerCase();
  if (i === "high") return "crit";
  if (i === "medium") return "warn";
  return "neutral";
}

// The backend's actual/previous fields are raw *levels* from the prior release
// (CPI index ~334, GDP in $B) — not the % change the event's canonical unit
// ("% YoY", "% QoQ") describes. Labeling 333.98 as "% YoY" was flatly wrong,
// and "Actual" for a future release is a contradiction. Show levels as levels
// ("Latest"/"Prior"), and the % change in its own column (yoy_change is always
// year-over-year, whatever the event's headline cadence).
function level(v: number | null, unit: string | null): string {
  if (v === null) return "—";
  // A unit of exactly "%" means the value itself is a rate (e.g. FOMC 3.63%).
  return unit === "%" ? `${v.toFixed(2)}%` : v.toLocaleString();
}

function yoy(v: number | null): string {
  if (v === null) return "—";
  return `${v > 0 ? "+" : ""}${v.toFixed(2)}%`;
}

function EventRow({ e }: { e: EconomicEvent }) {
  return (
    <tr>
      <td><span className="tag"><span className={`dot ${importanceDot(e.importance)}`} />{e.name}</span></td>
      <td className="mono">{e.date}</td>
      <td className="mono">{e.days_until !== null ? `${e.days_until}d` : "—"}</td>
      <td className="num mono">{level(e.actual, e.unit)}</td>
      <td className="num mono">{level(e.previous, e.unit)}</td>
      <td className="num mono">{yoy(e.yoy_change)}</td>
    </tr>
  );
}

function Content({ data, freshness }: { data: EconomicCalendar; freshness: Freshness }) {
  return (
    <>
      <Topbar title="Economic Calendar" subtitle="Upcoming macro releases" freshness={freshness} />
      <div className="content">
        {data.warnings.length > 0 ? (
          <div className="notice"><span className="dot warn" />{data.warnings.join(" ")}</div>
        ) : null}

        <Section title="Upcoming Events" aside={<span className="mono">{fmtAsOf(data.as_of)}</span>}>
          <Panel title="Releases & Meetings">
            <div className="table-wrap" style={{ border: "none" }}>
              <table>
                <thead>
                  <tr>
                    <th>Event</th><th>Date</th><th>In</th>
                    <th className="num">Latest</th><th className="num">Prior</th><th className="num">Δ YoY</th>
                  </tr>
                </thead>
                <tbody>
                  {data.events.length > 0 ? (
                    data.events.map((e, i) => <EventRow key={`${e.name}-${i}`} e={e} />)
                  ) : (
                    <tr><td colSpan={6}><div className="empty-state">Economic calendar unavailable.</div></td></tr>
                  )}
                </tbody>
              </table>
            </div>
            <div className="legend">
              <span><i style={{ background: "var(--crit)" }} />High</span>
              <span><i style={{ background: "var(--warn)" }} />Medium</span>
              <span><i style={{ background: "var(--ink-faint)" }} />Low</span>
            </div>
          </Panel>
        </Section>
        <Explainer
          title="How to read this — the calendar"
          intro="Upcoming macro releases that move rates and risk assets."
          points={[
            { label: "Latest / Prior:", text: "the raw released levels — e.g. the CPI index value itself, not a percentage." },
            { label: "Δ YoY:", text: "the year-over-year change — the number the headlines quote." },
            { label: "Importance:", text: "red = typically market-moving, amber = notable, grey = background." },
          ]}
          caveat="Dates are scheduled release dates and can shift."
        />
      </div>
    </>
  );
}

export default async function EconomicCalendarPage() {
  try {
    const [data, freshness] = await Promise.all([getEconomicCalendar(), getFreshness()]);
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
