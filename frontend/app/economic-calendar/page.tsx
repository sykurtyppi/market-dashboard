import { getEconomicCalendar, getFreshness, EconomicCalendar, EconomicEvent, Freshness } from "@/lib/api";
import Topbar from "@/components/Topbar";
import { Section, Panel } from "@/components/ui";

export const dynamic = "force-dynamic";

function importanceDot(importance: string | null): string {
  const i = (importance ?? "").toLowerCase();
  if (i === "high") return "crit";
  if (i === "medium") return "warn";
  return "neutral";
}

function val(v: number | null, unit: string | null): string {
  if (v === null) return "—";
  return unit === "% YoY" || unit === "%" ? `${v.toFixed(2)}` : v.toLocaleString();
}

function EventRow({ e }: { e: EconomicEvent }) {
  return (
    <tr>
      <td><span className="tag"><span className={`dot ${importanceDot(e.importance)}`} />{e.name}</span></td>
      <td className="mono">{e.date}</td>
      <td className="mono">{e.days_until !== null ? `${e.days_until}d` : "—"}</td>
      <td className="num mono">{val(e.actual, e.unit)}</td>
      <td className="num mono">{val(e.forecast, e.unit)}</td>
      <td className="num mono">{val(e.previous, e.unit)}</td>
      <td className="src">{e.unit}</td>
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

        <Section title="Upcoming Events" aside={<span className="mono">{data.as_of ? String(data.as_of).slice(0, 10) : "—"}</span>}>
          <Panel title="Releases & Meetings">
            <div className="table-wrap" style={{ border: "none" }}>
              <table>
                <thead>
                  <tr>
                    <th>Event</th><th>Date</th><th>In</th>
                    <th className="num">Actual</th><th className="num">Forecast</th><th className="num">Previous</th><th>Unit</th>
                  </tr>
                </thead>
                <tbody>
                  {data.events.length > 0 ? (
                    data.events.map((e, i) => <EventRow key={`${e.name}-${i}`} e={e} />)
                  ) : (
                    <tr><td colSpan={7}><div className="empty-state">Economic calendar unavailable.</div></td></tr>
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
