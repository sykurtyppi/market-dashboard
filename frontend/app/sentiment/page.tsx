import { getSentiment, getFreshness, Sentiment, Freshness } from "@/lib/api";
import Topbar from "@/components/Topbar";
import { Section } from "@/components/ui";

export const dynamic = "force-dynamic";

function Content({ data, freshness }: { data: Sentiment; freshness: Freshness }) {
  const fg = data.fear_greed;
  const score = fg.score;
  return (
    <>
      <Topbar title="Sentiment" subtitle="Fear & Greed and options positioning" freshness={freshness} />
      <div className="content">
        {data.warnings.length > 0 ? (
          <div className="notice"><span className="dot warn" />{data.warnings.join(" ")}</div>
        ) : null}

        <Section title="Market Sentiment" aside={<span className="mono">{data.as_of ? String(data.as_of).slice(0, 10) : "—"}</span>}>
          <div className="regime" style={{ gridTemplateColumns: "1.4fr 1fr 1fr" }}>
            <div className="regime-cell lead">
              <span className="k">Fear &amp; Greed</span>
              <span className="regime-score mono">
                {score === null ? "—" : score.toFixed(0)}<span className="u"> / 100</span>
              </span>
              <div className="regime-scale">
                {score !== null ? <i style={{ left: `${Math.min(Math.max(score, 0), 100)}%` }} /> : null}
              </div>
              <span className="note"><span className={`dot ${fg.state}`} />{fg.rating ?? "—"}</span>
            </div>
            <div className="regime-cell">
              <span className="k">Put / Call Ratio</span>
              <span className="v mono" style={{ fontSize: 20 }}>
                {data.put_call_ratio === null ? "—" : data.put_call_ratio.toFixed(3)}
              </span>
              <span className="note">{data.put_call_source ? `${data.put_call_source} P/C` : "Put/Call ratio"}</span>
            </div>
            <div className="regime-cell">
              <span className="k">Read</span>
              <span className="note" style={{ marginTop: 10, fontSize: 12.5 }}>
                {score === null ? "—"
                  : score < 25 ? "Extreme fear — historically a contrarian buy zone."
                  : score < 45 ? "Fear — cautious sentiment."
                  : score <= 60 ? "Neutral sentiment."
                  : score <= 78 ? "Greed — risk appetite elevated."
                  : "Extreme greed — froth risk."}
              </span>
            </div>
          </div>
        </Section>
      </div>
    </>
  );
}

export default async function SentimentPage() {
  try {
    const [data, freshness] = await Promise.all([getSentiment(), getFreshness()]);
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
