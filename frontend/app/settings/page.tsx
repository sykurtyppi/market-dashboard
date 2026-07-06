import { getSettings, getFreshness, Settings, Freshness } from "@/lib/api";
import Topbar from "@/components/Topbar";
import { Section } from "@/components/ui";

export const dynamic = "force-dynamic";

const SOURCE_LABEL: Record<string, string> = {
  environment: "Environment",
  streamlit_secrets: "Secrets store",
};

function Content({ data, freshness }: { data: Settings; freshness: Freshness }) {
  return (
    <>
      <Topbar title="Settings" subtitle="Configuration and credentials" freshness={freshness} />
      <div className="content">
        {data.warnings.length > 0 ? (
          <div className="notice"><span className="dot warn" />{data.warnings.join(" ")}</div>
        ) : null}

        <Section
          title="Credentials"
          aside={<span className="mono">{data.protected ? "Endpoint token-protected" : "Endpoint open"}</span>}
        >
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Data source key</th>
                  <th>Status</th>
                  <th>Source</th>
                </tr>
              </thead>
              <tbody>
                {data.credentials.map((c) => (
                  <tr key={c.name}>
                    <td className="mono">{c.name}</td>
                    <td>
                      <span className="tag">
                        <span className={`dot ${c.configured ? "good" : "neutral"}`} />
                        {c.configured ? "Configured" : "Not set"}
                      </span>
                    </td>
                    <td className="src">{c.source ? SOURCE_LABEL[c.source] ?? c.source : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="src" style={{ marginTop: 10 }}>
            Values are never exposed. Credentials are managed via environment variables or the
            deployment secret store — not editable from this page.
          </p>
        </Section>

        {data.config.map((group) => (
          <Section title={group.title} key={group.title}>
            <div className="table-wrap">
              <table>
                <tbody>
                  {group.items.map((item) => (
                    <tr key={item.label}>
                      <td style={{ color: "var(--ink-muted)", width: "40%" }}>{item.label}</td>
                      <td className="mono">{item.value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        ))}
      </div>
    </>
  );
}

export default async function SettingsPage() {
  try {
    const [data, freshness] = await Promise.all([getSettings(), getFreshness()]);
    return <Content data={data} freshness={freshness} />;
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Unknown error";
    const unauthorized = message.includes("401");
    return (
      <div className="content">
        <div className="panel" style={{ maxWidth: 560 }}>
          <div className="panel-head">
            <span className="t">{unauthorized ? "Settings are protected" : "Cannot reach the API"}</span>
          </div>
          <p style={{ color: "var(--ink-muted)", fontSize: 13 }}>
            {unauthorized
              ? "This deployment requires a valid API token to read settings. Set MARKET_API_TOKEN on the frontend server to match the backend."
              : message}
          </p>
        </div>
      </div>
    );
  }
}
