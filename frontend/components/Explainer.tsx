import type { CSSProperties, ReactNode } from "react";

// Collapsible "How to read this" for non-experts — the new-frontend equivalent
// of the old Streamlit guide expanders. Native <details> so it's accessible and
// needs no JS (server-safe); inline-styled to survive a stale dev stylesheet.
// One structured shape (intro → labeled points → caveat) keeps every page's
// explainer consistent.

const term: CSSProperties = { color: "var(--ink)", fontWeight: 600 };

export function Term({ children }: { children: ReactNode }) {
  return <span style={term}>{children}</span>;
}

interface ExplainerProps {
  title: string;
  intro?: ReactNode;
  points?: { label: ReactNode; text: ReactNode }[];
  caveat?: ReactNode;
}

export default function Explainer({ title, intro, points, caveat }: ExplainerProps) {
  return (
    <details style={{ border: "1px solid var(--line)", borderRadius: 9, background: "var(--surface)" }}>
      <summary style={{ cursor: "pointer", padding: "10px 14px", fontSize: 12.5, fontWeight: 600, color: "var(--ink)" }}>
        {title}
      </summary>
      <div style={{ padding: "2px 14px 14px", fontSize: 12.5, color: "var(--ink-muted)", lineHeight: 1.6 }}>
        {intro ? <p style={{ margin: "0 0 10px" }}>{intro}</p> : null}
        {points && points.length > 0 ? (
          <ul style={{ margin: "0 0 8px", paddingLeft: 18, display: "flex", flexDirection: "column", gap: 5 }}>
            {points.map((p, i) => (
              <li key={i}><Term>{p.label}</Term> {p.text}</li>
            ))}
          </ul>
        ) : null}
        {caveat ? (
          <p style={{ margin: 0, color: "var(--ink-faint)", fontSize: 11.5 }}>{caveat}</p>
        ) : null}
      </div>
    </details>
  );
}
