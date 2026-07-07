import { Freshness } from "@/lib/api";
import RefreshButton from "@/components/RefreshButton";

function stateFromFreshness(f: Freshness): "good" | "warn" | "crit" {
  if (f.is_fresh) return "good";
  if (f.status === "stale") return "warn";
  return "crit";
}

interface TopbarProps {
  title: string;
  subtitle: string;
  freshness: Freshness;
}

export default function Topbar({ title, subtitle, freshness }: TopbarProps) {
  const dot = stateFromFreshness(freshness);
  const label = freshness.is_fresh ? "Fresh" : freshness.status === "stale" ? "Stale" : "Outdated";
  return (
    <header className="topbar">
      <div>
        <h1>{title}</h1>
        <div className="crumb">{subtitle}</div>
      </div>
      <div className="topbar-right">
        {/* This pill reports the age of the latest daily snapshot; individual
            sources can be older or newer — System Health has per-source ages.
            The title makes that scope explicit so the two never look like
            contradictory claims. */}
        <span
          className="freshness"
          title="Age of the latest daily data snapshot. Some panels fetch live; per-source ages are on System Health."
        >
          <span className={`dot ${dot}`} />
          {label} · snapshot {freshness.age}
        </span>
        <RefreshButton />
        <button className="btn" type="button" disabled title="Coming in a later phase">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 3v12M8 11l4 4 4-4M5 21h14" /></svg>
          Export PDF
        </button>
      </div>
    </header>
  );
}
