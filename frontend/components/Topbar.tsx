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
        <span className="freshness">
          <span className={`dot ${dot}`} />
          {label} · updated {freshness.age}
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
