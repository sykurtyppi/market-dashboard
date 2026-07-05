import { Freshness } from "@/lib/api";

function stateFromFreshness(f: Freshness): "good" | "warn" | "crit" {
  if (f.is_fresh) return "good";
  if (f.status === "stale") return "warn";
  return "crit";
}

export default function Topbar({ freshness }: { freshness: Freshness }) {
  const dot = stateFromFreshness(freshness);
  const label = freshness.is_fresh ? "Fresh" : freshness.status === "stale" ? "Stale" : "Outdated";
  return (
    <header className="topbar">
      <div>
        <h1>Overview</h1>
        <div className="crumb">Market regime &amp; key risk indicators</div>
      </div>
      <div className="topbar-right">
        <span className="freshness">
          <span className={`dot ${dot}`} />
          {label} · updated {freshness.age}
        </span>
        <button className="btn ghost" type="button">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 12a9 9 0 019-9 9 9 0 016.7 3M21 12a9 9 0 01-9 9 9 9 0 01-6.7-3" /><path d="M21 3v5h-5M3 21v-5h5" /></svg>
          Refresh
        </button>
        <button className="btn" type="button">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 3v12M8 11l4 4 4-4M5 21h14" /></svg>
          Export PDF
        </button>
      </div>
    </header>
  );
}
