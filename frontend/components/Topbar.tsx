import { Freshness } from "@/lib/api";
import RefreshButton from "@/components/RefreshButton";
import MobileNavigation from "@/components/MobileNavigation";
import SourceHealth from "@/components/SourceHealth";

interface TopbarProps {
  title: string;
  subtitle: string;
  freshness?: Freshness;
}

export default function Topbar({ title, subtitle, freshness }: TopbarProps) {
  return (
    <header className="topbar">
      <MobileNavigation />
      <div className="topbar-title">
        <h1>{title}</h1>
        <div className="crumb">{subtitle}</div>
      </div>
      <div className="topbar-right">
        <SourceHealth snapshot={freshness} />
        <RefreshButton />
        <button className="btn export-btn" type="button" disabled title="Coming in a later phase">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 3v12M8 11l4 4 4-4M5 21h14" /></svg>
          Export PDF
        </button>
      </div>
    </header>
  );
}
