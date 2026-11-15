import Topbar from "@/components/Topbar";
import SourceHealth from "@/components/SourceHealth";

export const dynamic = "force-dynamic";

// Health must remain available when snapshot freshness or another page fails.
export default function SystemHealthPage() {
  return (
    <>
      <Topbar title="System Health" subtitle="Data source status and freshness" />
      <div className="content">
        <SourceHealth expanded />
      </div>
    </>
  );
}
