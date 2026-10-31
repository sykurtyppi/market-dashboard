export async function loadOverviewPageData(fetchOverview, fetchBriefing) {
  const [overview, briefing] = await Promise.all([
    fetchOverview(),
    fetchBriefing().catch(() => ({ latest: null, history: [] })),
  ]);
  return { overview, briefing };
}
