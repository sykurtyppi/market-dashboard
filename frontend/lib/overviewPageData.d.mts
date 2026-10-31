export declare function loadOverviewPageData<TOverview, TBriefing>(
  fetchOverview: () => Promise<TOverview>,
  fetchBriefing: () => Promise<TBriefing>,
): Promise<{
  overview: TOverview;
  briefing: TBriefing | { latest: null; history: never[] };
}>;
