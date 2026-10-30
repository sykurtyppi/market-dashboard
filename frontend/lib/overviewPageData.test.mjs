import assert from "node:assert/strict";
import test from "node:test";

import { loadOverviewPageData } from "./overviewPageData.mjs";


test("briefing failure preserves valid overview data", async () => {
  const overview = { latest: { date: "2026-10-29" } };
  const result = await loadOverviewPageData(
    async () => overview,
    async () => { throw new Error("briefing unavailable"); },
  );

  assert.deepEqual(result.overview, overview);
  assert.deepEqual(result.briefing, { latest: null, history: [] });
});


test("overview failure remains a page-level failure", async () => {
  await assert.rejects(
    loadOverviewPageData(
      async () => { throw new Error("overview unavailable"); },
      async () => ({ latest: null, history: [] }),
    ),
    /overview unavailable/,
  );
});
