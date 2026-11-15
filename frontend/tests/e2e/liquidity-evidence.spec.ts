import { expect, test } from "@playwright/test";

const backend = `http://127.0.0.1:${process.env.E2E_BACKEND_PORT ?? 3101}`;
test.beforeEach(async ({ request }) => { await request.get(`${backend}/__test/liquidity/missing`); });
test.afterEach(async ({ request }) => { await request.get(`${backend}/__test/liquidity/missing`); });

test("fresh snapshot does not hide older strict evidence or backward-matched proxy inputs", async ({ page, request }) => {
  await request.get(`${backend}/__test/liquidity/backward`);
  await page.goto("/");
  await expect(page.getByText("Fresh · snapshot 0h")).toBeVisible();
  await expect(page.getByText(/Snapshot refresh does not establish underlying observation completeness/)).toBeVisible();
  const panel = page.getByRole("region", { name: "Liquidity Evidence" });
  await expect(panel).toBeVisible();
  const strict = panel.getByRole("group", { name: "Strict stored net liquidity" });
  await expect(strict).toContainText("5,600");
  await expect(strict).toContainText("2026-10-21");
  const proxy = panel.getByRole("group", { name: "Exploratory liquidity proxy" });
  await expect(proxy).toContainText("5,700");
  await expect(proxy).toContainText("2026-10-28");
  await expect(proxy).toContainText("Backward-matched");
  const inputs = panel.getByRole("table", { name: "Proxy component observations" });
  for (const [series, value, date] of [["WALCL", "6,600", "2026-10-28"], ["WTREGEN", "800", "2026-10-28"], ["RRPONTSYD", "100", "2026-10-27"]]) {
    const row = inputs.getByRole("row").filter({ hasText: series });
    await expect(row).toContainText(value);
    await expect(row).toContainText(date);
  }
  await expect(inputs.getByRole("row").filter({ hasText: "RRPONTSYD" })).toContainText("1 day");
  const latest = panel.getByRole("table", { name: "Latest component observations" });
  for (const [series, value, date] of [["WALCL", "6,610", "2026-10-29"], ["WTREGEN", "800", "2026-10-28"], ["RRPONTSYD", "110", "2026-10-29"]]) {
    const row = latest.getByRole("row").filter({ hasText: series });
    await expect(row).toContainText(value);
    await expect(row).toContainText(date);
  }
  await expect(panel).toContainText("3/3 components available");
  await expect(panel).toContainText("USD billions");
  await expect(panel).toContainText("WTREGEN is a weekly average");
  await expect(panel).toContainText("4 calendar-day");
  await expect(panel).toContainText("not calendar-aware");
  await expect(panel).toContainText("Publication/vintage eligibility is unknown");
  await expect(panel).toContainText("not a historical, trading, or backtest signal");
  await expect(panel).toContainText("Source-specific backend note.");
  await expect(page.locator(".metrics .card").filter({ hasText: "Net Liquidity" })).toContainText("Observation: 2026-10-21");
  await expect(page.locator(".card").filter({ hasText: "Unknown date metric" })).toContainText("Observation date unknown");
  await expect(page.getByText("Composite Risk", { exact: true })).toBeVisible();
});

test("exact proxy match is labeled distinctly", async ({ page, request }) => {
  await request.get(`${backend}/__test/liquidity/exact`);
  await page.goto("/");
  const proxy = page.getByRole("group", { name: "Exploratory liquidity proxy" });
  await expect(proxy).toContainText("Exact date match");
  await expect(proxy).not.toContainText("Backward-matched");
});

test("unavailable evidence preserves missing inputs and coverage without zero fabrication", async ({ page, request }) => {
  await request.get(`${backend}/__test/liquidity/unavailable`);
  await page.goto("/");
  const panel = page.getByRole("region", { name: "Liquidity Evidence" });
  await expect(panel.getByRole("group", { name: "Strict stored net liquidity" })).toContainText("Unavailable");
  await expect(panel.getByRole("group", { name: "Exploratory liquidity proxy" })).toContainText("No RRP observation within tolerance.");
  await expect(panel).toContainText("Observation date unknown");
  await expect(panel).toContainText("2/3 components available");
  for (const name of ["Proxy component observations", "Latest component observations"]) {
    const row = panel.getByRole("table", { name }).getByRole("row").filter({ hasText: "RRPONTSYD" });
    await expect(row).toContainText("Unavailable");
    await expect(row).toContainText("Unknown");
  }
});

test("partial evidence keeps strict value visible when proxy is unavailable", async ({ page, request }) => {
  await request.get(`${backend}/__test/liquidity/partial`);
  await page.goto("/");
  const panel = page.getByRole("region", { name: "Liquidity Evidence" });
  await expect(panel).toContainText("Partial");
  await expect(panel.getByRole("group", { name: "Strict stored net liquidity" })).toContainText("5,600");
  await expect(panel.getByRole("group", { name: "Exploratory liquidity proxy" })).toContainText("RRP is outside tolerance.");
  await expect(panel.getByRole("group", { name: "Exploratory liquidity proxy" })).toContainText("2026-10-28");
  await expect(panel).toContainText("Candidate observations only; no proxy was calculated. Observed coverage does not establish matching eligibility.");
  await expect(panel).not.toContainText("Actual inputs used for the exploratory proxy");
  const rrp = panel.getByRole("table", { name: "Proxy component observations" }).getByRole("row").filter({ hasText: "RRPONTSYD" });
  await expect(rrp).toContainText("100");
  await expect(rrp).toContainText("2026-10-21");
  await expect(rrp).toContainText("7 days");
});

test("zero values remain visible in both estimates and component tables on mobile", async ({ page, request }) => {
  await request.get(`${backend}/__test/liquidity/zero`);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  const panel = page.getByRole("region", { name: "Liquidity Evidence" });
  for (const name of ["Strict stored net liquidity", "Exploratory liquidity proxy"]) {
    await expect(panel.getByRole("group", { name }).getByText("0 USD billions", { exact: true })).toBeVisible();
  }
  for (const name of ["Proxy component observations", "Latest component observations"]) {
    const row = panel.getByRole("table", { name }).getByRole("row").filter({ hasText: "RRPONTSYD" });
    await expect(row.getByRole("cell", { name: "0", exact: true })).toBeVisible();
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)).toBe(true);
});

test("older API without liquidity explicitly reports unknown evidence", async ({ page }) => {
  await page.goto("/");
  const panel = page.getByRole("region", { name: "Liquidity Evidence" });
  await expect(panel).toContainText("Liquidity evidence unavailable");
  await expect(panel).toContainText("Observation dates and component coverage are unknown");
  await expect(panel.getByRole("table")).toHaveCount(0);
  await expect(page.getByText("Composite Risk", { exact: true })).toBeVisible();
});
