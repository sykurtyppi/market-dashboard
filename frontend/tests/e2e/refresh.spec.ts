import { expect, test, type Page } from "@playwright/test";

type RefreshRun = {
  outcome: "successful" | "partial" | "failed";
  failed_phases: string[];
  error: string | null;
};

async function mockRefresh(page: Page, lastRun: RefreshRun) {
  await page.route("**/api/refresh", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ status: "started" }) }),
  );
  await page.route("**/api/refresh/status", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ running: false, last_run: lastRun }),
    }),
  );
}

test("shows confirmed success after the refresh API completes", async ({ page }) => {
  await mockRefresh(page, { outcome: "successful", failed_phases: [], error: null });
  await page.goto("/");

  await page.getByRole("button", { name: "Refresh" }).click();

  await expect(page.getByText("Data updated", { exact: true })).toBeVisible({ timeout: 10_000 });
  await expect(page.getByRole("button", { name: "Refresh" })).toBeEnabled();
});

test("shows a total failure without claiming data was updated", async ({ page }) => {
  await mockRefresh(page, { outcome: "failed", failed_phases: [], error: "database unavailable" });
  await page.goto("/");

  await page.getByRole("button", { name: "Refresh" }).click();

  await expect(page.getByText("Refresh failed: database unavailable", { exact: true })).toBeVisible({ timeout: 10_000 });
  await expect(page.getByText("Data updated", { exact: true })).toHaveCount(0);
});

test("shows degraded completion and names every failed phase", async ({ page }) => {
  await mockRefresh(page, {
    outcome: "partial",
    failed_phases: ["fred_indicators", "market_breadth"],
    error: null,
  });
  await page.goto("/");

  await page.getByRole("button", { name: "Refresh" }).click();

  await expect(
    page.getByText("Refresh partially completed — failed phases: fred_indicators, market_breadth", { exact: true }),
  ).toBeVisible({ timeout: 10_000 });
});