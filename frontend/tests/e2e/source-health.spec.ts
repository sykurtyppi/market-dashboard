import { expect, test } from "@playwright/test";

test("public proxy and System Health work independently of failed snapshot freshness", async ({ page, request }) => {
  const backend = `http://127.0.0.1:${process.env.PLAYWRIGHT_API_PORT ?? 3101}`;
  try {
    await request.get(`${backend}/__test/health?freshness=unavailable`);
    expect((await request.get(`${backend}/api/freshness`)).status()).toBe(500);
    const response = await request.get("/api/source-health");
    expect(response.status()).toBe(200);
    expect(response.headers()["cache-control"]).toBe("no-store");
    const payload = await response.json();
    expect(payload.overall_status).toBe("healthy");
    expect(payload.sources.vix.age_hours).toBe(60);
    expect(JSON.stringify(payload)).not.toContain("internal_metadata");
    await page.goto("/system-health");
    const panel = page.getByRole("region", { name: "Data source health" });
    await expect(panel.getByText("Data current (last: 2026-11-13)", { exact: true })).toBeVisible();
    for (const mode of ["invalid", "down"]) {
      await request.get(`${backend}/__test/health?mode=${mode}`);
      const failure = await request.get("/api/source-health");
      expect(failure.status()).toBe(502);
      expect(await failure.json()).toEqual({ detail: "Source health unavailable" });
    }
    expect((await request.post("/api/source-health")).status()).toBe(405);
  } finally {
    await request.get(`${backend}/__test/health`);
  }
});

const healthy = {
  overall_status: "healthy", timestamp: "2026-11-15T12:00:00", sources: {
    vix: { name: "VIX", status: "healthy", last_update: "2026-11-13", age_hours: 60, message: "Data current (last: 2026-11-13)" },
  },
};
const mixed = { ...healthy, overall_status: "stale", sources: {
  ...healthy.sources,
  hy: { name: "Credit Spread (HY)", status: "stale", last_update: "2026-11-11", age_hours: 108, message: "Data stale (2 business days behind)" },
  rrp: { name: "Fed RRP", status: "unknown", last_update: null, age_hours: null, message: "No data available" },
} };

test("weekend badge uses backend health, expands and filters source reasons, and retries read-only", async ({ page }) => {
  let reads = 0;
  let showMixed = false;
  const writes: string[] = [];
  page.on("request", (request) => { if (request.method() !== "GET") writes.push(request.url()); });
  await page.route("**/api/source-health", (route) => {
    reads += 1;
    return route.fulfill({ json: showMixed ? mixed : healthy });
  });
  await page.goto("/");
  const toggle = page.getByRole("button", { name: "Source health: Healthy" });
  await expect(toggle).toBeVisible();
  await toggle.focus();
  await page.keyboard.press("Enter");
  await expect(toggle).toHaveAttribute("aria-expanded", "true");
  const panel = page.getByRole("region", { name: "Source health details" });
  await expect(panel.getByText("Data current (last: 2026-11-13)", { exact: true })).toBeVisible();
  await expect(panel.getByText("60.0h elapsed")).toBeVisible();
  const initialReads = reads; // Dev StrictMode can cancel/restart the initial read.
  showMixed = true;
  await panel.getByRole("button", { name: "Recheck health" }).click();
  await expect(page.getByRole("button", { name: "Source health: Stale" })).toBeVisible();
  await panel.getByLabel("Source status", { exact: true }).selectOption("attention");
  await expect(panel.getByText("VIX", { exact: true })).toHaveCount(0);
  await panel.getByLabel("Find a source").fill("business");
  await expect(panel.getByText("Credit Spread (HY)", { exact: true })).toBeVisible();
  await expect(panel.getByText("Fed RRP", { exact: true })).toHaveCount(0);
  await panel.getByLabel("Find a source").fill("no match");
  await expect(panel.getByText("No sources match these filters.")).toBeVisible();
  await panel.getByRole("button", { name: "Clear filters" }).click();
  await expect(panel.getByText("Fed RRP", { exact: true })).toBeVisible();
  expect(reads).toBe(initialReads + 1);
  expect(writes).toEqual([]);
  await page.keyboard.press("Escape");
  await expect(panel).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Source health: Stale" })).toBeFocused();
});

test("failed recheck retains last-known health without claiming it is current; retry recovers", async ({ page }) => {
  let unavailable = false;
  await page.route("**/api/source-health", (route) => unavailable
    ? route.fulfill({ status: 503, json: { detail: "unavailable" } }) : route.fulfill({ json: mixed }));
  await page.goto("/");
  await page.getByRole("button", { name: "Source health: Stale" }).click();
  const panel = page.getByRole("region", { name: "Source health details" });
  unavailable = true;
  await panel.getByRole("button", { name: "Recheck health" }).click();
  await expect(panel.getByRole("alert")).toContainText("Last successful check shown");
  await expect(page.getByRole("button", { name: "Source health: Unavailable" })).toBeVisible();
  await expect(panel.getByText("Credit Spread (HY)", { exact: true })).toBeVisible();
  unavailable = false;
  await panel.getByRole("button", { name: "Retry health" }).click();
  await expect(page.getByRole("button", { name: "Source health: Stale" })).toBeVisible();
  await expect(panel.getByRole("alert")).toHaveCount(0);
});

test("initial health failure is explicit and can recover at mobile width without overflow", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  let unavailable = true;
  await page.route("**/api/source-health", (route) => unavailable
    ? route.fulfill({ status: 502, json: { detail: "unavailable" } })
    : route.fulfill({ json: mixed }));
  await page.goto("/");
  await page.getByRole("button", { name: "Source health: Unavailable" }).click();
  const panel = page.getByRole("region", { name: "Source health details" });
  await expect(panel.getByRole("alert")).toContainText("Could not check source health");
  unavailable = false;
  await panel.getByRole("button", { name: "Retry health" }).click();
  await expect(page.getByRole("button", { name: "Source health: Stale" })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});
