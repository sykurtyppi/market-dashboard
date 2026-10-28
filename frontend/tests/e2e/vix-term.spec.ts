import { expect, test } from "@playwright/test";

test("VIX page presents a full, source-labeled index curve", async ({ page }) => {
  await page.goto("/sectors");

  await expect(page.getByRole("heading", { name: "VIX Index Term Structure" })).toBeVisible();
  await expect(page.getByText("Index closes — not VIX futures")).toBeVisible();
  await expect(page.getByText("Yahoo Finance · Cboe volatility index closes")).toBeVisible();
  await expect(page.getByText("Upward sloping")).toBeVisible();

  for (const symbol of ["VIX9D", "VIX", "VIX3M", "VIX6M", "VIX1Y"]) {
    await expect(page.getByText(symbol, { exact: true })).toBeVisible();
  }

  await expect(page.getByRole("columnheader", { name: "Vs prior" })).toBeVisible();
  await expect(page.getByText("+9.46%")).toBeVisible();
});

test("VIX chart preserves a visible gap when an interior tenor is unavailable", async ({ page, request }) => {
  await request.get("http://127.0.0.1:3101/__test/sectors-partial");
  try {
    await page.goto("/sectors");

    await expect(page.getByText("VIX3M unavailable; adjacent curve comparison suppressed.")).toBeVisible();
    await expect(page.locator("polyline.vix-curve-line")).toHaveCount(2);
    const sixMonthRow = page.getByRole("row").filter({ hasText: "VIX6M" });
    await expect(sixMonthRow).toContainText("—");
  } finally {
    await request.get("http://127.0.0.1:3101/__test/sectors-full");
  }
});

test("VIX panel contains its scrollable table on a mobile viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/sectors");

  const dimensions = await page.evaluate(() => ({
    viewport: window.innerWidth,
    document: document.documentElement.scrollWidth,
  }));
  expect(dimensions.document).toBeLessThanOrEqual(dimensions.viewport);
  await expect(page.getByText("Index closes — not VIX futures")).toBeVisible();
});
