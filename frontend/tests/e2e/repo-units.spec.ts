import { expect, test } from "@playwright/test";

test("renders FRED RRPONTSYD in billions without rescaling", async ({ page }) => {
  await page.goto("/repo");

  const rrpCard = page.getByText("RRP (Overnight)").locator("..");
  await expect(rrpCard).toContainText("82.50B");
  await expect(rrpCard).toContainText("FRED (RRPONTSYD), $B");
  await expect(page.getByText("82,500.00B")).toHaveCount(0);
});