import { expect, test } from "@playwright/test";

const mobileViewport = { width: 390, height: 844 };

test.describe("mobile navigation", () => {
  test.use({ viewport: mobileViewport });

  test("opens accessibly, keeps the active route visible, and closes on Escape", async ({ page }) => {
    await page.goto("/options-flow");

    const trigger = page.getByRole("button", { name: "Open navigation" });
    await expect(trigger).toBeVisible();
    await expect(trigger).toHaveAttribute("aria-expanded", "false");
    await expect(trigger).toHaveAttribute("aria-controls", "mobile-navigation");

    await trigger.focus();
    await page.keyboard.press("Enter");

    await expect(trigger).toHaveAttribute("aria-expanded", "true");
    const drawer = page.getByRole("dialog", { name: "Navigation" });
    await expect(drawer).toBeVisible();
    await expect(drawer.getByText("Positioning", { exact: true })).toBeVisible();
    const activeRoute = drawer.getByRole("link", { name: "Options Flow" });
    await expect(activeRoute).toHaveAttribute("aria-current", "page");
    await expect(activeRoute).toBeInViewport();
    await expect(drawer.locator(":focus")).toHaveCount(1);

    const drawerLinks = drawer.getByRole("link");
    await drawerLinks.last().focus();
    await page.keyboard.press("Tab");
    await expect(drawerLinks.first()).toBeFocused();

    await page.keyboard.press("Escape");

    await expect(drawer).toBeHidden();
    await expect(trigger).toHaveAttribute("aria-expanded", "false");
    await expect(trigger).toBeFocused();
  });

  test("closes after selecting a route and returns focus to the trigger", async ({ page }) => {
    await page.goto("/options-flow");

    const trigger = page.getByRole("button", { name: "Open navigation" });
    await trigger.click();
    const drawer = page.getByRole("dialog", { name: "Navigation" });
    await drawer.getByRole("link", { name: "Cross-Asset" }).click();

    await expect(page).toHaveURL(/\/cross-asset$/);
    await expect(drawer).toBeHidden();
    await expect(trigger).toHaveAttribute("aria-expanded", "false");
    await expect(trigger).toBeFocused();
  });

  test("backdrop below the topbar isolates the page, closes the drawer, and restores focus", async ({ page }) => {
    await page.goto("/options-flow");

    const trigger = page.getByRole("button", { name: "Open navigation" });
    const content = page.locator(".content");
    await content.evaluate((element) => {
      element.addEventListener("click", () => {
        document.documentElement.dataset.underlyingClicks = String(
          Number(document.documentElement.dataset.underlyingClicks ?? "0") + 1,
        );
      });
    });

    await trigger.click();
    const drawer = page.getByRole("dialog", { name: "Navigation" });
    await expect(drawer).toBeVisible();

    const geometry = await page.locator(".mobile-nav-layer").evaluate((layer) => {
      const layerRect = layer.getBoundingClientRect();
      const backdropRect = layer.querySelector(".mobile-nav-backdrop")?.getBoundingClientRect();
      return {
        parent: layer.parentElement?.tagName,
        layer: { x: layerRect.x, y: layerRect.y, width: layerRect.width, height: layerRect.height },
        backdrop: backdropRect
          ? { x: backdropRect.x, y: backdropRect.y, width: backdropRect.width, height: backdropRect.height }
          : null,
      };
    });
    expect(geometry).toEqual({
      parent: "BODY",
      layer: { x: 0, y: 0, width: mobileViewport.width, height: mobileViewport.height },
      backdrop: { x: 0, y: 0, width: mobileViewport.width, height: mobileViewport.height },
    });

    await page.mouse.click(370, 400);

    await expect(drawer).toBeHidden();
    await expect.poll(() => page.evaluate(() => document.documentElement.dataset.underlyingClicks ?? "0")).toBe("0");
    await expect(trigger).toBeFocused();
  });

  test("releases the modal and scroll lock when resized to desktop", async ({ page }) => {
    await page.goto("/options-flow");
    await page.getByRole("button", { name: "Open navigation" }).click();
    await expect(page.getByRole("dialog", { name: "Navigation" })).toBeVisible();

    await page.setViewportSize({ width: 1024, height: 844 });

    await expect(page.getByRole("dialog", { name: "Navigation" })).toBeHidden();
    await expect.poll(() => page.evaluate(() => document.body.style.overflow)).not.toBe("hidden");
  });
});

test.describe("mobile responsive containment", () => {
  test.use({ viewport: mobileViewport });

  for (const route of ["/options-flow", "/institutional", "/cross-asset", "/volatility"]) {
    test(`${route} does not overflow the document horizontally`, async ({ page }) => {
      await page.goto(route);
      await expect(page.locator("body")).toBeVisible();

      const dimensions = await page.evaluate(() => ({
        documentWidth: document.documentElement.scrollWidth,
        viewportWidth: document.documentElement.clientWidth,
      }));

      expect(dimensions.documentWidth).toBeLessThanOrEqual(dimensions.viewportWidth);
    });
  }

  test("long refresh status remains contained in the mobile topbar", async ({ page }) => {
    await page.goto("/options-flow");
    await page.locator(".refresh-wrap").evaluate((element) => {
      const note = document.createElement("span");
      note.className = "refresh-note";
      note.textContent = "Refresh failed in an unusually long backend phase with diagnostic details";
      element.append(note);
    });

    const dimensions = await page.evaluate(() => ({
      documentWidth: document.documentElement.scrollWidth,
      viewportWidth: document.documentElement.clientWidth,
    }));
    expect(dimensions.documentWidth).toBeLessThanOrEqual(dimensions.viewportWidth);
  });
});
