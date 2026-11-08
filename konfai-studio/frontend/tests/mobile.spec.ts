// SPDX-License-Identifier: Apache-2.0
//
// A phone shows one pane at a time: the pane bar reaches the conversation, the workspace and Scout,
// the composer stays in view, and nothing makes the page wider than the screen.
import { expect, test } from "@playwright/test";
import { studioRoutes } from "./studio-fixtures";

test("on a phone the shell is one column with a pane bar and no horizontal scroll", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await studioRoutes(page);
  await page.goto("/tests/harness.html");
  await page.evaluate(() => (window as any).__mount("App"));

  const bar = page.getByRole("navigation", { name: "Panels" });
  await expect(bar).toBeVisible();
  await expect(page.getByPlaceholder(/Describe a task/)).toBeVisible();
  await expect(page.getByRole("button", { name: "Show experiments" })).toBeVisible();

  await bar.getByRole("button", { name: "Workspace" }).click();
  await expect(page.locator(".rpanel")).toBeVisible();
  await expect(page.getByPlaceholder(/Describe a task/)).toBeHidden();
  await bar.getByRole("button", { name: "Scout" }).click();
  await expect(page.getByText(/Visual inspections/i)).toBeVisible();
  await bar.getByRole("button", { name: "Conversation" }).click();
  await expect(page.getByPlaceholder(/Describe a task/)).toBeVisible();

  const widths = await page.evaluate(() => ({ scroll: document.documentElement.scrollWidth, inner: window.innerWidth }));
  expect(widths.scroll).toBeLessThanOrEqual(widths.inner);

  await page.getByRole("button", { name: "Show experiments" }).click();
  await expect(page.getByRole("button", { name: "Lung study" })).toBeVisible();
});

test("on a phone, hiding the terminal sheet keeps its shell", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await studioRoutes(page);
  let shellConnections = 0;
  await page.routeWebSocket("**/api/terminal", (socket) => {
    shellConnections += 1;
    socket.onMessage(() => undefined);
  });
  await page.goto("/tests/harness.html");
  await page.evaluate(() => (window as any).__mount("App"));

  const bar = page.getByRole("navigation", { name: "Panels" });
  await bar.getByRole("button", { name: "Open the terminal" }).click();
  await expect(page.locator(".term-sheet .cst")).toHaveText("connected");
  const opened = shellConnections;
  await bar.getByRole("button", { name: "Close the terminal" }).click();
  await expect(page.locator(".term-sheet")).toBeHidden();
  await bar.getByRole("button", { name: "Open the terminal" }).click();
  await expect(page.locator(".term-sheet .xterm")).toBeVisible();
  await expect(page.locator(".term-sheet .cst")).toHaveText("connected");
  expect(shellConnections).toBe(opened);
});
