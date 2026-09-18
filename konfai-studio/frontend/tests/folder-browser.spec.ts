// SPDX-License-Identifier: Apache-2.0
import { expect, test } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.route("**/api/browse**", (route) => {
    const path = new URL(route.request().url()).searchParams.get("path") || "/data";
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ path, dirs: ["cases"], files: ["ct.nii.gz"], parent: path === "/data" ? null : "/data" }),
    });
  });
  await page.goto("/tests/harness.html");
});

test("the path field has the focus on open, Escape closes, and a stale listing never survives a navigation", async ({
  page,
}) => {
  await page.evaluate(() => (window as any).__mount("FolderBrowser", { start: "/data", pickFile: true }));
  const field = page.locator(".modal-path");
  await expect(field).toBeFocused();
  await expect(page.locator(".modal-list .dir")).toContainText(["cases"]);

  await page.locator(".modal-list .dir").last().click();
  await expect(field).toHaveValue("/data/cases");
  await expect(page.locator(".modal-list .dir.up")).toHaveCount(1);

  await page.keyboard.press("Escape");
  await expect.poll(() => page.evaluate(() => (window as any).__closed ?? 0)).toBe(1);
});

test("Enter on a typed path re-reads it and a picked file is handed back whole", async ({ page }) => {
  await page.evaluate(() => (window as any).__mount("FolderBrowser", { start: "/data", pickFile: true }));
  const field = page.locator(".modal-path");
  await field.fill("/data/elsewhere");
  await field.press("Enter");
  await expect(page.locator(".modal-list .dir.up")).toHaveCount(1);
  await page.locator(".modal-list .file").click();
  await expect.poll(() => page.evaluate(() => (window as any).__picked)).toBe("/data/elsewhere/ct.nii.gz");
});

test("the folder dialog confines focus, blocks outside focus, and restores its opener on every close", async ({ page }) => {
  await page.evaluate(() => (window as any).__mount("FolderPicker"));
  const opener = page.getByRole("button", { name: "Browse folders" });
  await opener.click();
  const dialog = page.getByRole("dialog", { name: "Choose a folder" });
  const field = dialog.getByRole("textbox", { name: "Folder path" });
  const pick = dialog.getByRole("button", { name: "Use this folder" });
  await expect(field).toBeFocused();
  await expect(pick).toBeEnabled();
  // A native dialog may let Tab reach browser chrome (document.activeElement is then body), but
  // must never expose a control in the inert page behind it. Exercise both ends of the order.
  for (const key of ["Tab", "Shift+Tab"]) {
    for (let step = 0; step < 8; step += 1) {
      await page.keyboard.press(key);
      expect(await dialog.evaluate((node) => node.contains(document.activeElement) || document.activeElement === document.body)).toBe(true);
    }
  }
  await field.focus();
  await expect(field).toBeFocused();
  await page.getByRole("button", { name: "Outside control", includeHidden: true }).evaluate((node: HTMLElement) => node.focus());
  await expect(field).toBeFocused();
  await field.press("Escape");
  await expect(dialog).toHaveCount(0);
  await expect(opener).toBeFocused();

  await opener.press("Enter");
  await dialog.getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(opener).toBeFocused();
  await opener.press("Enter");
  await pick.click();
  await expect(page.getByRole("status")).toHaveText("/data");
  await expect(opener).toBeFocused();
  await opener.press("Enter");
  await page.mouse.click(2, 2); // the dimmed space outside the folder card
  await expect(dialog).toHaveCount(0);
  await expect(opener).toBeFocused();
});
