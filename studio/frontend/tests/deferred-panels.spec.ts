// SPDX-License-Identifier: Apache-2.0
import { expect, test } from "@playwright/test";
import { tinyNifti } from "./nifti";
import { studioRoutes } from "./studio-fixtures";

test("the workspace imports its terminal and viewer only when opened, then preserves their state", async ({ page }) => {
  await studioRoutes(page);
  let shellConnections = 0;
  await page.routeWebSocket("**/api/terminal", (socket) => {
    shellConnections += 1;
    socket.onMessage(() => undefined);
  });
  let volumeReads = 0;
  await page.route("**/files/volume?**", (route) => {
    volumeReads += 1;
    return route.fulfill({ contentType: "application/gzip", body: tinyNifti() });
  });
  const imports: string[] = [];
  page.on("request", (request) => {
    const path = new URL(request.url()).pathname;
    if (/\/src\/(Console|Viewer)\.tsx$/.test(path)) imports.push(path);
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Config.yml", exact: true }).click();
  await expect(page.locator(".cfg-body")).toHaveValue("Trainer: {}");
  await expect(page.getByRole("button", { name: "Terminal", exact: true })).toBeVisible();
  expect(imports).toEqual([]);
  expect(shellConnections).toBe(0);
  await expect(page.locator(".term-host, .exp-viewer canvas")).toHaveCount(0);

  await page.getByRole("button", { name: "Terminal", exact: true }).click();
  await expect(page.locator(".console.open .xterm")).toBeVisible();
  await expect(page.locator(".console .cst")).toHaveText("connected");
  expect(imports).toEqual(["/src/Console.tsx"]);
  // The dev entry uses StrictMode and rehearses effect cleanup on mount. Reopening the drawer
  // must preserve the surviving shell rather than start another one.
  const initialConnections = shellConnections;
  expect(initialConnections).toBeGreaterThanOrEqual(1);
  await page.getByRole("button", { name: "Collapse terminal", exact: true }).click();
  await expect(page.locator(".term-host")).toBeHidden();
  await page.getByRole("button", { name: "Open terminal", exact: true }).click();
  await expect(page.locator(".console.open .xterm")).toBeVisible();
  expect(shellConnections).toBe(initialConnections);
  await page.getByRole("button", { name: "Collapse terminal", exact: true }).click();

  await page.getByRole("button", { name: "ct.nii.gz", exact: true }).click();
  await expect(page.locator(".exp-viewer .v-dims")).not.toHaveText("", { timeout: 10_000 });
  expect(imports).toEqual(["/src/Console.tsx", "/src/Viewer.tsx"]);
  const canvas = await page.locator(".exp-viewer canvas").elementHandle();
  await page.getByRole("button", { name: "Config.yml", exact: true }).click();
  await expect(page.locator(".cfg-body")).toHaveValue("Trainer: {}");
  await expect(page.locator(".exp-viewer")).toBeHidden();
  await page.getByRole("button", { name: "ct.nii.gz", exact: true }).click();
  await expect(page.locator(".exp-viewer canvas")).toBeVisible();
  expect(await canvas!.evaluate((node) => node === document.querySelector(".exp-viewer canvas"))).toBe(true);
  expect(volumeReads).toBe(1);
});
