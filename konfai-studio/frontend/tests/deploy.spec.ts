// SPDX-License-Identifier: Apache-2.0
import { expect, test } from "@playwright/test";
import { tinyNifti } from "./nifti";

// The app has an in-browser build but its engine does not load (here: the server has none of its files).
// The pane says so and keeps Run disabled: a volume alone gives it nothing it could run.
test("the Deploy pane keeps Run disabled while its engine is not ready", async ({ page }) => {
  const entry = {
    name: "CT",
    engine: "konfai-edge",
    runtime: "CT/engine",
    configs: [{ name: "app.json", url: "CT/app.json" }],
    variants: [{ weights: "CT" }],
  };
  await page.route("**/api/apps/web?**", (route) => route.fulfill({ json: entry }));
  await page.route("**/api/apps/web/file?**", (route) => route.fulfill({ status: 404, json: { detail: "no file" } }));
  await page.goto("/tests/harness.html");
  await page.evaluate(() => (window as any).__mount("Deploy", { ref: "/apps/CT" }));
  await expect(page.locator(".deploy-status.error")).toContainText("konfai_edge.js: the server said 404");
  const volume = { name: "ct.nii.gz", mimeType: "application/gzip", buffer: tinyNifti() };
  await page.locator(".deploy-drop input").setInputFiles(volume);
  await expect(page.locator(".deploy-drop")).toContainText("ct.nii.gz");
  await expect(page.getByRole("button", { name: "Run inference" })).toBeDisabled();
});
