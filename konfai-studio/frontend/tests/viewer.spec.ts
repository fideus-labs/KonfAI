// SPDX-License-Identifier: Apache-2.0
import { expect, test } from "@playwright/test";
import { tinyNifti } from "./nifti";

test("a stale volume load neither replaces nor clears the newer selection", async ({ page }) => {
  // The first request is slow and fails; the second is fast and succeeds. NiiVue mutates its volume
  // list as a load settles, so the old bug put the stale outcome on screen: its failure cleared the
  // newer volume, and its success renamed it.
  const volume = tinyNifti();
  await page.route("**/files/volume**", async (route) => {
    const path = new URL(route.request().url()).searchParams.get("path") ?? "";
    if (path.includes("slow")) {
      await new Promise((r) => setTimeout(r, 1500));
      return route.abort("failed");
    }
    return route.fulfill({ status: 200, contentType: "application/gzip", body: volume });
  });
  await page.goto("/tests/harness.html");
  await page.evaluate(() => (window as any).__mount("Viewer", { path: "/data/slow.nii.gz" }));
  await page.waitForTimeout(200);
  await page.evaluate(() => (window as any).__mount("Viewer", { path: "/data/fast.nii.gz" }));

  await expect(page.locator(".v-dims")).not.toHaveText("", { timeout: 10_000 });
  await page.waitForTimeout(2000); // past the stale request's failure
  await expect(page.locator(".v-failed")).toHaveCount(0);
  await expect(page.locator(".v-dims")).not.toHaveText("");
});

test("a failed load says which file failed and shows no stale volume", async ({ page }) => {
  // Bytes that are not a volume: the reader rejects them, the way it rejects a missing or foreign file.
  await page.route("**/files/volume**", (route) =>
    route.fulfill({ status: 200, contentType: "application/octet-stream", body: "not a volume" }),
  );
  await page.goto("/tests/harness.html");
  await page.evaluate(() => (window as any).__mount("Viewer", { path: "/data/missing.nii.gz" }));

  await expect(page.locator(".v-failed")).toContainText("missing.nii.gz", { timeout: 10_000 });
  await expect(page.locator(".v-dims")).toHaveCount(0);
});

test("a Go-to focus is dropped when another volume is picked before its own has loaded", async ({ page }) => {
  // The requested volume is slow and another one is picked meanwhile: the old check took whichever volume
  // loaded and put the first one's point on it.
  const volume = tinyNifti();
  await page.route("**/files/volume**", async (route) => {
    if (route.request().url().includes("slow")) await new Promise((r) => setTimeout(r, 1000));
    return route.fulfill({ status: 200, contentType: "application/gzip", body: volume });
  });
  await page.goto("/tests/harness.html");
  const focus = { mm: [-2, -3, 4], nonce: 1, path: "/data/slow.nii.gz" };
  await page.evaluate((focus) => (window as any).__mount("Viewer", { path: focus.path, focus }), focus);
  await page.waitForTimeout(200);
  await page.evaluate((focus) => (window as any).__mount("Viewer", { path: "/data/other.nii.gz", focus }), focus);
  await expect(page.locator(".v-dims")).not.toHaveText("", { timeout: 10_000 });
  const crosshair = () => page.evaluate(() => Array.from((window as any).__nv.scene.crosshairPos as number[]));
  expect(await crosshair()).toEqual([0.5, 0.5, 0.5]);

  // A focus into the volume on screen still lands.
  const here = { ...focus, nonce: 2, path: "/data/other.nii.gz" };
  await page.evaluate((focus) => (window as any).__mount("Viewer", { path: focus.path, focus }), here);
  await expect.poll(crosshair).not.toEqual([0.5, 0.5, 0.5]);
});
