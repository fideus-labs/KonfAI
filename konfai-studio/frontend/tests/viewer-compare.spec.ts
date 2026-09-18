// SPDX-License-Identifier: Apache-2.0
import { expect, test, type Page } from "@playwright/test";
import { tinyNifti } from "./nifti";
import { idleStream, studioRoutes } from "./studio-fixtures";

async function volumes(page: Page): Promise<{ path: string | null; compare: string | null }> {
  return JSON.parse((await page.locator("#volumes").textContent()) ?? "{}");
}

test("comparing, a tree click fills the pane last clicked, either one", async ({ page }) => {
  await studioRoutes(page);
  await page.route("**/api/experiment/ls?**", (route) =>
    route.fulfill({ json: { root: "/workspace", dirs: [], files: ["a", "b", "c"].map((n) => ({ name: `${n}.nii.gz` })) } }),
  );
  const volume = tinyNifti();
  await page.route("**/files/volume**", (route) =>
    route.fulfill({ status: 200, contentType: "application/gzip", body: volume }),
  );
  await page.goto("/tests/harness.html");
  await page.evaluate((stream) => (window as any).__mount("PanelProbe", { stream }), idleStream());

  await page.getByText("a.nii.gz").click();
  await page.getByRole("button", { name: "Compare" }).click();
  const [paneA, paneB] = [page.locator(".v-pane").nth(0), page.locator(".v-pane").nth(1)];
  await expect(paneB).toHaveClass(/target/); // a fresh compare fills its empty second pane

  await page.getByText("b.nii.gz").click();
  await expect.poll(() => volumes(page)).toEqual({ path: "/workspace/a.nii.gz", compare: "/workspace/b.nii.gz" });

  await paneA.click();
  await expect(paneA).toHaveClass(/target/);
  await expect(paneB).not.toHaveClass(/target/);
  await page.getByText("c.nii.gz").click();
  await expect.poll(() => volumes(page)).toEqual({ path: "/workspace/c.nii.gz", compare: "/workspace/b.nii.gz" });

  await paneB.click();
  await page.getByText("a.nii.gz").click();
  await expect.poll(() => volumes(page)).toEqual({ path: "/workspace/c.nii.gz", compare: "/workspace/a.nii.gz" });
});
