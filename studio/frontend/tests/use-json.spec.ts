// SPDX-License-Identifier: Apache-2.0
import { expect, test } from "@playwright/test";

test("a new query's data never shows under the old one, and a failure is an error, not an empty result", async ({
  page,
}) => {
  await page.route("**/api/evaluations**", async (route) => {
    const session = new URL(route.request().url()).searchParams.get("session");
    if (session === "A") await new Promise((r) => setTimeout(r, 1200));
    if (session === "broken") return route.fulfill({ status: 500, body: "boom" });
    return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ runs: [{ run: session }] }) });
  });
  await page.goto("/tests/harness.html");
  const state = async () => JSON.parse(await page.locator("#json").innerText());

  await page.evaluate(() => (window as any).__mount("JsonProbe", { url: "/api/evaluations?session=B" }));
  await expect.poll(async () => (await state()).data?.runs?.[0]?.run).toBe("B");

  // Switch to the slow session: B's rows must not stand in for A's while A loads.
  await page.evaluate(() => (window as any).__mount("JsonProbe", { url: "/api/evaluations?session=A" }));
  const during = await state();
  expect(during.loading).toBe(true);
  expect(during.data).toBeNull();
  await expect.poll(async () => (await state()).data?.runs?.[0]?.run, { timeout: 5000 }).toBe("A");

  // Switch back to a slow A then to B before A answers: A's late answer never lands under B.
  await page.evaluate(() => (window as any).__mount("JsonProbe", { url: "/api/evaluations?session=A" }));
  await page.waitForTimeout(100);
  await page.evaluate(() => (window as any).__mount("JsonProbe", { url: "/api/evaluations?session=B" }));
  await expect.poll(async () => (await state()).data?.runs?.[0]?.run).toBe("B");
  await page.waitForTimeout(1500);
  expect((await state()).data.runs[0].run).toBe("B");

  await page.evaluate(() => (window as any).__mount("JsonProbe", { url: "/api/evaluations?session=broken" }));
  await expect.poll(async () => (await state()).error).not.toBeNull();
  expect((await state()).data).toBeNull();
});
