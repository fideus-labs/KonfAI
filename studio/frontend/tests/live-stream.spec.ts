// SPDX-License-Identifier: Apache-2.0
import { expect, test } from "@playwright/test";

test("a live stream that drops is reconnected, and the events before the drop are kept", async ({ page }) => {
  let connections = 0;
  await page.route("**/api/live**", (route) => {
    connections += 1;
    const frames =
      connections === 1
        ? 'data: {"type":"job","run":"RUN_1","kind":"train","status":"running"}\n\ndata: {"type":"log","line":"epoch 1"}\n\n'
        : 'data: {"type":"log","line":"epoch 2"}\n\n';
    return route.fulfill({ status: 200, contentType: "text/event-stream", body: frames }); // the body ends: a drop
  });
  await page.goto("/tests/harness.html");
  await page.evaluate(() => (window as any).__mount("StreamProbe", { session: "S" }));
  const state = async () => JSON.parse(await page.locator("#stream").innerText());

  await expect.poll(async () => (await state()).run).toBe("RUN_1");
  await expect.poll(() => connections, { timeout: 10_000 }).toBeGreaterThanOrEqual(2);
  await expect.poll(async () => (await state()).lines, { timeout: 10_000 }).toBeGreaterThanOrEqual(2);
  expect((await state()).run).toBe("RUN_1");
});
