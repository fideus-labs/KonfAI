// SPDX-License-Identifier: Apache-2.0
import { expect, test } from "@playwright/test";
import { finishedRun, idleStream, studioRoutes } from "./studio-fixtures";

const cases = [
  { endpoint: "previews", label: "model samples", run: "trial-a", kind: "train",
    data: { previews: [{ label: "Training/CT", steps: [1] }] }, result: ".sample-card" },
  { endpoint: "evaluations", label: "evaluation scores", run: "trial-a", kind: "evaluation",
    data: { runs: [{ run: "trial-a", split: "TRAIN", metrics: [{ name: "Dice", mean: 0.95 }],
      cases: 1, case_metrics: [], case_rows: [] }] }, result: ".evrun" },
  { endpoint: "leaderboard", label: "the leaderboard", run: "", kind: "train",
    data: { leaderboards: { Dice: [{ run_name: "trial-a", value: 0.95, direction: "max" }] } }, result: ".lb-card" },
];

for (const scenario of cases) {
  test(`${scenario.endpoint} shows a failed request and Retry recovers the actual panel`, async ({ page }) => {
    await studioRoutes(page);
    let requests = 0;
    await page.route(`**/api/${scenario.endpoint}?**`, (route) => {
      requests += 1;
      return requests === 1 ? route.fulfill({ status: 503, body: "unavailable" }) : route.fulfill({ json: scenario.data });
    });
    await page.goto("/tests/harness.html");
    const stream = idleStream([finishedRun("trial-a", scenario.kind), finishedRun("trial-b", "train")]);
    await page.evaluate((stream) => (window as any).__mount("PanelProbe", { stream }), stream);
    await page.getByRole("button", { name: scenario.run ? /trial-a/ : "Leaderboard", exact: !scenario.run }).click();

    const failure = page.getByRole("alert");
    await expect(failure).toContainText(`Could not load ${scenario.label}: 503`);
    await expect(page.getByText(/no evaluations? yet|No TRAIN evaluations yet/i)).toHaveCount(0);
    await failure.getByRole("button", { name: "Retry" }).click();
    await expect(failure).toHaveCount(0);
    await expect(page.locator(scenario.result)).toBeVisible();
    expect(requests).toBe(2);
  });
}
