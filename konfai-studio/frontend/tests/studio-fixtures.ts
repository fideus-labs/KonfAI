// SPDX-License-Identifier: Apache-2.0
import type { Page } from "@playwright/test";
import type { JobStream, RunFeed } from "../src/useJobStream";

export function finishedRun(run: string, kind: string): RunFeed {
  return { key: `${run} ${kind}`, run, kind, base: `${kind === "evaluation" ? "Evaluations" : "Statistics"}/${run}`,
    data: "", outputs: [], status: "done", startedAt: 0, series: {}, live: null };
}

export function idleStream(runs: RunFeed[] = []): JobStream {
  return { lines: [], runs, activeRun: "", run: "", status: "", kind: "", metricNonce: 0, doneNonce: 0 };
}

// Real components; only the local HTTP boundary is replaced. No model, shell or agent is launched.
export async function studioRoutes(page: Page) {
  await page.route("**/konfai-logo.png", (route) => route.fulfill({ contentType: "image/svg+xml",
    body: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"/>' }));
  await page.route("**/api/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/api/live") return route.fulfill({ status: 200, contentType: "text/event-stream", body: "" });
    const bodies: Record<string, unknown> = {
      "/api/auth": { required: false, authenticated: true },
      "/api/health": { agent: "ready" },
      "/api/sessions": { sessions: ["S"] },
      "/api/experiment": { checkpoints: [], predictions: [], jobs: [] },
      "/api/experiment/ls": { root: "/workspace", dirs: [], files: [{ name: "Config.yml" }, { name: "ct.nii.gz" }] },
      "/api/experiment/file": { name: "Config.yml", content: "Trainer: {}", editable: true },
    };
    return route.fulfill({ json: bodies[path] ?? {} });
  });
}
