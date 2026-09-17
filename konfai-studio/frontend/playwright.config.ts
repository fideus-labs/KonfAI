// SPDX-License-Identifier: Apache-2.0
//
// Browser tests of the behaviours the Python tests cannot see: request ordering in the viewer, the
// identity of a session's data across a switch, the live stream's reconnection, the keyboard of the
// folder browser. Components are mounted alone on tests/harness.html (served by the vite dev server),
// and every API route they call is answered by the test, so no BFF, no model and no patient data run.
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 30_000,
  fullyParallel: false,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? "github" : "list",
  use: { baseURL: "http://localhost:5173", headless: true },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "npm run dev -- --port 5173 --strictPort",
    url: "http://localhost:5173/tests/harness.html",
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
