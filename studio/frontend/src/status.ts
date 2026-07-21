// SPDX-License-Identifier: Apache-2.0

// A job status → a rail signal: orange while it runs, red if it died, green when done, else neutral.
export function jobState(status?: string): "run" | "err" | "done" | "idle" {
  if (!status) return "idle";
  if (/killed|failed|error|cancel/i.test(status)) return "err";
  if (/running|waiting|queued|connect/i.test(status)) return "run";
  if (/done|complete|finish/i.test(status)) return "done";
  return "idle";
}

// A job that is running, or queued/waiting to start (no terminal or connecting state).
export function isRunning(status: string): boolean {
  return /running|waiting|queued/i.test(status);
}
