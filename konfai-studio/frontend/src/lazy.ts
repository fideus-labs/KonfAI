// SPDX-License-Identifier: Apache-2.0

import { type ComponentType, lazy } from "react";

// A lazily loaded component whose chunk is fetched again once when the first fetch fails (a deploy
// that replaced the chunk under a page still open, a flaky connection); the second failure reaches
// the nearest error boundary like any other render error.
export function lazyWithRetry<T extends ComponentType<any>>(importer: () => Promise<{ default: T }>) {
  return lazy(() => importer().catch(() => importer()));
}
