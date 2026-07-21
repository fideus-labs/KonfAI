// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from "react";
import { getJson } from "./api";

// The cancellable load-on-deps effect, once. Refetches when `deps` change, keeps the last value during an
// in-flight reload, and clears to null on error or an empty url. `loading` runs true from a fetch's start
// to its end.
export function useJson<T>(url: string, deps: unknown[]): { data: T | null; loading: boolean } {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    if (!url) {
      setData(null);
      setLoading(false);
      return;
    }
    let live = true;
    setLoading(true);
    getJson<T>(url)
      .then((d) => {
        if (live) {
          setData(d);
          setLoading(false);
        }
      })
      .catch(() => {
        if (live) {
          setData(null);
          setLoading(false);
        }
      });
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return { data, loading };
}
