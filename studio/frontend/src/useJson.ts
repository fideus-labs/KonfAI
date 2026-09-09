// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from "react";
import { getJson } from "./api";

// The load-on-deps effect, once. Refetches when `deps` change; keeps the last value while the SAME url
// reloads and starts empty when the url changes (a new session's results never show under the old
// ones); `error` carries a failed request's message, so an empty answer and a failed one differ;
// `loading` runs true from a fetch's start to its end. The in-flight request is aborted on change.
export function useJson<T>(url: string, deps: unknown[]): { data: T | null; loading: boolean; error: string | null } {
  const [state, setState] = useState<{ url: string; data: T | null; error: string | null }>({
    url,
    data: null,
    error: null,
  });
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    if (!url) {
      setState({ url, data: null, error: null });
      setLoading(false);
      return;
    }
    const controller = new AbortController();
    setState((previous) => (previous.url === url ? previous : { url, data: null, error: null }));
    setLoading(true);
    getJson<T>(url, controller.signal)
      .then((d) => {
        setState({ url, data: d, error: null });
        setLoading(false);
      })
      .catch((e: unknown) => {
        if (controller.signal.aborted) return;
        setState({ url, data: null, error: e instanceof Error ? e.message : String(e) });
        setLoading(false);
      });
    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  const current = state.url === url;
  return { data: current ? state.data : null, loading, error: current ? state.error : null };
}
