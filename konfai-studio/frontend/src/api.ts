// SPDX-License-Identifier: Apache-2.0

// The fetch + JSON helpers the whole front routes through, one canonical Content-Type casing for POST.
// Callers keep their own error handling (.catch/.finally); these just do the request and parse the body.
// A refusal carries the server's own `detail` as its message, else the status.

export async function failure(r: Response): Promise<Error> {
  const body = await r.json().catch(() => null);
  return new Error(typeof body?.detail === "string" ? body.detail : `${r.status} ${r.statusText}`);
}

export async function getJson<T = any>(url: string, signal?: AbortSignal): Promise<T> {
  const r = await fetch(url, { signal });
  if (!r.ok) throw await failure(r);
  return r.json();
}

export async function postJson<T = any>(url: string, body: unknown, headers?: Record<string, string>): Promise<T> {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...headers },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw await failure(r);
  return r.json();
}
