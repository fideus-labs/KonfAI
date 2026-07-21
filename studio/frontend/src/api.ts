// SPDX-License-Identifier: Apache-2.0

// The fetch + JSON helpers the whole front routes through — one canonical Content-Type casing for POST.
// Callers keep their own error handling (.catch/.finally); these just do the request and parse the body.

export async function getJson<T = any>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

export async function postJson<T = any>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}
