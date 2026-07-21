// SPDX-License-Identifier: Apache-2.0

// Per-experiment UI state, consolidated from the parallel `Record<string, X>` maps that used to sit
// side-by-side in App (one map per field, all keyed by the same session name). One entry per session;
// every field is optional so a session that never touched a slice simply omits it — read it with the
// same `?? fallback` the call sites already used.
export interface SessionUiState {
  title?: string; // display name (server-titled; falls back to the raw session id)
  dataset?: string; // dataset path recorded for the experiment
  device?: string; // compute-device override
  status?: string; // latest polled job status
  volume?: string | null; // primary volume shown in the viewer
  compareVol?: string | null; // second volume for the viewer's compare pane
  runNonce?: number; // bumped to re-subscribe the job stream
  inject?: { text: string; nonce: number }; // a prompt to replay into the chat
  busy?: boolean; // the experiment's agent is working
}

export type SessionUi = Record<string, SessionUiState>;

// Immutably update one session's slice, creating it if absent. Other sessions are untouched, and the
// slice's unpatched fields (e.g. a stable `inject` object) keep their reference.
export function patchSession(ui: SessionUi, session: string, patch: Partial<SessionUiState>): SessionUi {
  return { ...ui, [session]: { ...ui[session], ...patch } };
}

// Replace one field across the whole map from a fresh server map — the old `setX(server ?? {})`
// full-replace semantics, but scoped to a single field so the other per-session slices survive:
// every existing session's field is reset to the server value (undefined when absent), and sessions
// new to the server map are added.
export function replaceSessionField<K extends keyof SessionUiState>(
  ui: SessionUi,
  field: K,
  values: Record<string, SessionUiState[K]>,
): SessionUi {
  const next: SessionUi = {};
  for (const s of Object.keys(ui)) next[s] = { ...ui[s], [field]: values[s] };
  for (const s of Object.keys(values)) if (!next[s]) next[s] = { [field]: values[s] };
  return next;
}
