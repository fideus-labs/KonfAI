// SPDX-License-Identifier: Apache-2.0

import { type FormEvent, useState } from "react";
import { getJson } from "./api";

// The lock screen for a remote deployment. Exchanges the shared access token for an httpOnly session
// cookie; the token is never stored client-side. Shown only when the server reports auth is required.
export default function Login({ onAuthed }: { onAuthed: () => void }) {
  const [token, setToken] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (!token.trim() || busy) return;
    setBusy(true);
    setErr("");
    try {
      const r = await fetch("/api/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token: token.trim() }),
      });
      if (r.ok) {
        // Confirm the session cookie actually stuck. Over plain http a Secure cookie is silently dropped,
        // which would otherwise bounce straight back to this screen with no explanation.
        const st = await getJson<{ authenticated?: boolean }>("/api/auth").catch(() => ({ authenticated: true }));
        if (st.authenticated) {
          onAuthed();
          return;
        }
        setErr("Signed in, but the session cookie wasn't stored — are you on HTTPS? (see docs/REMOTE.md)");
        return;
      }
      setErr(r.status === 401 ? "That access token isn't right." : "Sign-in failed — try again.");
    } catch {
      setErr("Can't reach the server.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="lock">
      <form className="lock-card" onSubmit={submit}>
        <img className="lock-mark" src="/konfai-logo.png" alt="KonfAI" />
        <div className="lock-title">Studio</div>
        <div className="lock-sub">Enter the access token to continue.</div>
        <input
          className="lock-input"
          type="password"
          autoFocus
          autoComplete="current-password"
          placeholder="Access token"
          value={token}
          onChange={(e) => setToken(e.target.value)}
        />
        {err && <div className="lock-err">{err}</div>}
        <button className="lock-btn" type="submit" disabled={busy || !token.trim()}>
          {busy ? "Unlocking…" : "Unlock"}
        </button>
      </form>
    </div>
  );
}
