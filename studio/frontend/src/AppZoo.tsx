// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";

export type StudioApp = {
  ref: string;
  source?: string;
  app_name?: string;
  display_name?: string;
  short_description?: string;
  description?: string;
  theme?: string;
  logo?: string;
};

// app.json short descriptions carry HTML (<b>, <br>, <cite>…) — show them as plain text.
function plain(html: string | undefined): string {
  if (!html) return "";
  return html
    .replace(/<br\s*\/?>/gi, " ")
    .replace(/<[^>]+>/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function initials(a: StudioApp): string {
  const s = (a.app_name || a.ref.split(":").pop() || a.ref).replace(/[^A-Za-z0-9]/g, "");
  return (s.slice(0, 2) || "AP").toUpperCase();
}

function hue(ref: string): number {
  let h = 0;
  for (let i = 0; i < ref.length; i++) h = (h * 31 + ref.charCodeAt(i)) % 360;
  return h;
}

// Not hardcoded per app: a real `theme` field wins; else the display_name prefix ("Synthesis: MR"
// → "Synthesis") that the app.json descriptions already carry; else a keyword guess on the ref.
function themeOf(a: StudioApp): string {
  if (a.theme) return a.theme;
  const dn = a.display_name || "";
  if (dn.includes(":")) {
    const prefix = dn.split(":")[0].trim();
    if (prefix) return prefix;
  }
  const s = `${a.ref} ${a.app_name || ""}`.toLowerCase();
  if (s.includes("synth")) return "Synthesis";
  if (s.includes("reg")) return "Registration";
  if (s.includes("seg")) return "Segmentation";
  return "Other";
}

const THEME_ORDER = ["Synthesis", "Registration", "Segmentation", "Other"];

// The App Zoo — a model gallery in its own window. Browse by theme, register/remove sources, and
// start an experiment from an app (the agent then runs it on your dataset).
export default function AppZoo({
  apps,
  loading,
  onUse,
  onAdd,
  onRemove,
  onDeploy,
  onClose,
}: {
  apps: StudioApp[];
  loading?: boolean;
  onUse: (ref: string) => void;
  onAdd: (ref: string) => void;
  onRemove: (ref: string) => void;
  onDeploy: (app: StudioApp) => void;
  onClose: () => void;
}) {
  const [q, setQ] = useState("");
  const query = q.trim().toLowerCase();
  const matches = apps.filter((a) =>
    query ? `${a.ref} ${a.app_name || ""} ${a.source || ""} ${themeOf(a)}`.toLowerCase().includes(query) : true,
  );
  const groups = new Map<string, StudioApp[]>();
  for (const a of matches) {
    const t = themeOf(a);
    (groups.get(t) || groups.set(t, []).get(t)!).push(a);
  }
  const themes = [...groups.keys()].sort((a, b) => THEME_ORDER.indexOf(a) - THEME_ORDER.indexOf(b));

  return (
    <div className="modal-back" onClick={onClose}>
      <div className="zoo" onClick={(e) => e.stopPropagation()}>
        <div className="zoo-head">
          <span className="zoo-title">
            KonfAI Apps <span className="zoo-count">{apps.length}</span>
          </span>
          <input className="zoo-search" placeholder="Search apps…" value={q} onChange={(e) => setQ(e.target.value)} />
          <button
            className="zoo-addbtn"
            onClick={() => {
              const r = window.prompt("Add an app source — a HuggingFace repo (owner/name[:app]) or a local path:");
              if (r && r.trim()) onAdd(r.trim());
            }}
          >
            + Add source
          </button>
          <button className="zoo-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>
        <div className="zoo-body">
          {loading && apps.length === 0 && (
            <div className="zoo-loading">
              <span className="zoo-spin" /> Loading apps…
            </div>
          )}
          {!loading && matches.length === 0 && <div className="zoo-empty">No apps match “{q}”.</div>}
          {themes.map((t) => (
            <section key={t} className="zoo-theme">
              <div className="zoo-theme-head">
                {t} <span className="zoo-theme-count">{groups.get(t)!.length}</span>
              </div>
              <div className="zoo-grid">
                {groups.get(t)!.map((a) => (
                  <div key={a.ref} className="zoo-tile">
                    {a.logo ? (
                      <img className="zoo-logo" src={a.logo} alt="" />
                    ) : (
                      <span className="zoo-mono" style={{ background: `hsl(${hue(a.ref)} 32% 42%)` }}>
                        {initials(a)}
                      </span>
                    )}
                    <div className="zoo-name" title={a.ref}>
                      {a.display_name || a.app_name || a.ref.split("/").pop()}
                    </div>
                    <div className="zoo-ref">{a.ref}</div>
                    {plain(a.short_description) && <div className="zoo-desc">{plain(a.short_description)}</div>}
                    <div className="zoo-actions">
                      <button className="zoo-use" onClick={() => onUse(a.ref)}>
                        Use →
                      </button>
                      <button className="zoo-deploy" onClick={() => onDeploy(a)} title="Run in-tab (zero egress)">
                        Deploy
                      </button>
                      <button className="zoo-remove" onClick={() => onRemove(a.ref)} title="Remove from catalogue">
                        Remove
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          ))}
        </div>
      </div>
    </div>
  );
}
