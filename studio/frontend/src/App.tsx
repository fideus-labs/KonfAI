// SPDX-License-Identifier: Apache-2.0

import { type MouseEvent as ReactMouseEvent, useEffect, useRef, useState } from "react";
import AppZoo, { type StudioApp } from "./AppZoo";
import Deploy from "./Deploy";
import Chat, { clearChat } from "./Chat";
import Console from "./Console";
import FolderBrowser from "./FolderBrowser";
import Login from "./Login";
import RightPanel from "./RightPanel";
import { useJobStream } from "./useJobStream";
import { getJson, postJson } from "./api";
import { useJson } from "./useJson";
import { jobState } from "./status";
import { type SessionUi, patchSession, replaceSessionField } from "./sessionState";

type Gpu = {
  index: number;
  name: string;
  used_gb: number | null;
  total_gb: number | null;
  util_percent: number | null; // compute load: VRAM alone cannot say whether the card is working
};
type Ram = { used_gb: number; total_gb: number };
type Brain = { id: string; label: string; detail: string; available: boolean; models?: { id: string; label: string }[] };

// A draft "experiment" with no server session yet: the first prompt materialises it.
const NEW = "__new__";

// On a token-required deployment, sign-out / session-expiry must not leave chat transcripts behind in
// localStorage (Chat.tsx persists the last messages per experiment): they'd be readable on a shared
// browser without the token. Clear the per-experiment chat keys; UI prefs (split, rail width) stay.
function purgeStudioChat() {
  for (const k of Object.keys(localStorage)) if (k.startsWith("konfai-studio:chat:")) localStorage.removeItem(k);
}

function PanelIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.7">
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <line x1="9" y1="4" x2="9" y2="20" />
    </svg>
  );
}

function SearchIcon() {
  return (
    <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round">
      <circle cx="11" cy="11" r="7" />
      <line x1="16.5" y1="16.5" x2="21" y2="21" />
    </svg>
  );
}

function PowerIcon() {
  return (
    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
      <path d="M12 3v8" />
      <path d="M6.2 6.4a8 8 0 1 0 11.6 0" />
    </svg>
  );
}

function AppsIcon() {
  return (
    <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor">
      <rect x="3" y="3" width="7.5" height="7.5" rx="1.6" />
      <rect x="13.5" y="3" width="7.5" height="7.5" rx="1.6" />
      <rect x="3" y="13.5" width="7.5" height="7.5" rx="1.6" />
      <rect x="13.5" y="13.5" width="7.5" height="7.5" rx="1.6" />
    </svg>
  );
}

// One live resource readout (GPU VRAM or system RAM); the dot brightens with usage.
function Sparkline({ points }: { points: number[] }) {
  const w = 220;
  const h = 60;
  const n = points.length;
  const xy = points.map((p, i) => `${(i / (n - 1)) * w},${h - (Math.max(0, Math.min(100, p)) / 100) * h}`).join(" ");
  return (
    <svg className="spark" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-hidden="true">
      <polygon points={`0,${h} ${xy} ${w},${h}`} className="spark-fill" />
      <polyline points={xy} className="spark-line" />
    </svg>
  );
}

function Meter({
  label,
  used,
  total,
  title,
  traces,
  text,
  fill,
}: {
  label: string;
  used: number | null;
  total: number | null;
  title?: string;
  // What the click reveals, one sparkline each: a GPU has two things worth watching over time (how
  // busy it is, and how full), RAM and CPU only one.
  traces?: { label: string; points: number[] }[];
  // A meter reads "used/total GB" by default; CPU has no total to speak of, so it renders its own text.
  text?: string;
  fill?: number; // 0..1 for the dot, when there is no used/total to derive it from
}) {
  const [open, setOpen] = useState(false);
  const frac = fill ?? (used != null && total ? Math.min(used / total, 1) : 0);
  const shown = (traces ?? []).filter((t) => t.points.length > 1);
  const clickable = shown.length > 0;
  return (
    <span
      className="chip meter"
      title={title}
      style={{ cursor: clickable ? "pointer" : "default" }}
      onClick={() => clickable && setOpen((o) => !o)}
    >
      <span className="gdot" style={{ opacity: 0.3 + frac * 0.7 }} />
      <span className="k">{label}</span>
      {text ?? (used != null && total != null ? ` ${used}/${total} GB` : "")}
      {open && clickable && (
        <>
          <span className="pop-back" onClick={(e) => { e.stopPropagation(); setOpen(false); }} />
          <span className="meter-pop" onClick={(e) => e.stopPropagation()}>
            {shown.map((t) => (
              <span key={t.label} className="meter-pop-trace">
                <span className="meter-pop-head">
                  {label} · {t.label}
                  <span className="meter-pop-now">{Math.round(t.points[t.points.length - 1])}%</span>
                </span>
                <Sparkline points={t.points} />
              </span>
            ))}
          </span>
        </>
      )}
    </span>
  );
}

// Auth gate: decide once whether this deployment needs a token, and only mount the workspace when the
// browser holds a valid session. Trusted-local (no token) falls straight through. A mid-session 401
// (session expired) flips back to the lock screen via the `ks-unauth` event from main.tsx.
export default function App() {
  const [auth, setAuth] = useState<"checking" | "in" | "out">("checking");
  const [remote, setRemote] = useState(false);
  const { data: authInfo, loading: authLoading } = useJson<{ required?: boolean; authenticated?: boolean }>(
    "/api/auth",
    [],
  );

  // A mid-session 401 (session expired) flips back to the lock screen via the `ks-unauth` event from main.tsx.
  useEffect(() => {
    const onUnauth = () => {
      purgeStudioChat(); // a session that expired must not leave transcripts behind
      setAuth("out");
    };
    window.addEventListener("ks-unauth", onUnauth);
    return () => window.removeEventListener("ks-unauth", onUnauth);
  }, []);

  // Decide once from /api/auth whether this deployment needs a token and whether the browser holds a valid
  // session. A failed check (server unreachable) falls through to the app's own offline state.
  useEffect(() => {
    if (authLoading) return;
    if (!authInfo) {
      setAuth("in");
      return;
    }
    setRemote(!!authInfo.required);
    if (authInfo.required && !authInfo.authenticated) {
      purgeStudioChat();
      setAuth("out");
    } else {
      setAuth("in");
    }
  }, [authLoading, authInfo]);

  if (auth === "checking") return <div className="lock" />;
  if (auth === "out") return <Login onAuthed={() => setAuth("in")} />;
  return <Studio remote={remote} />;
}

function Studio({ remote }: { remote: boolean }) {
  const [status, setStatus] = useState("connecting…");
  const [stopped, setStopped] = useState(false); // the user shut the local server down from the titlebar
  const [sessions, setSessions] = useState<string[]>([]); // starts empty: no phantom "default" experiment
  // The status poll reads this to spot an experiment it does not know yet; a ref, because the poll's
  // interval closure would otherwise forever see the mount-time empty list.
  const sessionsRef = useRef<string[]>([]);
  useEffect(() => {
    sessionsRef.current = sessions;
  }, [sessions]);

  function refreshSessions() {
    // The rail's list, refetched without touching `active`: catching up on another browser's work must
    // never yank this one off the experiment it is reading.
    getJson("/api/sessions")
      .then((d) => {
        setSessions(d.sessions ?? []);
        setUi((u) => replaceSessionField(u, "title", d.titles ?? {}));
        setUi((u) => replaceSessionField(u, "dataset", d.datasets ?? {}));
      })
      .catch(() => {});
  }
  const [active, setActive] = useState(NEW);
  // One consolidated map of per-experiment UI state (title, dataset, device, status, volumes, run nonce,
  // inject prompt, busy): replaces the dozen parallel `Record<string, X>` maps keyed by session name.
  const [ui, setUi] = useState<SessionUi>({});
  const [recent, setRecent] = useState<string[]>([]);
  const [fileRecent, setFileRecent] = useState<string[]>([]);
  const [gpus, setGpus] = useState<Gpu[]>([]);
  const [ram, setRam] = useState<Ram | null>(null);
  const [ramHist, setRamHist] = useState<number[]>([]); // RAM utilisation % over time
  const [cpu, setCpu] = useState<number | null>(null); // host CPU load %, all cores
  const [cpuHist, setCpuHist] = useState<number[]>([]);
  const [gpuHist, setGpuHist] = useState<Record<number, number[]>>({}); // per-GPU compute load % over time
  const [gpuVramHist, setGpuVramHist] = useState<Record<number, number[]>>({}); // per-GPU VRAM % over time
  const [brains, setBrains] = useState<Brain[]>([]);
  const [brain, setBrain] = useState("");
  const [model, setModel] = useState("");
  const [modelText, setModelText] = useState(""); // free-text model for the local backend
  const [defaultDevice, setDefaultDevice] = useState("auto"); // what a fresh experiment starts on
  const [menu, setMenu] = useState<{ session: string; x: number; y: number } | null>(null);
  // What the right-clicked experiment contains: greys out actions that would just fail (fails open to null).
  const { data: menuCaps } = useJson<{ bundlable?: boolean; exportable?: boolean }>(
    menu ? `/api/experiment?session=${encodeURIComponent(menu.session)}` : "",
    [menu?.session],
  );
  const [datasetBrowsing, setDatasetBrowsing] = useState(false);
  const [appPick, setAppPick] = useState<string | null>(null); // app ref awaiting its dataset before launch
  const [apps, setApps] = useState<StudioApp[]>([]);
  const [appsLoading, setAppsLoading] = useState(true);
  const [zooOpen, setZooOpen] = useState(false);
  const [deployApp, setDeployApp] = useState<StudioApp | null>(null);

  function refreshApps() {
    setAppsLoading(true);
    getJson("/api/apps?session=apps")
      .then((d) => setApps(d.apps ?? []))
      .catch(() => {})
      .finally(() => setAppsLoading(false));
  }

  function openZoo() {
    setZooOpen(true);
    refreshApps();
  }
  const [railHidden, setRailHidden] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [railSearch, setRailSearch] = useState("");
  const [split, setSplit] = useState(() => {
    const s = parseFloat(localStorage.getItem("konfai-studio:split") || "");
    return s > 0.25 && s < 0.8 ? s : 0.46; // chat-column fraction of the workspace width
  });
  const [dragging, setDragging] = useState(false); // raises a shield over NiiVue while resizing
  const workRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    localStorage.setItem("konfai-studio:split", String(split));
  }, [split]);

  const [railWidth, setRailWidth] = useState(() => {
    const w = parseInt(localStorage.getItem("konfai-studio:railw") || "", 10);
    return w >= 150 && w <= 420 ? w : 208;
  });
  useEffect(() => {
    localStorage.setItem("konfai-studio:railw", String(railWidth));
  }, [railWidth]);

  // Generic horizontal drag: attach move/up listeners, restore cursor on release. A full-screen shield is
  // raised for the duration so the drag keeps working over the NiiVue canvas (which otherwise swallows the
  // mousemove): resizing must not stall just because a volume is loaded.
  function drag(onMove: (x: number) => void) {
    return (e: ReactMouseEvent) => {
      e.preventDefault();
      setDragging(true);
      const move = (ev: MouseEvent) => onMove(ev.clientX);
      const up = () => {
        document.removeEventListener("mousemove", move);
        document.removeEventListener("mouseup", up);
        document.body.style.cursor = "";
        setDragging(false);
      };
      document.addEventListener("mousemove", move);
      document.addEventListener("mouseup", up);
      document.body.style.cursor = "col-resize";
    };
  }

  const startColDrag = drag((x) => {
    const rect = workRef.current?.getBoundingClientRect();
    if (rect) setSplit(Math.min(0.8, Math.max(0.25, (x - rect.left) / rect.width)));
  });
  const startRailDrag = drag((x) => setRailWidth(Math.min(420, Math.max(150, x))));
  const [pick, setPick] = useState<{ session: string; action: "bundle" | "export" } | null>(null);
  const [toast, setToast] = useState("");
  const [tools, setTools] = useState(0);

  useEffect(() => {
    getJson("/api/health")
      .then((d) => {
        setStatus(d.agent === "ready" ? "ready" : "starting…");
        setTools(d.tools ?? 0);
      })
      .catch(() => setStatus("offline"));
    getJson("/api/sessions")
      .then((d) => {
        if (d.sessions?.length) {
          setSessions(d.sessions);
          setActive(d.sessions[0]); // land on the most recent experiment (its Live), not an empty draft
        }
        setUi((u) => replaceSessionField(u, "title", d.titles ?? {}));
        setUi((u) => replaceSessionField(u, "dataset", d.datasets ?? {}));
      })
      .catch(() => {});
    getJson("/api/datasets")
      .then((d) => setRecent(d.datasets ?? []))
      .catch(() => {});
    getJson("/api/files")
      .then((d) => setFileRecent(d.files ?? []))
      .catch(() => {});
    refreshApps();
    getJson("/api/llm")
      .then((d) => {
        setBrains(d.options ?? []);
        setBrain(d.current ?? "");
        setModel(d.model ?? "");
        setModelText(d.model ?? "");
      })
      .catch(() => {});
    getJson("/api/device")
      .then((d) => {
        setUi((u) => replaceSessionField(u, "device", d.devices ?? {}));
        setDefaultDevice(d.default ?? "auto");
      })
      .catch(() => {});
  }, []);

  // Live VRAM + RAM in the title bar: a light poll, co-located with training.
  useEffect(() => {
    const pull = () => {
      getJson("/api/system")
        .then((d) => {
          const nextGpus: Gpu[] = d.gpus ?? [];
          const nextRam: Ram | null = d.ram ?? null;
          const nextCpu: number | null = typeof d.cpu_percent === "number" ? d.cpu_percent : null;
          setGpus(nextGpus);
          setRam(nextRam);
          setCpu(nextCpu);
          const cap = 90; // ~4.5 min of history at a 3s poll
          const pct = (used: number | null, total: number | null) => (total ? Math.min(100, (used ?? 0) / total * 100) : 0);
          if (nextRam) setRamHist((h) => [...h, pct(nextRam.used_gb, nextRam.total_gb)].slice(-cap));
          if (nextCpu != null) setCpuHist((h) => [...h, nextCpu].slice(-cap));
          // Both are worth watching and they answer different questions: load says whether the card is
          // working, VRAM how close the run is to running out.
          setGpuHist((prev) => {
            const next: Record<number, number[]> = { ...prev };
            for (const g of nextGpus)
              if (g.util_percent != null) next[g.index] = [...(prev[g.index] ?? []), g.util_percent].slice(-cap);
            return next;
          });
          setGpuVramHist((prev) => {
            const next: Record<number, number[]> = { ...prev };
            for (const g of nextGpus)
              if (g.used_gb != null && g.total_gb) next[g.index] = [...(prev[g.index] ?? []), pct(g.used_gb, g.total_gb)].slice(-cap);
            return next;
          });
        })
        .catch(() => {});
      getJson("/api/sessions/status")
        .then((d) => {
          const statuses: Record<string, string> = d.statuses ?? {};
          setUi((u) => replaceSessionField(u, "status", statuses));
          // The same poll that colours the dots is how this browser learns of an experiment created in
          // another one: an unknown key means the rail's list is stale.
          if (Object.keys(statuses).some((s) => !sessionsRef.current.includes(s))) refreshSessions();
        })
        .catch(() => {});
    };
    pull();
    const id = setInterval(pull, 3000);
    return () => clearInterval(id);
  }, []);

  function chooseBrain(id: string) {
    setBrain(id); // optimistic; applies to each task on its next message
    postJson("/api/llm", { brain: id })
      .then((d) => {
        setBrains(d.options ?? []);
        setBrain(d.current ?? id);
        // A model pinned for another backend doesn't carry over: fall back to that backend's default.
        const models = (d.options ?? []).find((b: Brain) => b.id === (d.current ?? id))?.models ?? [];
        if (models.length && !models.some((m: { id: string }) => m.id === d.model)) chooseModel("");
      })
      .catch(() => {});
  }

  function chooseModel(id: string) {
    setModel(id);
    setModelText(id);
    postJson("/api/llm", { model: id })
      .then((d) => {
        setModel(d.model ?? id);
        setModelText(d.model ?? id);
      })
      .catch(() => {});
  }

  const deviceOf = (session: string) => ui[session]?.device ?? defaultDevice;

  function chooseDevice(session: string, val: string) {
    setUi((u) => patchSession(u, session, { device: val }));
    if (session === NEW) {
      setDefaultDevice(val); // a draft's choice becomes the default the new experiment inherits
      return;
    }
    postJson("/api/device", { session, device: val }).catch(() => {});
  }

  // A machine with a GPU should train on it: default a fresh experiment to GPU 0 once, so jobs don't
  // silently fall back to (very slow) CPU. Per-experiment overrides still win.
  const deviceDefaulted = useRef(false);
  useEffect(() => {
    if (!deviceDefaulted.current && gpus.length > 0 && defaultDevice === "auto") {
      deviceDefaulted.current = true;
      setDefaultDevice(String(gpus[0].index));
    }
  }, [gpus, defaultDevice]);

  // Plug in a backend's credentials from the UI: they persist server-side and every live agent is
  // rebuilt, so the next message uses them without restarting Studio.
  function connectLlm(fields: Record<string, string>) {
    return postJson("/api/llm", fields).then((d) => {
      setBrains(d.options ?? []);
      return (d.options ?? []).find((b: Brain) => b.id === brain)?.available ?? false;
    });
  }

  const llmProps = {
    brains,
    brain,
    model,
    modelText,
    onBrain: chooseBrain,
    onModel: chooseModel,
    onModelText: setModelText,
    onConnect: connectLlm,
  };

  // Remember a task's dataset server-side (persists, and the Experiment tree mounts it).
  function recordDataset(session: string, path: string) {
    setUi((u) => patchSession(u, session, { dataset: path }));
    postJson("/api/sessions/dataset", { session, path }).catch(() => {});
  }

  function chooseDataset(path: string) {
    // KonfAI's own words: a dataset has cases and groups. The assistant echoes the vocabulary it is
    // handed, so "channels" and "splits" must not appear here.
    const prompt = `Inspect the dataset at ${path}: cases, groups, shape and spacing, label classes, and anything incomplete. Preview one case. Don't train yet.`;
    postJson("/api/datasets", { path })
      .then((d) => setRecent(d.datasets ?? []))
      .catch(() => {});
    if (active === NEW) {
      startExperiment(prompt, path);
      return;
    }
    recordDataset(active, path);
    setUi((u) => patchSession(u, active, { inject: { text: prompt, nonce: (u[active]?.inject?.nonce ?? 0) + 1 } }));
  }

  // The Chat composer sends the "read this file" turn itself; here we just record it in the
  // shared history so any task can reuse it.
  function recordFile(ref: string) {
    postJson("/api/files", { path: ref })
      .then((d) => setFileRecent(d.files ?? []))
      .catch(() => {});
  }

  // "+ New experiment" just clears the selection: nothing is created until the first prompt.
  function newDraft() {
    clearChat(NEW); // a fresh draft, not the leftovers of the last one
    setActive(NEW);
  }

  // Materialise a draft: create the task, focus it, and send the first turn into it (the LLM
  // titles it from that turn). Reused by the composer, the suggestions, and the dataset picker.
  // Guarded against re-entry so a fast double activation can't create two sessions.
  const creating = useRef(false);
  function startExperiment(text: string, dataset?: string) {
    if (creating.current) return;
    creating.current = true;
    postJson("/api/sessions", {})
      .then((d) => {
        const id = d.current as string;
        clearChat(id); // a reused id (a deleted experiment-1 frees it) must not inherit the old chat
        clearChat(NEW); // the draft's turn is replayed into the new experiment via inject, not localStorage
        setSessions(d.sessions);
        setUi((u) => replaceSessionField(u, "title", d.titles ?? {}));
        if (dataset) recordDataset(id, dataset);
        const draftDevice = deviceOf(NEW);
        if (draftDevice !== defaultDevice) chooseDevice(id, draftDevice); // carry the draft's device over
        setActive(id);
        setUi((u) => patchSession(u, id, { inject: { text, nonce: (u[id]?.inject?.nonce ?? 0) + 1 } }));
      })
      .catch(() => {})
      .finally(() => {
        creating.current = false;
      });
  }

  function useApp(ref: string, dataset?: string) {
    const base = `I want to use the app "${ref}". Inspect it with describe_app, then import_app it into the session so it runs as a normal experiment`;
    const text = dataset
      ? `${base} on my dataset at ${dataset}. Check the dataset fits the app (its expected input groups against my dataset's group names), then run_prediction with the imported checkpoints; ask whether I want inference, evaluation, or fine-tuning (run_resume with weights_only).`
      : `${base} on my dataset: ask me for the dataset path if you don't have it, then run_prediction with the imported checkpoints; ask whether I want inference, evaluation, or fine-tuning (run_resume with weights_only).`;
    if (dataset) {
      // Keep it in the recent list too; startExperiment records it per-session so it mounts in the tree.
      postJson("/api/datasets", { path: dataset })
        .then((d) => setRecent(d.datasets ?? []))
        .catch(() => {});
    }
    startExperiment(text, dataset);
  }

  function addApp(ref: string) {
    setToast(`Registering ${ref}…`);
    postJson("/api/apps/register", { ref, session: "apps" })
      .then((d) => {
        if (d.apps) setApps(d.apps);
        setToast(d.ok ? `Added ${ref}` : d.result || "Could not add that app");
      })
      .catch(() => setToast("Failed to add app."));
  }

  function removeApp(ref: string) {
    if (!window.confirm(`Remove the app source “${ref}” from the catalogue?`)) return;
    postJson("/api/apps/unregister", { ref, session: "apps" })
      .then((d) => {
        if (d.apps) setApps(d.apps);
        if (!d.ok) setToast(d.result || "Could not remove that app");
      })
      .catch(() => setToast("Failed to remove app."));
  }

  function renameExperiment(s: string) {
    setMenu(null);
    const current = ui[s]?.title ?? s;
    const next = window.prompt("Rename experiment", current);
    const title = next?.trim();
    if (!title || title === current) return;
    setUi((u) => patchSession(u, s, { title })); // optimistic
    postJson("/api/sessions/rename", { session: s, title })
      .then((d) => d.titles && setUi((u) => replaceSessionField(u, "title", d.titles)))
      .catch(() => setToast("Rename failed."));
  }

  function deleteExperiment(s: string) {
    setMenu(null);
    if (!window.confirm(`Delete “${ui[s]?.title ?? s}” and its workspace (jobs, checkpoints)? This cannot be undone.`))
      return;
    clearChat(s); // drop its stored chat so a reused id starts clean
    postJson("/api/sessions/delete", { name: s })
      .then((d) => {
        setSessions(d.sessions);
        setUi((u) => replaceSessionField(u, "title", d.titles ?? {}));
        if (!d.sessions.includes(active)) setActive(NEW);
      })
      .catch(() => setToast("Delete failed."));
  }

  // Bundle/export are deterministic MCP actions (no LLM): they only need a destination folder.
  function runAction(session: string, action: "bundle" | "export", output: string) {
    setPick(null);
    const label = ui[session]?.title ?? session;
    setToast(`${action === "bundle" ? "Packaging" : "Exporting"} “${label}”…`);
    postJson(`/api/sessions/${action}`, { session, output })
      .then((d) => setToast(d.result || (d.ok ? "Done." : "Failed.")))
      .catch(() => setToast("Request failed."));
  }

  useEffect(() => {
    if (!toast) return;
    const id = setTimeout(() => setToast(""), 9000);
    return () => clearTimeout(id);
  }, [toast]);

  const bumpRun = (s: string) => setUi((u) => patchSession(u, s, { runNonce: (u[s]?.runNonce ?? 0) + 1 }));

  // One live subscription for the active task, shared by the feed (right) and the console (bottom).
  const stream = useJobStream(active, ui[active]?.runNonce ?? 0);

  // A task's latest job status: the live stream for the task in front, the rail's poll for the others.
  const statusOf = (s: string) => (s === active && stream.status ? stream.status : ui[s]?.status) ?? "";

  // Stopping the server is a LOCAL privilege: never offered on a remote (token-gated) session,
  // and the endpoint refuses non-loopback clients anyway.
  const stoppable = !remote && ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname);
  function quitStudio() {
    if (!window.confirm("Stop the KonfAI Studio server?\n\nThe page will go offline; running jobs may be interrupted."))
      return;
    // The header is what the server checks against a drive-by POST; only show the stopped screen
    // once the request actually succeeded, or the UI would lie about a server that is still up.
    postJson("/api/quit", {}, { "X-KonfAI-Studio": "quit" })
      .then(() => setStopped(true))
      .catch((e) => window.alert(`Could not stop the server: ${e?.message ?? e}`));
  }

  function signOut() {
    purgeStudioChat(); // drop transcripts before the httpOnly cookie is cleared and the page reloads
    fetch("/api/logout", { method: "POST" }).finally(() => location.reload());
  }

  if (stopped)
    return (
      <div className="stopped-wrap">
        <div>
          <h2>Studio server stopped</h2>
          <p>
            You can close this tab. Launch it again from Slicer or run <code>konfai-studio</code>.
          </p>
        </div>
      </div>
    );

  return (
    <div className="app">
      <div className="titlebar">
        <img className="wordmark" src="/konfai-logo.png" alt="KonfAI" />
        <span className="studio">Studio</span>
        <button className="apps-btn" onClick={openZoo} title="Browse KonfAI Apps">
          <AppsIcon /> KonfAI Apps
        </button>
        <span className="spacer" />
        {gpus.map((g) => (
          // Load first, then memory: a card can hold 20 GB and compute nothing, and it is the load that
          // says whether the run is actually moving.
          <Meter
            key={g.index}
            label={`GPU ${g.index}`}
            used={g.used_gb}
            total={g.total_gb}
            text={
              g.util_percent != null && g.used_gb != null && g.total_gb != null
                ? ` ${g.util_percent}% · ${g.used_gb}/${g.total_gb} GB`
                : undefined
            }
            title={`${g.name}: compute load and VRAM`}
            traces={[
              { label: "load", points: gpuHist[g.index] ?? [] },
              { label: "VRAM", points: gpuVramHist[g.index] ?? [] },
            ]}
          />
        ))}
        {ram && <Meter label="RAM" used={ram.used_gb} total={ram.total_gb} title="System RAM" traces={[{ label: "load", points: ramHist }]} />}
        {cpu != null && (
          <Meter
            label="CPU"
            used={null}
            total={null}
            text={` ${Math.round(cpu)}%`}
            fill={Math.min(cpu / 100, 1)}
            title="Host CPU load, averaged over all cores"
            traces={[{ label: "load", points: cpuHist }]}
          />
        )}
        {stoppable && (
          <button className="power-btn" onClick={quitStudio} title="Stop the Studio server (local session only)">
            <PowerIcon />
          </button>
        )}
      </div>

      <div
        className={railHidden ? "body rail-off" : "body"}
        style={railHidden ? undefined : { gridTemplateColumns: `${railWidth}px 5px minmax(0, 1fr)` }}
      >
        {railHidden && (
          <button className="rail-reopen" title="Show experiments" onClick={() => setRailHidden(false)}>
            <PanelIcon />
          </button>
        )}
        <aside className="rail">
          <div className="rail-head">
            <span>Experiments</span>
            <span className="rail-tools">
              <button
                className={searchOpen ? "rail-icon on" : "rail-icon"}
                title="Search experiments"
                onClick={() => {
                  setSearchOpen((o) => !o);
                  setRailSearch("");
                }}
              >
                <SearchIcon />
              </button>
              <button className="rail-icon" title="Hide experiments" onClick={() => setRailHidden(true)}>
                <PanelIcon />
              </button>
            </span>
          </div>
          {searchOpen && (
            <input
              className="rail-search"
              autoFocus
              placeholder="Search experiments…"
              value={railSearch}
              onChange={(e) => setRailSearch(e.target.value)}
              onKeyDown={(e) => e.key === "Escape" && (setSearchOpen(false), setRailSearch(""))}
            />
          )}
          <button className={active === NEW ? "rail-new-btn on" : "rail-new-btn"} onClick={newDraft}>
            <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round">
              <line x1="12" y1="5" x2="12" y2="19" />
              <line x1="5" y1="12" x2="19" y2="12" />
            </svg>
            New
          </button>
          <div className="rail-list">
            {sessions
              .filter((s) => !railSearch || (ui[s]?.title ?? s).toLowerCase().includes(railSearch.toLowerCase()))
              .map((s) => {
              const state = jobState(statusOf(s));
              const spinning = ui[s]?.busy || state === "run";
              return (
                <button
                  key={s}
                  className={s === active ? "task on" : "task"}
                  onClick={() => setActive(s)}
                  onContextMenu={(e) => {
                    e.preventDefault();
                    setMenu({ session: s, x: e.clientX, y: e.clientY });
                  }}
                  title="Right-click for actions"
                >
                  {spinning ? (
                    <span className="tspin st-run" />
                  ) : (
                    <span className={`tdot st-${state}`} />
                  )}
                  <span className="tname">{ui[s]?.title ?? s}</span>
                </button>
              );
            })}
          </div>
        </aside>

        {!railHidden && <div className="rail-divider" onMouseDown={startRailDrag} title="Drag to resize" />}

        <div className="main">
          <div className="work" ref={workRef} style={{ gridTemplateColumns: `${split}fr 6px ${1 - split}fr` }}>
            <div className="chat-col">
              <div className="chat-stack">
                {sessions.map((s) => (
                  <Chat
                    key={s}
                    session={s}
                    active={s === active}
                    inject={ui[s]?.inject}
                    recentFiles={fileRecent}
                    onVolume={(p, c) =>
                      // Two volumes surfaced together is a comparison ("the sCT next to the real CT"):
                      // the second fills the compare pane, which opens itself. The nonce says the
                      // assistant pointed at something: an event, not a state, so re-showing the same
                      // volume still brings the viewer forward instead of leaving a run tab in front.
                      setUi((u) =>
                        patchSession(u, s, {
                          volume: p,
                          volumeNonce: (u[s]?.volumeNonce ?? 0) + 1,
                          ...(c ? { compareVol: c } : {}),
                        }),
                      )
                    }
                    onTitle={(title) => setUi((u) => patchSession(u, s, { title }))}
                    onAttach={recordFile}
                    onBusy={(b) => setUi((u) => patchSession(u, s, { busy: b }))}
                    onPickDataset={() => setDatasetBrowsing(true)}
                    datasetRecent={recent}
                    onChooseDataset={chooseDataset}
                    onOpenZoo={openZoo}
                    llm={llmProps}
                    runStatus={statusOf(s)}
                    device={deviceOf(s)}
                    gpus={gpus.map((g) => g.index)}
                    onDevice={(v) => chooseDevice(s, v)}
                  />
                ))}
                {active === NEW && (
                  <Chat
                    session={NEW}
                    active
                    recentFiles={fileRecent}
                    onAttach={recordFile}
                    onPickDataset={() => setDatasetBrowsing(true)}
                    datasetRecent={recent}
                    onChooseDataset={chooseDataset}
                    onOpenZoo={openZoo}
                    llm={llmProps}
                    device={deviceOf(NEW)}
                    gpus={gpus.map((g) => g.index)}
                    onDevice={(v) => chooseDevice(NEW, v)}
                    onSubmit={(text) => {
                      startExperiment(text);
                      return true;
                    }}
                  />
                )}
              </div>
            </div>
            <div className="col-divider" onMouseDown={startColDrag} title="Drag to resize" />
            <RightPanel
              session={active}
              volumePath={ui[active]?.volume ?? null}
              volumeNonce={ui[active]?.volumeNonce ?? 0}
              onVolumePathChange={(p) => setUi((u) => patchSession(u, active, { volume: p }))}
              comparePath={ui[active]?.compareVol ?? null}
              onComparePathChange={(p) => setUi((u) => patchSession(u, active, { compareVol: p || null }))}
              stream={stream}
              dataset={ui[active]?.dataset ?? ""}
              onReload={() => bumpRun(active)}
            />
          </div>
          <Console />
        </div>
      </div>

      <div className="statusbar">
        <span className="seg">
          <span className="dot" />
          {status}
        </span>
        <span className="seg dim">konfai-mcp{tools ? ` · ${tools} tools` : ""}</span>
        <span className="seg dim">
          {sessions.length} experiment{sessions.length === 1 ? "" : "s"}
        </span>
        <span className="seg dim">{apps.length} apps</span>
        <span className="push" />
        {remote ? (
          <>
            <span className="seg dim">private · your data stays on the server</span>
            <button className="seg signout" onClick={signOut} title="Sign out of Studio">
              Sign out
            </button>
          </>
        ) : (
          <span className="seg dim">
            {/* No offline claim per backend: the openai-compatible base_url may point anywhere. */}
            local · your imaging data stays on this machine
          </span>
        )}
      </div>

      {menu && (
        <>
          <div
            className="menu-back"
            onClick={() => setMenu(null)}
            onContextMenu={(e) => {
              e.preventDefault();
              setMenu(null);
            }}
          />
          <div className="ctx-menu" style={{ left: menu.x, top: menu.y }}>
            <button onClick={() => renameExperiment(menu.session)}>Rename…</button>
            <button
              disabled={menuCaps ? !menuCaps.bundlable : false}
              title={menuCaps && !menuCaps.bundlable ? "No checkpoints to package yet: train first" : undefined}
              onClick={() => {
                setPick({ session: menu.session, action: "bundle" });
                setMenu(null);
              }}
            >
              Bundle as app…
            </button>
            <button
              onClick={() => {
                setPick({ session: menu.session, action: "export" });
                setMenu(null);
              }}
            >
              Export experiment…
            </button>
            <button className="danger" onClick={() => deleteExperiment(menu.session)}>
              Delete experiment
            </button>
          </div>
        </>
      )}

      {datasetBrowsing && (
        <FolderBrowser
          start=""
          title="Choose a dataset folder"
          cta="Use this dataset"
          onPick={(p) => {
            setDatasetBrowsing(false);
            chooseDataset(p);
          }}
          onClose={() => setDatasetBrowsing(false)}
        />
      )}

      {appPick && (
        <FolderBrowser
          start=""
          title={`Choose a dataset for ${appPick}`}
          cta="Use app on this dataset"
          onPick={(p) => {
            const ref = appPick;
            setAppPick(null);
            useApp(ref, p); // launch with the dataset -> it's recorded and mounts in the workspace tree
          }}
          onClose={() => setAppPick(null)}
        />
      )}

      {pick && (
        <FolderBrowser
          start={ui[pick.session]?.dataset || ""}
          title={pick.action === "bundle" ? "Choose where to save the app bundle" : "Choose where to save the run record"}
          cta={pick.action === "bundle" ? "Bundle here" : "Export here"}
          onPick={(dest) => runAction(pick.session, pick.action, dest)}
          onClose={() => setPick(null)}
        />
      )}

      {zooOpen && (
        <AppZoo
          apps={apps}
          loading={appsLoading}
          onUse={(ref) => {
            setZooOpen(false);
            setAppPick(ref); // choose the dataset first, then launch the app on it
          }}
          onAdd={addApp}
          onRemove={removeApp}
          onClose={() => setZooOpen(false)}
          onDeploy={(app) => {
            setZooOpen(false);
            setDeployApp(app);
          }}
        />
      )}

      {deployApp && <Deploy app={deployApp} onClose={() => setDeployApp(null)} />}

      {toast && (
        <div className="toast" onClick={() => setToast("")}>
          {toast}
        </div>
      )}
      {dragging && <div className="drag-shield" />}
    </div>
  );
}
