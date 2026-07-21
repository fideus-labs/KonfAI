// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import FolderBrowser from "./FolderBrowser";
import { getJson } from "./api";
import { readSSE } from "./sse";

type Part =
  | { kind: "text"; text: string }
  | { kind: "error"; text: string }
  | { kind: "tool"; name: string; input: unknown; status: "running" | "done" | "error"; preview: string };

type Msg =
  | { id: number; role: "user"; text: string }
  | { id: number; role: "assistant"; parts: Part[] };

let nextId = 1;

// Chat transcripts persist per task so a page reload keeps the conversation (the agent, still running
// server-side, keeps its own context). Local, single-user — localStorage is enough.
const storeKey = (session: string) => `konfai-studio:chat:${session}`;

// Experiment ids are reused (a deleted experiment-1 frees the id), and the draft always reuses "__new__",
// so a fresh experiment must drop any stale chat left in localStorage under that key.
export function clearChat(session: string): void {
  try {
    localStorage.removeItem(storeKey(session));
  } catch {
    /* localStorage unavailable — nothing persisted anyway */
  }
}

function loadMessages(session: string): Msg[] {
  try {
    const saved = JSON.parse(localStorage.getItem(storeKey(session)) || "[]") as Msg[];
    for (const m of saved) if (m.id >= nextId) nextId = m.id + 1; // keep new ids unique
    return saved;
  } catch {
    return [];
  }
}

// Predictive next prompts. First choice = the MCP tool's own `next_actions`; else these per-phase starters.
type Phase = "start" | "explored" | "running";
const SUGGESTIONS: Record<Phase, string[]> = {
  start: ["List the available apps", "Inspect a dataset", "What can you do with my data?"],
  explored: ["Train a model from scratch", "Fine-tune an app on this data", "Preview one case in the viewer"],
  running: ["Show the training progress", "Run prediction on the validation set", "Evaluate the predictions"],
};

// An MCP `next_actions` entry is a tool name; phrase it as an instruction the user would actually type.
const ACTION_PROMPT: Record<string, string> = {
  inspect_dataset: "Inspect the dataset",
  browse_dataset: "Browse the dataset",
  list_apps: "List the available apps",
  run_train: "Start training",
  run_resume: "Resume training",
  run_prediction: "Run prediction",
  run_evaluation: "Evaluate the predictions",
  import_app: "Use this app in the session",
  describe_app: "Inspect the app",
  list_app_parameters: "Show the app's tunable parameters",
  get_run_metrics: "Show the evaluation scores",
  read_training_curves: "Show the training curves",
  read_live_metrics: "Show the live metrics",
  compare_runs: "Compare the runs",
  leaderboard: "Show the leaderboard",
  package_app_from_session: "Package this run as an app",
  export_run_record: "Export the run record",
  wait_for_job: "Wait for the job to finish",
  get_job_status: "Check the job status",
};

function actionPrompt(name: string): string {
  return ACTION_PROMPT[name] ?? name.replace(/_/g, " ");
}

function deviceLabel(device: string): string {
  if (device === "cpu") return "CPU";
  if (device === "auto") return "Auto";
  const gs = device.split(",").filter(Boolean);
  return `GPU ${gs.join(",")}`;
}

// Compute-device picker for the composer: Auto / CPU are exclusive; GPUs multi-select (DDP).
function DevicePicker({ device, gpus, onDevice }: { device: string; gpus: number[]; onDevice: (v: string) => void }) {
  const [open, setOpen] = useState(false);
  const selected = new Set(device.split(",").filter((p) => /^\d+$/.test(p)).map(Number));
  const toggleGpu = (idx: number) => {
    const next = new Set(selected);
    if (next.has(idx)) next.delete(idx);
    else next.add(idx);
    onDevice(next.size ? [...next].sort((a, b) => a - b).join(",") : "auto");
  };
  return (
    <div className="cbar-dev">
      <button
        type="button"
        className="cbar-sel"
        title="Which device(s) this experiment's jobs run on"
        onClick={() => setOpen((o) => !o)}
      >
        {deviceLabel(device)}
      </button>
      {open && (
        <>
          <div className="attach-back" onClick={() => setOpen(false)} />
          <div className="dev-pop">
            <button className={device === "auto" ? "dev-opt on" : "dev-opt"} onClick={() => { onDevice("auto"); setOpen(false); }}>
              Auto
            </button>
            <button className={device === "cpu" ? "dev-opt on" : "dev-opt"} onClick={() => { onDevice("cpu"); setOpen(false); }}>
              CPU
            </button>
            {gpus.length > 0 && <div className="dev-sep">GPUs · multi-select</div>}
            {gpus.map((idx) => (
              <button key={idx} className={selected.has(idx) ? "dev-opt on" : "dev-opt"} onClick={() => toggleGpu(idx)}>
                <span className="dev-check">{selected.has(idx) ? "✓" : ""}</span> GPU {idx}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

const attachPrompt = (ref: string) =>
  `I'm attaching the file at ${ref} as context for this experiment. Read it and tell me what it contains and how it applies to my task. If it's a research paper, reproduce its method as a KonfAI experiment — author the config and explain your choices, validate it, but don't launch training until I confirm.`;

export default function Chat({
  session,
  active,
  inject,
  recentFiles = [],
  onVolume,
  onRun,
  onTitle,
  onAttach,
  onSubmit,
  onBusy,
  onPickDataset,
  datasetRecent,
  onChooseDataset,
  onOpenZoo,
  llm,
  device,
  gpus,
  onDevice,
}: {
  session: string;
  active: boolean;
  inject?: { text: string; nonce: number };
  recentFiles?: string[];
  onVolume?: (path: string) => void;
  onRun?: () => void;
  onTitle?: (title: string) => void;
  onAttach?: (ref: string) => void;
  onSubmit?: (text: string) => boolean; // intercept the send (a draft materialises into a task)
  onBusy?: (busy: boolean) => void; // report "the agent is working" up to the rail
  onPickDataset?: () => void; // open the dataset browser
  datasetRecent?: string[];
  onChooseDataset?: (path: string) => void; // pick a recent dataset directly
  onOpenZoo?: () => void; // open the App Zoo window
  llm?: {
    brains: { id: string; label: string; available: boolean; models?: { id: string; label: string }[] }[];
    brain: string;
    model: string;
    modelText: string;
    onBrain: (id: string) => void;
    onModel: (id: string) => void;
    onModelText: (s: string) => void;
  };
  device?: string; // 'auto' | 'cpu' | GPU indices '0'/'0,1'
  gpus?: number[]; // available GPU indices
  onDevice?: (val: string) => void;
}) {
  const [messages, setMessages] = useState<Msg[]>(() => loadMessages(session));
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [phase, setPhase] = useState<Phase>("start");
  const [nextActions, setNextActions] = useState<string[]>([]);
  const [nextPrompts, setNextPrompts] = useState<{ label: string; prompt: string }[]>([]); // LLM: button + full prompt
  const [attaching, setAttaching] = useState(false);
  const [attachText, setAttachText] = useState("");
  const [browsing, setBrowsing] = useState(false);
  const [listening, setListening] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const logRef = useRef<HTMLDivElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const recRef = useRef<any>(null);
  const ctrlRef = useRef<AbortController | null>(null);

  // Abort an in-flight turn if this task is unmounted (e.g. the experiment is deleted mid-response),
  // so the read loop stops and stops writing back into the parent's state for a dead session.
  useEffect(() => () => ctrlRef.current?.abort(), []);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [messages]);

  useEffect(() => {
    onBusy?.(busy);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [busy]);

  useEffect(() => {
    try {
      localStorage.setItem(storeKey(session), JSON.stringify(messages.slice(-200)));
    } catch {
      /* quota / private mode — degrade silently */
    }
  }, [session, messages]);

  // A dataset chosen for this task injects an "inspect it" message into the conversation.
  useEffect(() => {
    if (inject && inject.nonce > 0) send(inject.text);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inject?.nonce]);

  function patchAssistant(id: number, fn: (parts: Part[]) => Part[]) {
    setMessages((prev) =>
      prev.map((m) => (m.id === id && m.role === "assistant" ? { ...m, parts: fn(m.parts) } : m)),
    );
  }

  async function send(override?: string) {
    const text = (override ?? input).trim();
    if (!text || busy) return;
    if (onSubmit?.(text)) {
      if (override === undefined) setInput("");
      return; // a draft: the parent creates the task and re-sends there
    }
    if (override === undefined) setInput("");
    setBusy(true);
    setNextActions([]); // superseded by this turn's own next_actions
    setNextPrompts([]);
    setMessages((prev) => [...prev, { id: nextId++, role: "user", text }]);
    const aid = nextId++;
    setMessages((prev) => [...prev, { id: aid, role: "assistant", parts: [] }]);

    const ctrl = new AbortController();
    ctrlRef.current = ctrl;
    try {
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, session }),
        signal: ctrl.signal,
      });
      for await (const ev of readSSE(resp)) handleEvent(aid, ev);
    } catch (e) {
      if (!(e instanceof DOMException && e.name === "AbortError"))
        patchAssistant(aid, (p) => [...p, { kind: "error", text: String(e) }]);
    } finally {
      setBusy(false);
    }
  }

  function handleEvent(aid: number, ev: Record<string, unknown>) {
    const type = ev.type as string;
    if (type === "text") {
      patchAssistant(aid, (parts) => {
        const last = parts[parts.length - 1];
        if (last && last.kind === "text") {
          return [...parts.slice(0, -1), { kind: "text", text: last.text + (ev.text as string) }];
        }
        return [...parts, { kind: "text", text: ev.text as string }];
      });
    } else if (type === "tool_call") {
      const name = ev.name as string;
      patchAssistant(aid, (parts) => [...parts, { kind: "tool", name, input: ev.input, status: "running", preview: "" }]);
      if (name.startsWith("run_") || name.startsWith("fine_tune")) {
        onRun?.();
        setPhase("running");
      }
    } else if (type === "tool_result") {
      patchAssistant(aid, (parts) => {
        const copy = [...parts];
        for (let i = copy.length - 1; i >= 0; i--) {
          const p = copy[i];
          if (p.kind === "tool" && p.name === ev.name && p.status === "running") {
            copy[i] = { ...p, status: ev.ok ? "done" : "error", preview: ev.preview as string };
            break;
          }
        }
        return copy;
      });
      setPhase((p) => (p === "running" ? p : "explored"));
    } else if (type === "volume") {
      onVolume?.(ev.path as string);
      setPhase((p) => (p === "running" ? p : "explored"));
    } else if (type === "next_actions") {
      setNextActions((ev.actions as string[]) ?? []);
    } else if (type === "next_prompts") {
      setNextPrompts((ev.prompts as { label: string; prompt: string }[]) ?? []);
    } else if (type === "title") {
      onTitle?.(ev.title as string);
    } else if (type === "error") {
      patchAssistant(aid, (parts) => [...parts, { kind: "error", text: ev.message as string }]);
    }
  }

  function attach(ref: string) {
    const value = ref.trim();
    setAttaching(false);
    setBrowsing(false);
    setAttachText("");
    if (!value) return;
    onAttach?.(value);
    send(attachPrompt(value));
  }

  // Accept anything dropped and route it to the LLM. A local folder becomes a dataset; a local file
  // (paper, config) is attached; a URL / arXiv id goes through the read-and-reproduce flow; any other
  // text — a link, a note, a snippet — is dropped into the composer so it rides the next message.
  async function stat(path: string): Promise<{ exists: boolean; is_dir: boolean }> {
    try {
      return await getJson<{ exists: boolean; is_dir: boolean }>(`/api/stat?path=${encodeURIComponent(path)}`);
    } catch {
      return { exists: false, is_dir: false };
    }
  }

  async function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    // getData must be read synchronously, before any await
    const raw = (e.dataTransfer.getData("text/uri-list") || e.dataTransfer.getData("text/plain") || "").trim();
    if (!raw) return;
    let first = raw.split(/\r?\n/).find((l) => l && !l.startsWith("#")) ?? "";

    if (first.startsWith("file://")) first = decodeURIComponent(first.replace(/^file:\/\/(localhost)?/, ""));
    const looksLocal = first.startsWith("/") || /^[A-Za-z]:[\\/]/.test(first);
    if (looksLocal) {
      const d = await stat(first);
      if (d.exists && d.is_dir) return onChooseDataset?.(first);
      if (d.exists) return attach(first);
    }
    if (/^https?:\/\//i.test(first) || /arxiv\.org|^\d{4}\.\d{4,5}(v\d+)?$/i.test(first)) return attach(first);

    setInput((v) => (v ? `${v}\n${raw}` : raw));
  }

  // Voice dictation via the browser's Web Speech API — no dependency, nothing leaves the device
  // beyond the browser's own speech service. Absent on unsupported browsers, so the mic is hidden.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const SpeechRecognition = (typeof window !== "undefined" && ((window as any).SpeechRecognition || (window as any).webkitSpeechRecognition)) || null;

  function toggleDictation() {
    if (listening) {
      recRef.current?.stop();
      return;
    }
    try {
      const rec = new SpeechRecognition();
      rec.lang = navigator.language || "en-US";
      rec.interimResults = true;
      rec.continuous = true;
      const base = input ? input.trim() + " " : "";
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      rec.onresult = (e: any) => {
        let text = "";
        for (let i = 0; i < e.results.length; i++) text += e.results[i][0].transcript;
        setInput(base + text);
      };
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      rec.onerror = (e: any) => {
        setListening(false);
        if (e?.error === "not-allowed" || e?.error === "service-not-allowed")
          setInput((v) => v || "(microphone blocked — allow mic access for this page)");
      };
      rec.onend = () => setListening(false);
      recRef.current = rec;
      setListening(true);
      rec.start();
    } catch {
      setListening(false);
    }
  }

  const lastId = messages[messages.length - 1]?.id;

  return (
    <section className={active ? "pane chat" : "pane chat hidden"}>
      <div className="log" ref={logRef}>
        {messages.length === 0 && (
          <div className="welcome">
            <img className="welcome-logo" src="/konfai-logo.png" alt="KonfAI" />
            <div className="welcome-title">Start a new experiment</div>
            <div className="welcome-sub">Point Studio at your data, then describe the task in plain words.</div>
            {onPickDataset && (
              <button className="welcome-cta" onClick={onPickDataset}>
                Choose a dataset…
              </button>
            )}
            {datasetRecent && datasetRecent.length > 0 && (
              <div className="welcome-recent">
                <span className="welcome-recent-lead">Recent</span>
                {datasetRecent.slice(0, 4).map((d) => (
                  <button key={d} className="welcome-recent-item" onClick={() => onChooseDataset?.(d)} title={d}>
                    {d.split("/").pop() || d}
                  </button>
                ))}
              </div>
            )}
            {onOpenZoo && (
              <button className="welcome-zoo" onClick={onOpenZoo}>
                Browse KonfAI Apps →
              </button>
            )}
            <div className="welcome-or">or just describe a task below</div>
          </div>
        )}
        {messages.map((m) =>
          m.role === "user" ? (
            <div key={m.id} className="u">
              {m.text}
            </div>
          ) : (
            <div key={m.id} className="a">
              <div className="who">KonfAI</div>
              {m.parts.length === 0 && busy && m.id === lastId ? (
                <div className="thinking" aria-label="waiting for the model">
                  <i />
                  <i />
                  <i />
                </div>
              ) : (
                m.parts.map((p, i) =>
                  p.kind === "text" ? (
                    <div key={i} className="text md">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{p.text}</ReactMarkdown>
                    </div>
                  ) : p.kind === "error" ? (
                    <div key={i} className="err">
                      ⚠ {p.text}
                    </div>
                  ) : (
                    <div key={i} className={`tool ${p.status}`}>
                      <div className="th">
                        <span className="d" />
                        <span className="nm">{p.name}</span>
                        <span className="st">{p.status === "running" ? "running…" : p.status}</span>
                      </div>
                      <pre>{p.preview || JSON.stringify(p.input)}</pre>
                    </div>
                  ),
                )
              )}
            </div>
          ),
        )}
      </div>
      {!busy &&
        (() => {
          // The LLM's proposed next prompts win — a short label on the button, the full prompt sent on click.
          // Otherwise the MCP tool's next_actions, otherwise the static per-phase starters.
          const chips: { key: string; label: string; prompt: string }[] = nextPrompts.length
            ? nextPrompts.map((p) => ({ key: p.label, label: p.label, prompt: p.prompt }))
            : (nextActions.length ? nextActions.map(actionPrompt) : SUGGESTIONS[phase]).map((s) => ({
                key: s,
                label: s,
                prompt: s,
              }));
          const seen = new Set<string>();
          const shown = chips.filter((c) => !seen.has(c.key) && seen.add(c.key));
          const led = nextPrompts.length > 0 || nextActions.length > 0;
          return (
            <div className="suggest">
              {led && <span className="suggest-lead">Next</span>}
              {shown.map((c) => (
                <button key={c.key} className="chip-prompt" title={c.prompt} onClick={() => send(c.prompt)}>
                  {c.label}
                </button>
              ))}
            </div>
          );
        })()}
      <div
        className={dragOver ? "composer drag" : "composer"}
        onDragOver={(e) => {
          e.preventDefault();
          if (!dragOver) setDragOver(true);
        }}
        onDragLeave={(e) => {
          if (e.currentTarget === e.target) setDragOver(false);
        }}
        onDrop={onDrop}
      >
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          placeholder="Describe a task…  (Enter to send, Shift+Enter for newline)"
        />
        {dragOver && <div className="drag-hint">Drop anything — a dataset, a paper, a link or notes</div>}
        <div className="composer-bar">
          <button
            className="cbtn"
            onClick={() => setAttaching((a) => !a)}
            title="Add files — paper, config, notes"
            aria-label="Add files"
          >
            +
          </button>
          {llm && (
            <div className="cbar-llm">
              <select
                className="cbar-sel"
                value={llm.brain}
                onChange={(e) => llm.onBrain(e.target.value)}
                title="Which model runs the assistant"
              >
                {llm.brains.map((b) => (
                  <option key={b.id} value={b.id} disabled={!b.available}>
                    {b.label}
                    {b.available ? "" : " — unavailable"}
                  </option>
                ))}
              </select>
              {(llm.brains.find((b) => b.id === llm.brain)?.models ?? []).length > 0 ? (
                <select className="cbar-sel" value={llm.model} onChange={(e) => llm.onModel(e.target.value)} title="Model">
                  {(llm.brains.find((b) => b.id === llm.brain)?.models ?? []).map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.label}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  className="cbar-sel cbar-model-free"
                  value={llm.modelText}
                  placeholder="model…"
                  onChange={(e) => llm.onModelText(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && llm.onModel(llm.modelText.trim())}
                  onBlur={() => llm.modelText.trim() !== llm.model && llm.onModel(llm.modelText.trim())}
                />
              )}
            </div>
          )}
          {onDevice && <DevicePicker device={device ?? "auto"} gpus={gpus ?? []} onDevice={onDevice} />}
          <span className="cbar-spacer" />
          {
            <button
              className={listening ? "cbtn mic on" : "cbtn mic"}
              disabled={!SpeechRecognition}
              onClick={toggleDictation}
              title={
                SpeechRecognition
                  ? listening
                    ? "Stop dictation"
                    : "Dictate"
                  : "Dictation needs the Web Speech API — use Chrome, Chromium or Edge"
              }
              aria-label="Dictate"
            >
              <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <rect x="9" y="3" width="6" height="11" rx="3" />
                <path d="M5 11a7 7 0 0 0 14 0" />
                <line x1="12" y1="18" x2="12" y2="21" />
              </svg>
            </button>
          }
          {busy ? (
            <button className="csend stop" onClick={() => ctrlRef.current?.abort()} title="Stop this response" aria-label="Stop">
              <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">
                <rect x="6" y="6" width="12" height="12" rx="2.5" fill="currentColor" />
              </svg>
            </button>
          ) : (
            <button
              className={input.trim() ? "csend ready" : "csend"}
              onClick={() => send()}
              title="Send"
              aria-label="Send"
            >
              <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
                <path
                  d="M12 19V5M12 5l-6 6M12 5l6 6"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
          )}
        </div>
        {attaching && <div className="attach-back" onClick={() => setAttaching(false)} />}
        {attaching && (
          <div className="attach-pop">
            {onPickDataset && (
              <button
                className="attach-browse"
                onClick={() => {
                  setAttaching(false);
                  onPickDataset();
                }}
              >
                Choose a dataset folder…
              </button>
            )}
            <button
              className="attach-browse"
              onClick={() => {
                setAttaching(false);
                setBrowsing(true);
              }}
            >
              Browse files…
            </button>
            <input
              className="paper-input"
              autoFocus
              value={attachText}
              onChange={(e) => setAttachText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") attach(attachText);
                if (e.key === "Escape") setAttaching(false);
              }}
              placeholder="…or paste a path / URL (arXiv, PDF, config)"
            />
            {recentFiles.length > 0 && <div className="dsmenu-head">Recent</div>}
            {recentFiles.map((r) => (
              <button key={r} className="dsitem" onClick={() => attach(r)} title={r}>
                {r}
              </button>
            ))}
          </div>
        )}
        {browsing && (
          <FolderBrowser
            start=""
            onPickFile={(p) => attach(p)}
            onClose={() => setBrowsing(false)}
            title="Choose a file to attach"
          />
        )}
      </div>
    </section>
  );
}
