// SPDX-License-Identifier: Apache-2.0

import { Component, useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import Viewer from "./Viewer";
import type { JobStream, LiveStatus, Point, RunFeed, Series } from "./useJobStream";
import { getJson, postJson } from "./api";
import { useJson } from "./useJson";
import { isRunning, jobState } from "./status";

// A render error in the feed must never blank the whole app: catch it, show a line, and retry when new
// data arrives (resetKey changes on each stream update).
class FeedBoundary extends Component<{ resetKey: unknown; children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  componentDidUpdate(prev: { resetKey: unknown }) {
    if (this.state.failed && prev.resetKey !== this.props.resetKey) this.setState({ failed: false });
  }
  render() {
    if (this.state.failed) return <div className="feed-sub">A panel hit a display error: recovering…</div>;
    return this.props.children;
  }
}

function curve(points: Point[], w: number, h: number): { line: string; area: string } {
  const ys = points.map((p) => p.y);
  const xs = points.map((p) => p.x);
  const ymin = Math.min(...ys);
  const yr = Math.max(...ys) - ymin || 1;
  const xmin = Math.min(...xs);
  const xr = Math.max(...xs) - xmin || 1;
  const pts = points.map((p) => `${(((p.x - xmin) / xr) * w).toFixed(1)},${(h - ((p.y - ymin) / yr) * h).toFixed(1)}`);
  return { line: pts.join(" "), area: `${pts.join(" ")} ${w},${h} 0,${h}` };
}

function fmt(v: number | undefined): string {
  return typeof v === "number" ? v.toFixed(4) : ": ";
}

// The last segment names the metric ("PRED:SEG:Dice" -> "Dice"), unless it is an index: a bare number
// on its own titles a chart with nothing, so it keeps the metric it indexes ("PRED:SEG:Dice:1" -> "Dice:1").
function shortMetric(name: string): string {
  const parts = name.split(/[:/]/).filter(Boolean);
  const last = parts[parts.length - 1];
  if (last === undefined) return name;
  return parts.length > 1 && /^\d+$/.test(last) ? `${parts[parts.length - 2]}:${last}` : last;
}

// Close a modal on Escape.
function useEscapeToClose(onClose: () => void) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
}

// Wall-clock HH:MM:SS, when a feed item was last produced.
function fmtClock(ms: number): string {
  return new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

// A run's status at a glance: a spinner while it runs, a coloured dot once it settles (green done, red
// error): so "still going" never looks like "finished".
function StatusMark({ status }: { status: string }) {
  if (isRunning(status)) return <span className="rt-spin" title={status} />;
  return <span className={`rt-dot ${status}`} title={status} />;
}

// A compact y-axis tick label that stays readable across metric scales (losses ~1e-3, counts ~1e2).
function yTick(v: number): string {
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return v.toFixed(2);
  if (a >= 0.01) return v.toFixed(3);
  return v.toExponential(1);
}

const W = 320;
const H = 120;

// An interactive TensorBoard-style chart: hover for the value at a point, drag horizontally to zoom into
// an x-range, double-click to reset.
function ZoomChart({ series }: { series: Series }) {
  const [range, setRange] = useState<[number, number] | null>(null);
  const [hover, setHover] = useState<{ px: number; py: number; p: Point } | null>(null);
  const [sel, setSel] = useState<{ a: number; b: number } | null>(null);
  const ref = useRef<SVGSVGElement>(null);
  const VW = 1000;
  const VH = 420;
  const pts = series.points;
  const perCase = !(series.stage === "training" || series.stage === "validation"); // markers only: see MetricChart
  const xUnit = perCase ? "case" : "iteration";
  if (pts.length === 0) return null;
  const xs = pts.map((p) => p.x);
  const [xmin, xmax] = range ?? [Math.min(...xs), Math.max(...xs)];
  const vis = pts.filter((p) => p.x >= xmin && p.x <= xmax);
  const ys = (vis.length ? vis : pts).map((p) => p.y);
  const ymin = Math.min(...ys);
  const ymax = Math.max(...ys);
  const xr = xmax - xmin || 1;
  const yr = ymax - ymin || 1;
  const X = (x: number) => ((x - xmin) / xr) * VW;
  const Y = (y: number) => VH - ((y - ymin) / yr) * VH;
  const color = lineColor(series.label);
  const pxAt = (e: React.MouseEvent) => {
    const r = ref.current!.getBoundingClientRect();
    return ((e.clientX - r.left) / r.width) * VW;
  };
  const onMove = (e: React.MouseEvent) => {
    const xval = xmin + (pxAt(e) / VW) * xr;
    let near: Point | null = null;
    for (const p of vis) if (!near || Math.abs(p.x - xval) < Math.abs(near.x - xval)) near = p;
    setHover(near ? { px: X(near.x), py: Y(near.y), p: near } : null);
    if (sel) setSel({ ...sel, b: pxAt(e) });
  };
  const onUp = () => {
    if (sel && Math.abs(sel.b - sel.a) > 12) {
      setRange([xmin + (Math.min(sel.a, sel.b) / VW) * xr, xmin + (Math.max(sel.a, sel.b) / VW) * xr]);
    }
    setSel(null);
  };
  const line = vis.map((p) => `${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join(" ");
  return (
    <div className="zoomchart">
      <svg
        ref={ref}
        className="zoom-svg"
        viewBox={`0 0 ${VW} ${VH}`}
        preserveAspectRatio="none"
        onMouseMove={onMove}
        onMouseDown={(e) => setSel({ a: pxAt(e), b: pxAt(e) })}
        onMouseUp={onUp}
        onMouseLeave={() => {
          setHover(null);
          setSel(null);
        }}
        onDoubleClick={() => setRange(null)}
      >
        {[0.25, 0.5, 0.75].map((f) => (
          <line key={f} className="grid" x1="0" x2={VW} y1={VH * f} y2={VH * f} vectorEffect="non-scaling-stroke" />
        ))}
        {!perCase && (
          <polyline className="line" points={line} vectorEffect="non-scaling-stroke" style={{ stroke: color }} />
        )}
        {(perCase || vis.length <= 80) &&
          vis.map((p, i) => <circle key={i} cx={X(p.x)} cy={Y(p.y)} r="3" style={{ fill: color }} vectorEffect="non-scaling-stroke" />)}
        {sel && <rect className="zoom-sel" x={Math.min(sel.a, sel.b)} y="0" width={Math.abs(sel.b - sel.a)} height={VH} />}
        {hover && <line className="zoom-cross" x1={hover.px} x2={hover.px} y1="0" y2={VH} vectorEffect="non-scaling-stroke" />}
        {hover && <circle cx={hover.px} cy={hover.py} r="5" style={{ fill: color }} vectorEffect="non-scaling-stroke" />}
      </svg>
      <span className="zoom-yhi">{yTick(ymax)}</span>
      <span className="zoom-ylo">{yTick(ymin)}</span>
      {hover && (
        <div className="zoom-tip" style={{ left: `${(hover.px / VW) * 100}%`, top: `${(hover.py / VH) * 100}%` }}>
          {xUnit} {hover.p.x} · <b>{hover.p.y.toFixed(5)}</b>
        </div>
      )}
      <div className="zoom-hint">{range ? "double-click to reset" : "drag to zoom · hover for values"}</div>
    </div>
  );
}

// Click a chart → zoom it: an interactive, TensorBoard-style view of the same live curve.
function CurveZoom({ id, series, runName, onClose }: { id: string; series: Series; runName: string; onClose: () => void }) {
  useEscapeToClose(onClose);
  return (
    <div className="lightbox" onClick={onClose}>
      <div className="curve-zoom" onClick={(e) => e.stopPropagation()}>
        <div className="curve-zoom-head">
          <span>
            {id} · {series.label} · <span className="dim">{runName}</span>
          </span>
          <button onClick={onClose}>✕</button>
        </div>
        <ZoomChart series={series} />
      </div>
    </div>
  );
}

type Preview = { label: string; steps: number[]; step: number };

// One output montage with its step history: the latest by default, a slider to scrub back. Frames are
// fetched lazily per step (never inlined), so a long history stays cheap.
function SampleRow({ session, base, type, label, steps, onOpen }: { session: string; base: string; type: string; label: string; steps: number[]; onOpen: (url: string) => void }) {
  const [sel, setSel] = useState<number | null>(null); // null = follow the latest step as new ones arrive
  const idx = sel == null ? steps.length - 1 : Math.min(sel, steps.length - 1);
  const step = steps[idx];
  const src = `/api/preview_image?session=${encodeURIComponent(session)}&base=${encodeURIComponent(base)}&tag=${encodeURIComponent(label)}&step=${step}`;
  return (
    <div className="sample-row">
      <div className="sample-row-head">
        <span className="sample-type">{type}</span>
        <span className="sample-step">
          step {step}
          {sel == null && steps.length > 1 && <span className="sample-live"> · live</span>}
        </span>
      </div>
      <button className="sample-big" onClick={() => onOpen(src)} title={`${label} · step ${step}: click to enlarge`}>
        <img src={src} alt={label} />
      </button>
      {steps.length > 1 && (
        <input
          className="sample-slider"
          type="range"
          min={0}
          max={steps.length - 1}
          value={idx}
          onChange={(e) => setSel(Number(e.target.value) === steps.length - 1 ? null : Number(e.target.value))}
          title="Scrub the step history"
        />
      )}
    </div>
  );
}

// Thumbnails of what the model is producing (TensorBoard image summaries). Click enlarges.
function Samples({
  session,
  base,
  refresh,
  onOpen,
}: {
  session: string;
  base: string;
  refresh: number;
  onOpen: (url: string) => void;
}) {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const id = window.setInterval(() => setTick((t) => t + 1), 12000); // konfai writes sample images mid-run: pick them up as they land
    return () => window.clearInterval(id);
  }, []);
  const { data } = useJson<{ previews?: Preview[] }>(
    `/api/previews?session=${encodeURIComponent(session)}&base=${encodeURIComponent(base)}`,
    [session, base, refresh, tick],
  );
  const previews = data?.previews ?? [];

  if (previews.length === 0) return null;
  // One card per phase (Training / Validation), each assembling its outputs as labelled rows (CT, MR,
  // Head.Tanh) with a step-history slider. The montage's own tag becomes the row label; phase the card.
  const byPhase = new Map<string, { type: string; label: string; steps: number[] }[]>();
  for (const p of previews) {
    const [phase, ...rest] = p.label.split("/");
    const row = { type: rest.join("/") || phase, label: p.label, steps: p.steps };
    (byPhase.get(phase) ?? byPhase.set(phase, []).get(phase)!).push(row);
  }
  const phases = [...byPhase.entries()].sort(([a], [b]) => (/valid/i.test(a) ? 1 : 0) - (/valid/i.test(b) ? 1 : 0));
  return (
    <section className="feed-sec">
      <div className="feed-label">Samples · model outputs</div>
      <div className="sample-cards">
        {phases.map(([phase, rows]) => (
          <div key={phase} className="sample-card">
            <div className="sample-card-head">
              <span className={`stage ${/valid/i.test(phase) ? "validation" : "training"}`}>{phase}</span>
            </div>
            <div className="sample-figure">
              {rows
                .sort((a, b) => a.type.localeCompare(b.type))
                .map((r) => (
                  <SampleRow key={r.label} session={session} base={base} type={r.type} label={r.label} steps={r.steps} onOpen={onOpen} />
                ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

type EvalMetric = { name: string; direction?: string; mean?: number; std?: number; min?: number; max?: number };
type EvalCase = { case: string; values: Record<string, number> };
type EvalRun = {
  run: string;
  split: string;
  metrics: EvalMetric[];
  cases: number;
  case_metrics: string[];
  case_rows: EvalCase[];
};

function EvaluationView({
  session,
  refresh,
  hideWhenEmpty,
  onHasData,
  onOpenCase,
}: {
  session: string;
  refresh: number;
  hideWhenEmpty: boolean;
  onHasData?: (has: boolean) => void;
  onOpenCase?: (run: string, caseName: string) => void;
}) {
  const { data, loading } = useJson<{ runs?: EvalRun[] }>(
    `/api/evaluations?session=${encodeURIComponent(session)}`,
    [session, refresh],
  );
  const runs: EvalRun[] = data?.runs ?? [];
  const loaded = !loading;
  useEffect(() => {
    if (loading || !data) return; // report only on a real load, never on the error path
    onHasData?.((data.runs ?? []).length > 0);
  }, [loading, data, onHasData]);

  if (runs.length === 0) {
    if (hideWhenEmpty || !loaded) return null;
    return <div className="empty">no evaluation yet: run an evaluation and its scores will appear here.</div>;
  }

  return (
    <div className="evals">
      {runs.map((r, i) => (
        <div key={`${r.run}-${r.split}-${i}`} className="evrun">
          <div className="evrun-head">
            <span className="evname" title={r.run}>
              {r.run}
            </span>
            <span className="evsplit">{r.split}</span>
            <span className="evcases">
              {r.cases} case{r.cases === 1 ? "" : "s"}
            </span>
          </div>
          <table className="evtable">
            <thead>
              <tr>
                <th>metric</th>
                <th>mean</th>
                <th>std</th>
                <th>min</th>
                <th>max</th>
              </tr>
            </thead>
            <tbody>
              {r.metrics.map((m) => (
                <tr key={m.name} title={m.name}>
                  <td className="evm">
                    {shortMetric(m.name)}
                    {m.direction === "max" && <span className="dir up">↑</span>}
                    {m.direction === "min" && <span className="dir down">↓</span>}
                  </td>
                  <td className="num strong">{fmt(m.mean)}</td>
                  <td className="num">{fmt(m.std)}</td>
                  <td className="num">{fmt(m.min)}</td>
                  <td className="num">{fmt(m.max)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {r.case_rows.length > 0 && (
            <details className="evcase-wrap">
              <summary>
                Per case <span className="dim">({r.case_rows.length})</span>
              </summary>
              <div className="evcase-scroll">
                <table className="evtable cases">
                  <thead>
                    <tr>
                      <th>case</th>
                      {r.case_metrics.map((m) => (
                        <th key={m} title={m}>
                          {shortMetric(m)}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {r.case_rows.map((c) => (
                      <tr
                        key={c.case}
                        className={onOpenCase ? "case-row" : undefined}
                        onClick={onOpenCase ? () => onOpenCase(r.run, c.case) : undefined}
                        title={onOpenCase ? "Open this case's prediction" : c.case}
                      >
                        <td className="evcase" title={c.case}>
                          {c.case}
                        </td>
                        {r.case_metrics.map((m) => (
                          <td key={m} className="num">
                            {fmt(c.values[m])}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          )}
        </div>
      ))}
    </div>
  );
}

type ExpInfo = {
  checkpoints: string[];
  predictions: string[];
  jobs: { run?: string; kind?: string; status?: string }[];
  dataset?: string;
};
type Listing = { root: string; dirs: string[]; files: { name: string; size?: number }[] };

const VOLUME_FILE_RE = /\.(nii(\.gz)?|mha|mhd|nrrd)$/i;

function fmtSize(b?: number): string {
  if (b == null) return "";
  if (b < 1024) return `${b} B`;
  if (b < 1048576) return `${(b / 1024).toFixed(1)} KB`;
  if (b < 1073741824) return `${(b / 1048576).toFixed(1)} MB`;
  return `${(b / 1073741824).toFixed(2)} GB`;
}

// One directory node, listed lazily on expand. `list` abstracts where it reads from: the jailed
// experiment workspace or the (read-only) dataset folder outside it.
function DirNode({
  list,
  rel,
  name,
  depth,
  sel,
  focusRel,
  refreshKey,
  onFile,
}: {
  list: (rel: string) => Promise<Listing>;
  rel: string;
  name: string;
  depth: number;
  sel: string;
  focusRel: string;
  refreshKey: number;
  onFile: (rel: string, abs: string) => void;
}) {
  const [open, setOpen] = useState(depth === 0);
  const [ls, setLs] = useState<Listing | null>(null);

  // Re-lists on open, and again whenever refreshKey bumps (a running job wrote new files).
  useEffect(() => {
    if (!open) return;
    list(rel)
      .then(setLs)
      .catch(() => setLs({ root: "", dirs: [], files: [] }));
  }, [open, rel, list, refreshKey]);

  // Auto-expand along the path to the focused volume so it reveals itself in the tree.
  useEffect(() => {
    if (focusRel && !open && (rel === "" || focusRel === rel || focusRel.startsWith(rel + "/"))) setOpen(true);
  }, [focusRel, open, rel]);

  return (
    <div className="tnode">
      {depth > 0 && (
        <button className="trow dir" style={{ paddingLeft: depth * 14 }} onClick={() => setOpen((o) => !o)}>
          <span className="tarrow">{open ? "▾" : "▸"}</span> {name}
        </button>
      )}
      {open &&
        ls &&
        ls.dirs.map((d) => (
          <DirNode
            key={d}
            list={list}
            rel={rel ? `${rel}/${d}` : d}
            name={d}
            depth={depth + 1}
            sel={sel}
            focusRel={focusRel}
            refreshKey={refreshKey}
            onFile={onFile}
          />
        ))}
      {open &&
        ls &&
        ls.files.map((f) => {
          const frel = rel ? `${rel}/${f.name}` : f.name;
          const volume = VOLUME_FILE_RE.test(f.name);
          return (
            <button
              key={f.name}
              className={sel === frel || focusRel === frel ? "trow file on" : volume ? "trow file vol" : "trow file"}
              style={{ paddingLeft: (depth + 1) * 14 }}
              onClick={() => onFile(frel, `${ls.root}/${frel}`)}
              title={volume ? `${frel}: click to open (or fill the compare pane when Compare is on)` : frel}
            >
              <span className="tf-name">{f.name}</span>
              <span className="tf-size">{fmtSize(f.size)}</span>
            </button>
          );
        })}
    </div>
  );
}

// The whole experiment folder (configs, dataset, checkpoints, predictions, statistics), as a lazy
// tree, with one adaptive content pane: YAML/text opens in the editor (atomic jailed save for .yml),
// a volume opens inline in NiiVue. No separate Viewer tab: the file's type decides how it's shown.
function ExperimentView({
  session,
  volumePath,
  onVolumePathChange,
  refresh,
  dataset: picked,
  live,
  visible,
  focusDir,
  focusOpen,
  comparePath,
  onComparePathChange,
}: {
  session: string;
  volumePath: string | null;
  onVolumePathChange: (p: string) => void;
  refresh?: number;
  dataset?: string; // re-read the experiment when one is picked: it mounts the input tree and a chip
  live?: boolean;
  visible?: boolean; // the Workspace pane is on screen: keep its tree fresh even with no job running
  focusDir?: string; // a workspace dir to reveal/expand in the tree (from a run's "Browse files")
  focusOpen?: string; // what to auto-open there: "config" | "volume" | "metric" | "first"
  comparePath?: string | null;
  onComparePathChange?: (p: string) => void;
}) {
  const [info, setInfo] = useState<ExpInfo | null>(null);
  const [refreshKey, setRefreshKey] = useState(0); // bumped to re-list open dirs (job writing files)
  const [sel, setSel] = useState("");
  const [doc, setDoc] = useState<{ name: string; content: string; editable: boolean } | null>(null);
  const [draft, setDraft] = useState("");
  const [dirty, setDirty] = useState(false);
  const [stale, setStale] = useState(false); // the open file changed on disk; saving over it was refused
  const [note, setNote] = useState("");
  const [treeKey, setTreeKey] = useState(0);
  const [showVolume, setShowVolume] = useState(false);
  const [compareMode, setCompareMode] = useState(false); // when on, a tree volume fills the second pane

  // The chat can ask for a comparison ("show me the sCT next to the real CT"): a second volume arriving
  // opens the compare pane by itself, rather than being loaded into a view nobody switched on.
  useEffect(() => {
    if (comparePath) {
      setCompareMode(true);
      setShowVolume(true);
    }
  }, [comparePath]);
  const [root, setRoot] = useState(""); // absolute workspace root, learned from the tree

  const dataset = info?.dataset || "";

  // List one directory of the workspace (jailed) or of the dataset folder (read-only, outside it).
  const wsList = useCallback(
    (rel: string): Promise<Listing> =>
      fetch(`/api/experiment/ls?session=${encodeURIComponent(session)}&path=${encodeURIComponent(rel)}`)
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error("ls failed"))))
        .then((d: Listing) => {
          if (d.root) setRoot(d.root);
          return d;
        }),
    [session],
  );
  const dsList = useCallback(
    (rel: string): Promise<Listing> =>
      fetch(`/api/browse?path=${encodeURIComponent(rel ? `${dataset}/${rel}` : dataset)}`)
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error("browse failed"))))
        .then((d: { dirs?: string[]; files?: string[] }) => ({
          root: dataset,
          dirs: d.dirs ?? [],
          files: (d.files ?? []).map((n) => ({ name: n })),
        })),
    [dataset],
  );

  // The focused volume, made relative to whichever tree it lives under, that tree auto-expands to it.
  const relTo = (base: string) =>
    base && volumePath && volumePath.startsWith(base + "/") ? volumePath.slice(base.length + 1) : "";
  const wsFocus = relTo(root);
  const dsFocus = relTo(dataset);

  useEffect(() => {
    setInfo(null);
    setSel("");
    setDoc(null);
    setDirty(false);
    setNote("");
    setShowVolume(false);
    setTreeKey((k) => k + 1); // fresh tree per task
    getJson(`/api/experiment?session=${encodeURIComponent(session)}`)
      .then(setInfo)
      .catch(() => setInfo(null));
  }, [session]);

  // Refresh the overview + re-list open dirs when a job milestone fires or a live tick elapses, so
  // new checkpoints / statistics / logs appear in the tree without collapsing it. Picking a dataset
  // counts: it is what mounts the input tree beside the workspace, and the overview names it.
  useEffect(() => {
    if (refresh === undefined && !picked) return;
    setRefreshKey((k) => k + 1);
    getJson(`/api/experiment?session=${encodeURIComponent(session)}`)
      .then(setInfo)
      .catch(() => {});
  }, [refresh, picked, session]);

  useEffect(() => {
    // Not only while a job runs: the agent writes configs and code between jobs (a turn's
    // write_session_file), and a tree that only refreshes on job life left those files invisible
    // for minutes. Listing a local directory every 4s while the pane is on screen costs nothing.
    if (!live && !visible) return;
    const id = setInterval(() => setRefreshKey((k) => k + 1), 4000);
    return () => clearInterval(id);
  }, [live, visible]);

  // A volume becoming focused (tree click routes through onVolumePathChange, or the agent loads one)
  // brings the inline viewer forward.
  useEffect(() => {
    if (volumePath) setShowVolume(true);
  }, [volumePath]);

  // "Browse files" from a run tab: reveal the run's folder and auto-open its key artifact (a config in the
  // editor, a prediction volume in NiiVue, Metric_TRAIN.json in the editor) routed through openFile.
  useEffect(() => {
    if (!focusDir) return;
    let alive = true;
    const pick = (files: { name: string }[]) => {
      if (focusOpen === "config")
        return files.find((f) => /config.*\.ya?ml$/i.test(f.name)) ?? files.find((f) => /\.ya?ml$/i.test(f.name));
      if (focusOpen === "volume") return files.find((f) => VOLUME_FILE_RE.test(f.name));
      if (focusOpen === "metric")
        return files.find((f) => /metric_train\.json$/i.test(f.name)) ?? files.find((f) => /^metric_.*\.json$/i.test(f.name));
      return files[0];
    };
    const find = async (rel: string, depth = 0): Promise<{ rel: string; abs: string } | null> => {
      if (depth > 4) return null;
      const listing = await wsList(rel).catch(() => null);
      if (!listing) return null;
      const hit = pick(listing.files);
      if (hit) {
        const frel = rel ? `${rel}/${hit.name}` : hit.name;
        return { rel: frel, abs: `${listing.root}/${frel}` };
      }
      for (const sub of listing.dirs) {
        const found = await find(rel ? `${rel}/${sub}` : sub, depth + 1);
        if (found) return found;
      }
      return null;
    };
    find(focusDir).then((target) => alive && target && openFile(target.rel, target.abs));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusDir, focusOpen]);

  function load(rel: string) {
    setNote("");
    setStale(false);
    fetch(`/api/experiment/file?session=${encodeURIComponent(session)}&path=${encodeURIComponent(rel)}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("unreadable file"))))
      .then((d) => {
        setDoc(d);
        setDraft(d.content);
        setDirty(false);
      })
      .catch((e) => setNote(String(e.message || e)));
  }

  function openFile(rel: string, abs: string) {
    if (VOLUME_FILE_RE.test(rel)) {
      setSel(rel);
      setDoc(null);
      setShowVolume(true); // re-selecting the same volume must still show it (prop wouldn't change)
      // Compare toggle on → the click fills the second pane; otherwise it opens the primary.
      if (compareMode) onComparePathChange?.(abs);
      else onVolumePathChange(abs);
      return;
    }
    if (dirty && !window.confirm("Discard unsaved changes to this file?")) return;
    setSel(rel);
    setShowVolume(false);
    load(rel);
  }

  // The agent and every workflow write configs too, so the save carries the text the editor opened and
  // the server refuses it when the file no longer holds it. Reload is then the only honest way forward.
  async function save() {
    if (!doc) return;
    setNote("saving…");
    const r = await fetch("/api/config/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session, name: doc.name, content: draft, base: doc.content }),
    }).catch(() => null);
    if (r?.ok) {
      setDoc((d) => (d ? { ...d, content: draft } : d));
      setDirty(false);
      setNote("saved ✓");
      return;
    }
    const detail = r ? await r.json().then((e) => e.detail).catch(() => "") : "";
    setStale(r?.status === 409);
    setNote(detail || "save failed");
  }

  return (
    <div className="expview">
      {info && (
        <div className="exp-over">
          <span className="exp-chip" title={info.checkpoints.join("\n") || "no checkpoints"}>
            <b>{info.checkpoints.length}</b> checkpoint{info.checkpoints.length === 1 ? "" : "s"}
          </span>
          <span className="exp-chip" title={info.predictions.join("\n") || "no predictions"}>
            <b>{info.predictions.length}</b> prediction{info.predictions.length === 1 ? "" : "s"}
          </span>
          <span className="exp-chip" title={info.jobs.map((j) => `${j.kind} · ${j.run ?? "?"} · ${j.status}`).join("\n") || "no jobs"}>
            <b>{info.jobs.length}</b> job{info.jobs.length === 1 ? "" : "s"}
          </span>
          {info.jobs[0] && (
            <span className={`exp-chip st ${info.jobs[0].status}`}>
              latest: {info.jobs[0].kind} · {info.jobs[0].status}
            </span>
          )}
          {dataset && (
            <span className="exp-chip" title={dataset}>
              dataset: {dataset.split("/").pop()}
            </span>
          )}
        </div>
      )}
      <div className="cfg">
        <div className="tree" key={treeKey}>
          <div className="tree-mount-label">Workspace</div>
          <DirNode
            list={wsList}
            rel=""
            name=""
            depth={0}
            sel={sel}
            focusRel={focusDir || wsFocus}
            refreshKey={refreshKey}
            onFile={openFile}
          />
          {dataset && (
            <>
              <div className="tree-mount-label">Dataset · input (read-only)</div>
              <DirNode list={dsList} rel="" name="" depth={0} sel={sel} focusRel={dsFocus} refreshKey={0} onFile={openFile} />
            </>
          )}
        </div>
        <div className="cfg-content">
          {/* NiiVue stays mounted so it keeps its loaded volume across selections. */}
          <div className={showVolume ? "exp-viewer" : "exp-viewer hidden"}>
            <Viewer
              path={volumePath}
              onPathChange={onVolumePathChange}
              comparePath={comparePath}
              onComparePathChange={onComparePathChange}
              compareMode={compareMode}
              onCompareModeChange={(on) => {
                setCompareMode(on);
                if (!on) onComparePathChange?.(""); // leaving compare drops the second volume
              }}
            />
          </div>
          {!showVolume && doc && (
            <div className="cfg-edit">
              <textarea
                className="cfg-body edit"
                value={draft}
                readOnly={!doc.editable}
                spellCheck={false}
                onChange={(e) => {
                  setDraft(e.target.value);
                  setDirty(true);
                  setNote("");
                }}
              />
              <div className="cfg-bar">
                <span className="cfg-note">
                  {note || (dirty ? "unsaved changes" : doc.editable ? doc.name : `${doc.name} · read-only`)}
                </span>
                {stale && (
                  <button
                    className="cfg-save"
                    onClick={() => (!dirty || window.confirm("Discard your edits and read the file again?")) && load(sel)}
                  >
                    Reload
                  </button>
                )}
                {doc.editable && (
                  <button className="cfg-save" onClick={save} disabled={!dirty}>
                    Save
                  </button>
                )}
              </div>
            </div>
          )}
          {!showVolume && !doc && (
            <div className="cfg-empty">
              Browse the experiment folder. YAML and text open here to read/edit, volumes open inline in NiiVue.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Validation reads amber, everything else sage-green, so a metric's train/val lines are told apart at a
// glance (orange = validation, the one comparison that matters on a training chart).
// The workflow order runs happen in: drives the sub-tab order (train first, evaluation last).
const KIND_ORDER: Record<string, number> = { train: 0, finetune: 1, prediction: 2, uncertainty: 3, evaluation: 4, transform: 5 };

// Delete removes `<root>/<run>` at the session root. An app run lives in its own trial subtree, where
// that path names a DIFFERENT run that happens to share its name, so it is not offered there.
const SESSION_ROOTS = ["Statistics", "Predictions", "Evaluations", "Uncertainties", "Transforms"];
const deletable = (base: string) => SESSION_ROOTS.includes(base.split("/")[0]);

const SAGE = "var(--sage)";
const AMBER = "var(--working)";
const SAGE_HEX = "#52b89b"; // for the SVG gradient stops (an attribute won't resolve a CSS var)
function lineColor(label: string): string {
  return /valid/i.test(label) ? AMBER : SAGE;
}

// The metric name a series key belongs to (its chart identity): the last token of the part after the line
// label, so "Training/UNetpp5:MAE" and "Validation/UNetpp5:MAE" share the "MAE" chart.
function chartId(key: string, lineLabel: string): string {
  return shortMetric(key.slice(lineLabel.length + 1)) || key;
}

type ChartLine = { key: string; label: string; points: Point[] };

// One metric for one phase, TensorBoard-style: a continuous curve on the global-iteration axis (training
// sage, validation amber markers). Train and validation are separate cards, so each is clearly its own
// curve. Click → the full TB history (which overlays every tag matching the metric).
function MetricChart({
  id,
  lines,
  phase,
  stage,
  at,
  onExpand,
}: {
  id: string;
  lines: ChartLine[];
  phase?: string;
  stage?: string;
  at?: number;
  onExpand?: (metric: string) => void;
}) {
  const all = lines.flatMap((l) => l.points);
  if (all.length === 0) return null;
  const xs = all.map((p) => p.x);
  const ys = all.map((p) => p.y);
  const xmin = Math.min(...xs);
  const xr = Math.max(...xs) - xmin || 1;
  const ymin = Math.min(...ys);
  const ymax = Math.max(...ys);
  const yr = ymax - ymin || 1;
  const alone = Math.max(...xs) === xmin; // one case measured: centre it rather than pin it to the left edge
  const px = (p: Point) => (alone ? W / 2 : ((p.x - xmin) / xr) * W).toFixed(1);
  const py = (p: Point) => (H - ((p.y - ymin) / yr) * H).toFixed(1);
  const gid = "grad_" + id.replace(/[^a-z0-9]/gi, "");
  // Evaluation and prediction advance one point per case, not per training iteration: label the axis so.
  const perCase = !(stage === "training" || stage === "validation");
  const xUnit = perCase ? "case" : "iteration";
  return (
    <div
      className={onExpand ? "mcard clickable" : "mcard"}
      onClick={onExpand ? () => onExpand(id) : undefined}
      title={onExpand ? "Click to zoom the curve" : undefined}
    >
      <div className="mcard-head">
        <span className="mk-wrap">
          <span className="mk" title={id}>
            {id}
          </span>
          {phase && <span className={`stage ${/valid/i.test(phase) ? "validation" : "training"}`}>{phase}</span>}
        </span>
        <span className="legend">
          {lines.map((l) => {
            const last = l.points[l.points.length - 1];
            return (
              <span key={l.key} className="leg" style={{ color: lineColor(l.label) }} title={l.label}>
                {last ? last.y.toFixed(4) : ": "}
              </span>
            );
          })}
        </span>
      </div>
      <div className="chart">
        <svg className="mspark" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
          <defs>
            <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={SAGE_HEX} stopOpacity="0.24" />
              <stop offset="100%" stopColor={SAGE_HEX} stopOpacity="0" />
            </linearGradient>
          </defs>
          <rect className="plot" x="0" y="0" width={W} height={H} />
          {[0.25, 0.5, 0.75].map((f) => (
            <line key={f} className="grid" x1="0" x2={W} y1={H * f} y2={H * f} />
          ))}
          {lines.map((l) => {
            if (l.points.length === 0) return null;
            const pts = l.points.map((p) => `${px(p)},${py(p)}`).join(" ");
            const color = lineColor(l.label);
            // A case is not a step in a series. Which cases reach the log is decided by the progress
            // bar's own flushing (a four-case evaluation logs the first and the last), and cases have no
            // order anyway, so a line between them draws a trend across cases nobody measured. Per case:
            // markers only. The complete per-case values are in the evaluation table, from the metric file.
            const sparse = perCase || l.points.length <= 32; // dots when few; a dense curve stays a clean line
            const last = l.points[l.points.length - 1];
            return (
              <g key={l.key} style={{ color }}>
                {!perCase && color === SAGE && (
                  <polyline className="area" points={`${pts} ${W},${H} 0,${H}`} fill={`url(#${gid})`} />
                )}
                {!perCase && (
                  <polyline className="line" points={pts} vectorEffect="non-scaling-stroke" style={{ stroke: color }} />
                )}
                {sparse && l.points.map((p, i) => <circle key={i} className="mk-dot" cx={px(p)} cy={py(p)} r="2.4" style={{ fill: color }} />)}
                <circle className="dot" cx={px(last)} cy={py(last)} r="3.2" vectorEffect="non-scaling-stroke" style={{ fill: color }} />
              </g>
            );
          })}
        </svg>
        {/* One point, or a metric that never moved: a scale needs two ends to mean anything, and three
            copies of the same number is not a scale. Keep the value alone. */}
        {ymax !== ymin && <span className="yhi">{yTick(ymax)}</span>}
        <span className="ymid">{yTick((ymax + ymin) / 2)}</span>
        {ymax !== ymin && <span className="ylo">{yTick(ymin)}</span>}
      </div>
      <div className="mcard-foot">
        <span className="mstep">
          {xUnit} {Math.max(...xs)}
        </span>
        {at ? <span className="mtime">{fmtClock(at)}</span> : null}
      </div>
    </div>
  );
}

// The tail of a run's console, scrolled to its end. A traceback's last line is the one that names the
// cause, and anchored at the top the box showed the framework's inner frames and clipped it.
function ConsoleTail({ lines, failed }: { lines: string[]; failed: boolean }) {
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const node = box.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [lines]);
  return (
    <div className={failed ? "feed-log error" : "feed-log"} ref={box}>
      {lines.map((line, i) => (
        <div key={i} className="ln">
          {line}
        </div>
      ))}
    </div>
  );
}

// A run's current activity: the one thing TensorBoard has no notion of: which phase is running, its
// progress bar + it/s + ETA, and live host resources (GPU memory while training, a RAM trace while caching).
function LiveStrip({ live, active }: { live: LiveStatus; active: boolean }) {
  const p = live.progress;
  const res: string[] = [];
  if (live.memoryGpuGb != null) res.push(`GPU ${live.memoryGpuGb.toFixed(1)}G${live.memoryGpuPercent != null ? ` (${Math.round(live.memoryGpuPercent)}%)` : ""}`);
  if (live.memoryGb != null) res.push(`RAM ${live.memoryGb.toFixed(1)}G${live.memoryPercent != null ? ` (${Math.round(live.memoryPercent)}%)` : ""}`);
  if (live.cpuPercent != null) res.push(`CPU ${Math.round(live.cpuPercent)}%`);
  const metrics = Object.entries(live.metrics);
  return (
    <div className={active ? "live-strip on" : "live-strip"}>
      <div className="live-row">
        <span className="live-phase">{live.label}</span>
        <span className="live-time">{fmtClock(live.at)}</span>
        {p && (
          // Rate and ETA describe a run that is still moving. On a stopped one they are the last frame's
          // guess at a future that never happened; the counter still says where it got to.
          <span className="live-prog">
            {p.step}/{p.total}
            {active && ` · ${p.rate}${p.rate_unit} · ETA ${p.remaining}`}
          </span>
        )}
        {res.length > 0 && <span className="live-res">{res.join(" · ")}</span>}
      </div>
      {p && (
        <div className="feed-prog-track">
          <div className="feed-prog-fill" style={{ width: `${p.percent}%` }} />
        </div>
      )}
      {metrics.length > 0 && (
        <div className="live-metrics">
          {metrics.map(([k, v]) => (
            <span key={k} className="live-metric" title={k}>
              <b>{shortMetric(k)}</b> {v.toFixed(4)}
            </span>
          ))}
        </div>
      )}
      {live.stage === "caching" && live.ram.length > 1 && (
        <div className="ram-trace">
          <svg className="ram-spark" viewBox={`0 0 ${W} 40`} preserveAspectRatio="none">
            <polyline className="line" points={ramPoints(live.ram)} vectorEffect="non-scaling-stroke" />
          </svg>
          <span className="ram-cap">RAM (GB)</span>
        </div>
      )}
    </div>
  );
}

function ramPoints(points: Point[]): string {
  const c = curve(points, W, 40);
  return c.line;
}

// One run's feed: its live strip, then its unified metric charts (one per metric, whole run).
function RunSection({
  run,
  showLabel,
  active,
  onExpand,
}: {
  run: RunFeed;
  showLabel: boolean;
  active: boolean;
  onExpand: (runKey: string, seriesKey: string) => void;
}) {
  // One card per (metric × phase): train and validation are separate curves, never merged. Same-metric
  // cards sit next to each other (training first), learning rate last.
  const cards = Object.entries(run.series)
    .map(([key, s]) => ({ key, id: chartId(key, s.label), label: s.label, points: s.points, stage: s.stage, at: s.at }))
    .sort((a, b) => {
      const lrA = a.id === "lr" ? 1 : 0;
      const lrB = b.id === "lr" ? 1 : 0;
      if (lrA !== lrB) return lrA - lrB;
      if (a.id !== b.id) return a.id.localeCompare(b.id);
      return (/valid/i.test(a.label) ? 1 : 0) - (/valid/i.test(b.label) ? 1 : 0);
    });
  return (
    <section className="run-sec">
      {showLabel && (
        <div className="run-sec-head">
          <span className="run-name">{run.run}</span>
          <span className={`cst ${run.status}`}>{run.status}</span>
        </div>
      )}
      {run.live && <LiveStrip live={run.live} active={active} />}
      {cards.length > 0 ? (
        <div className="chart-grid">
          {cards.map((c) => (
            <MetricChart
              key={c.key}
              id={c.id}
              phase={c.label}
              stage={c.stage}
              at={c.at}
              lines={[{ key: c.key, label: c.label, points: c.points }]}
              onExpand={() => onExpand(run.key, c.key)}
            />
          ))}
        </div>
      ) : (
        // A run that is over and has no curves has nothing coming: its outputs were deleted, or it died
        // before its first metric. Saying "waiting" there describes an arrival that will never happen.
        !run.live && (
          <div className="feed-sub">{active ? "waiting for the first metric…" : "no metrics for this run"}</div>
        )
      )}
    </section>
  );
}

// A zoomable/pannable image viewer for the sample montages: wheel to zoom, drag to pan, double-click to
// reset. Montages are wide (5 slices side by side), so real inspection needs zoom, not just fit-to-screen.
function ImageLightbox({ src, onClose }: { src: string; onClose: () => void }) {
  useEscapeToClose(onClose);
  const [scale, setScale] = useState(1);
  const [pos, setPos] = useState({ x: 0, y: 0 });
  const drag = useRef<{ x: number; y: number } | null>(null);
  const zoom = (factor: number) => setScale((s) => Math.min(10, Math.max(1, s * factor)));
  const reset = () => {
    setScale(1);
    setPos({ x: 0, y: 0 });
  };
  return (
    <div className="lightbox zoomable" onClick={onClose}>
      <div className="lb-bar" onClick={(e) => e.stopPropagation()}>
        <button onClick={() => zoom(1 / 1.3)} title="Zoom out">
          −
        </button>
        <span className="lb-pct">{Math.round(scale * 100)}%</span>
        <button onClick={() => zoom(1.3)} title="Zoom in">
          +
        </button>
        <button onClick={reset} title="Reset">
          Reset
        </button>
        <button onClick={onClose} title="Close">
          ✕
        </button>
      </div>
      <div
        className="lb-stage"
        onClick={(e) => e.stopPropagation()}
        onDoubleClick={reset}
        onWheel={(e) => zoom(e.deltaY < 0 ? 1.15 : 1 / 1.15)}
        onMouseDown={(e) => (drag.current = { x: e.clientX - pos.x, y: e.clientY - pos.y })}
        onMouseMove={(e) => drag.current && setPos({ x: e.clientX - drag.current.x, y: e.clientY - drag.current.y })}
        onMouseUp={() => (drag.current = null)}
        onMouseLeave={() => (drag.current = null)}
        style={{ cursor: scale > 1 ? "grab" : "zoom-in" }}
      >
        <img
          src={src}
          alt="sample"
          draggable={false}
          style={{ transform: `translate(${pos.x}px, ${pos.y}px) scale(${scale})` }}
        />
      </div>
    </div>
  );
}


type LbRow = { run_name: string; value: number; direction?: string };

// Rank the experiment's runs by their evaluation metrics, one ranked table per metric, via konfai-mcp's
// leaderboard (which reads the Metric_<SPLIT>.json files live). Appears once there are ≥2 runs.
// Pick any two ranked runs and see exactly what differs in their launch configs: model, losses, optimizer,
// augmentations, and any live interventions recorded during the run.
function LbDiff({ session, runs }: { session: string; runs: string[] }) {
  const [a, setA] = useState(runs[0] ?? "");
  const [b, setB] = useState(runs[1] ?? runs[0] ?? "");
  const [diff, setDiff] = useState<{ identical: boolean; text: string } | null>(null);
  const [loading, setLoading] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  async function run() {
    if (!a || !b || a === b) return;
    setLoading(true);
    setDiff(null);
    try {
      const d = await getJson(
        `/api/run/config_diff?session=${encodeURIComponent(session)}&run_a=${encodeURIComponent(a)}&run_b=${encodeURIComponent(b)}`,
      );
      setDiff(d.ok ? { identical: d.identical, text: d.diff } : { identical: false, text: d.detail || "diff failed" });
    } catch {
      setDiff({ identical: false, text: "diff failed" });
    } finally {
      setLoading(false);
      // The diff renders at the bottom of the leaderboard, usually below the fold: bring it in.
      requestAnimationFrame(() => box.current?.scrollIntoView({ block: "nearest", behavior: "smooth" }));
    }
  }
  return (
    <div className="lb-diff" ref={box}>
      <div className="lb-diff-head">
        <span className="feed-label">Config diff</span>
        <select className="lb-sel" value={a} onChange={(e) => setA(e.target.value)}>
          {runs.map((r) => (
            <option key={r} value={r}>
              {r}
            </option>
          ))}
        </select>
        <span className="lb-vs">vs</span>
        <select className="lb-sel" value={b} onChange={(e) => setB(e.target.value)}>
          {runs.map((r) => (
            <option key={r} value={r}>
              {r}
            </option>
          ))}
        </select>
        <button className="v-btn" onClick={run} disabled={!a || !b || a === b || loading}>
          {loading ? "…" : "Diff"}
        </button>
      </div>
      {diff &&
        (diff.identical ? (
          <div className="empty">These two runs have identical configs.</div>
        ) : (
          <pre className="lb-diff-body">
            {diff.text.split("\n").map((ln, i) => (
              <div
                key={i}
                className={
                  ln.startsWith("+") ? "d-add" : ln.startsWith("-") ? "d-del" : ln.startsWith("@@") ? "d-hunk" : "d-ctx"
                }
              >
                {ln || " "}
              </div>
            ))}
          </pre>
        ))}
    </div>
  );
}

function Leaderboard({ session }: { session: string }) {
  const [splits, setSplits] = useState<string[]>(["TRAIN"]);
  const [split, setSplit] = useState("TRAIN");

  const { data, loading } = useJson<{ leaderboards?: Record<string, LbRow[]>; available_splits?: string[] }>(
    `/api/leaderboard?session=${encodeURIComponent(session)}&split=${encodeURIComponent(split)}`,
    [session, split],
  );
  const boards = data?.leaderboards ?? {};
  const loaded = !loading;
  useEffect(() => {
    if (data && Array.isArray(data.available_splits) && data.available_splits.length) setSplits(data.available_splits);
  }, [data]);

  const metrics = Object.keys(boards).sort();
  const runNames = [...new Set(metrics.flatMap((m) => boards[m].map((r) => r.run_name)))];
  return (
    <div className="lb">
      <div className="lb-head">
        <span className="feed-label">Leaderboard · runs ranked by evaluation</span>
        {splits.length > 1 && (
          <span className="lb-splits">
            {splits.map((s) => (
              <button key={s} className={s === split ? "lb-split on" : "lb-split"} onClick={() => setSplit(s)}>
                {s}
              </button>
            ))}
          </span>
        )}
      </div>
      {!loaded && metrics.length === 0 ? (
        // The board's own shape, greyed: a skeleton promises what is coming where "Loading…" promises nothing.
        <div className="lb-grid" aria-hidden="true">
          {[0, 1].map((k) => (
            <div key={k} className="lb-card sk-card">
              <div className="sk-line sk-w40" />
              <div className="sk-line sk-w90" />
              <div className="sk-line sk-w80" />
            </div>
          ))}
        </div>
      ) : metrics.length === 0 ? (
        <div className="empty">{`No ${split} evaluations yet: evaluate two or more runs to rank them here.`}</div>
      ) : (
        <div className="lb-grid">
          {metrics.map((m) => (
            <div key={m} className="lb-card">
              <div className="lb-metric">
                <span className="mk">{shortMetric(m)}</span>
                <span className="dir">{boards[m][0]?.direction === "max" ? "↑ higher is better" : "↓ lower is better"}</span>
              </div>
              <table className="evtable">
                <tbody>
                  {boards[m].map((row, i) => (
                    <tr key={row.run_name} className={i === 0 ? "lb-best" : undefined}>
                      <td className="lb-rank">{i + 1}</td>
                      <td className="lb-run" title={row.run_name}>
                        {row.run_name}
                      </td>
                      <td className="num strong">{fmt(row.value)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
        </div>
      )}
      {runNames.length >= 2 && <LbDiff session={session} runs={runNames} />}
    </div>
  );
}

// Steer a running training without restarting it: set a new learning rate and/or validation interval, which
// the trainer applies at its next poll boundary and logs as an intervention in the run's config.
function TuneControls({ session }: { session: string }) {
  const [lr, setLr] = useState("");
  const [iv, setIv] = useState("");
  const [note, setNote] = useState("");
  async function apply() {
    const body: { session: string; lr?: number; it_validation?: number } = { session };
    const lrNum = Number(lr);
    const ivNum = Number(iv);
    if (lr.trim() && Number.isFinite(lrNum)) body.lr = lrNum;
    if (iv.trim() && Number.isFinite(ivNum)) body.it_validation = ivNum;
    if (body.lr === undefined && body.it_validation === undefined) return;
    setNote("…");
    try {
      const d = await postJson("/api/job/tunables", body);
      setNote(d.ok ? "applied ✓" : "failed");
      if (d.ok) {
        setLr("");
        setIv("");
      }
    } catch {
      setNote("failed");
    }
  }
  return (
    <span className="tune-controls">
      <input className="tune-in" value={lr} onChange={(e) => setLr(e.target.value)} placeholder="lr" title="New learning rate" />
      <input
        className="tune-in"
        value={iv}
        onChange={(e) => setIv(e.target.value)}
        placeholder="it_val"
        title="New validation interval (iterations)"
      />
      <button className="v-btn" onClick={apply} title="Apply to the running run at its next poll boundary">
        Tune
      </button>
      {note && <span className="tune-note">{note}</span>}
    </span>
  );
}

// The large right pane. EXPERIMENT = the workspace folder tree with an adaptive content pane
// (text/YAML editor or inline NiiVue for volumes). LIVE = a scrolling feed of what a run produces
// (metrics by stage, sample images, evaluation scores).
export default function RightPanel({
  session,
  volumePath,
  volumeNonce,
  onVolumePathChange,
  comparePath,
  onComparePathChange,
  stream,
  dataset,
  onReload,
}: {
  session: string;
  volumePath: string | null;
  volumeNonce?: number;
  onVolumePathChange: (p: string) => void;
  comparePath: string | null;
  onComparePathChange: (p: string) => void;
  stream: JobStream;
  dataset?: string; // the experiment's dataset: picking one has to reach the overview and the tree

  onReload?: () => void; // reconnect the live stream (e.g. after deleting a run) so state rebuilds from disk
}) {
  const [tab, setTab] = useState<string>("config"); // "config" (Workspace) or a run name
  // Run tabs closed by their ✕: a view choice, nothing on disk: they come back via the "+n closed"
  // chip, on reload, or by themselves when their run starts working again.
  const [hiddenTabs, setHiddenTabs] = useState<string[]>([]);
  const [subKey, setSubKey] = useState(""); // the selected kind (a run key) within the open run name
  const [browsePath, setBrowsePath] = useState(""); // a workspace dir to reveal in the tree (from a run tab)
  const [browseOpen, setBrowseOpen] = useState(""); // what to auto-open there: "config" | "volume" | "first"
  const [auto, setAuto] = useState(true);
  const [light, setLight] = useState<string | null>(null);
  const [hasEval, setHasEval] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [expand, setExpand] = useState<{ runKey: string; seriesKey: string } | null>(null); // curve-zoom modal
  const [validating, setValidating] = useState(false);
  const [note, setNote] = useState(""); // a refused action says why, instead of looking like a no-op

  useEffect(() => {
    setHasEval(false);
    setStopping(false);
    setValidating(false);
    setNote(""); // a refusal belongs to the session it happened in
    setTab("config");
    // Closed tabs are a view choice on THIS experiment's runs: carried across, a same-named run in the
    // next experiment would arrive silently hidden, under a phantom "+n closed" chip.
    setHiddenTabs([]);
  }, [session]);

  // Two levels: the top tab is a run NAME (MR2CT_01…), and inside it a sub-tab per KIND (train / prediction
  // / evaluation). `sel` is the (name, kind) run whose feed is shown. The active run (latest job) drives
  // the Stop / Validate controls.
  const active = stream.runs.find((r) => r.key === stream.activeRun);
  const runNames: string[] = [];
  for (const r of stream.runs) if (!runNames.includes(r.run)) runNames.push(r.run);
  // Sub-tabs follow the workflow order (train, then prediction, then evaluation) the order you actually
  // ran them, not newest-first (and robust while a training is still writing its log).
  const kindsForName = stream.runs
    .filter((r) => r.run === tab)
    .sort((a, b) => (KIND_ORDER[a.kind] ?? 9) - (KIND_ORDER[b.kind] ?? 9)); // one per kind under the open name
  const sel = kindsForName.find((r) => r.key === subKey) ?? kindsForName[0];
  const isActiveSel = !!sel && sel.key === stream.activeRun;
  const running = isRunning(active?.status ?? stream.status);
  // The session's job itself: true while it starts up and while an isolated app run has not been
  // discovered yet, so Stop never depends on a run being found (or selected).
  const jobRunning = isRunning(stream.status);
  // A job LAUNCH re-arms the auto-follow: the user just asked for this run (through the agent or
  // a button), so bringing its tab forward is delivering, not stealing. A manual tab pick holds
  // again once this job is over.
  const wasRunning = useRef(false);
  useEffect(() => {
    if (jobRunning && !wasRunning.current) setAuto(true);
    wasRunning.current = jobRunning;
  }, [jobRunning]);
  const stage = active?.live?.stage ?? "";
  // An app job carries two names: its own (`app_MR`) and the one its run takes from the log directory
  // (`ImpactSynth`). While the run is unknown, a synthetic tab under the job's name shows the console;
  // once the run is discovered AND live, it is the one to watch (progress bar, curves), and the
  // synthetic tab has nothing more to say.
  const handover = jobRunning && active && active.run !== stream.run && isRunning(active.status) ? active : null;

  // Once nothing is live any more, drop the transient "Stopping…"/"Validating…" labels: the Stop control
  // gives way to Delete, and a one-way flag must never outlive the run it described.
  useEffect(() => {
    if (!jobRunning) setStopping(false);
    if (!running) setValidating(false);
  }, [running, jobRunning]);

  async function stopJob() {
    setStopping(true);
    // A refusal ("no running job", a reaper that could not signal) is an answer, not a no-op: without
    // it the button sits on "Stopping…" over a run that never stops, explaining nothing.
    try {
      const d = await postJson("/api/job/cancel", { session });
      if (d && d.ok === false) {
        setStopping(false);
        setNote(d.detail || "the job could not be stopped");
      }
    } catch {
      setStopping(false);
      setNote("the job could not be stopped");
    }
  }

  async function deleteRun(runName: string, kind: string, what: string) {
    if (!window.confirm(`Delete ${what} of ${runName}? This removes its output folders on disk.`)) return;
    try {
      const d = await postJson("/api/run/delete", { session, run_name: runName, kind });
      if (d && d.ok === false) {
        setNote(d.detail || "the run could not be deleted");
        return;
      }
    } catch {
      setNote("the run could not be deleted");
      return;
    }
    setTab("config");
    onReload?.(); // reconnect so the run list rebuilds from disk (the deleted run drops out)
  }

  async function openTensorboard() {
    // Open synchronously (in the click) so it isn't popup-blocked, but WITHOUT `noopener`: that flag makes
    // window.open return null, so we could never navigate the tab to the resolved URL.
    const tab = window.open("about:blank", "_blank");
    try {
      const d = await getJson(`/api/tensorboard?session=${encodeURIComponent(session)}`);
      if (!tab) return;
      if (d.url) tab.location.href = d.url;
      else {
        tab.document.title = "TensorBoard";
        tab.document.body.style.cssText = "font:14px system-ui;padding:24px;color:#333";
        tab.document.body.textContent = d.detail || "TensorBoard is unavailable for this experiment.";
      }
    } catch {
      tab?.close();
    }
  }

  async function validateJob() {
    setValidating(true);
    try {
      await postJson("/api/job/validate", { session });
    } catch {
      /* the validation metrics stream into the feed when the pass runs */
    } finally {
      window.setTimeout(() => setValidating(false), 2500); // it runs at the next iteration boundary
    }
  }

  function show(t: string) {
    setAuto(false);
    setTab(t);
  }

  // Jump to the run's output folder in the Workspace tree and open its most useful artifact: a training →
  // its config, a prediction → the first predicted image, an evaluation → its Metric_TRAIN.json. `data`
  // names the folder when it is not the run's own (a transform chain's destination); it defaults to the
  // run's first one.
  // The workspace tree only serves paths inside the session: a chain that wrote to an absolute
  // destination of its own is named rather than offered, because opening it would answer nothing.
  function inWorkspace(path?: string): boolean {
    if (!path) return true;
    // Absolute in either spelling: POSIX, a home, a Windows drive, or a UNC share.
    if (/^([/~\\]|[A-Za-z]:)/.test(path)) return false;
    return !path.split(/[\\/]/).includes("..");
  }

  function browseRun(r: RunFeed, data: string = r.data) {
    // Prefer the run's real base (isolated app runs live under "<app_output>-<hash>/…"); fall back to the
    // session-root layout for an older stream that predates the base field.
    const root =
      r.base ||
      (r.kind === "prediction"
        ? `Predictions/${r.run}`
        : r.kind === "evaluation"
          ? `Evaluations/${r.run}`
          : r.kind === "uncertainty"
            ? `Uncertainties/${r.run}`
            : r.kind === "transform"
              ? `Transforms/${r.run}`
              : `Statistics/${r.run}`);
    // A transform's run dir holds a log, a plan and a config copy; its volumes land wherever each
    // Write: pointed, which the run records in outputs.json and the feed carries as `data`. Opening
    // the run dir would answer "what did it produce" with a YAML file.
    const [dir, open] = data
      ? [data, "volume"]
      : r.kind === "prediction"
        ? [`${root}/Dataset`, "volume"]
        : r.kind === "evaluation"
          ? [root, "metric"]
          : r.kind === "uncertainty"
            ? [root, "volume"]
            : [root, "config"];
    setBrowsePath(dir);
    setBrowseOpen(open);
    setAuto(false);
    setTab("config");
  }

  // A run starting (or its first metric) brings its name tab + kind forward, so the warm-up (caching) and
  // then the curves are visible without a manual switch.
  useEffect(() => {
    if (active && auto) {
      setTab(active.run);
      setSubKey(active.key);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stream.activeRun, stream.metricNonce]);

  // The assistant pointing at a volume brings the Workspace pane forward. Keyed on the nonce, not the
  // path: re-showing the same volume with a different companion ("the CT beside its segmentation") left
  // the pane on the run tab, so the reply said the two were open side by side over a progress bar.
  useEffect(() => {
    if (volumePath && auto) setTab("config");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [volumePath, volumeNonce]);

  // A job that started but owns no run tab yet (an app job, whose run is named only once it writes):
  // open the feed regardless, so its console is watchable while it warms up.
  useEffect(() => {
    if (jobRunning && stream.run && !runNames.includes(stream.run)) setTab(stream.run);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobRunning, stream.run]);

  // The moment the job's real run comes alive, the view sitting on the synthetic tab follows it: the
  // user was already watching this job, so handing over to its run (progress bar included) is not
  // stealing focus, and staying would leave them on raw console lines for the whole prediction.
  useEffect(() => {
    if (handover && tab === stream.run) {
      setTab(handover.run);
      setSubKey(handover.key);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [handover?.key, tab]);

  // A closed tab whose run starts working again reopens itself: hiding is about clutter, and a
  // running job must stay reachable from the bar.
  useEffect(() => {
    if (jobRunning && stream.run) setHiddenTabs((h) => (h.includes(stream.run) ? h.filter((n) => n !== stream.run) : h));
  }, [jobRunning, stream.run]);

  return (
    <section className="rpanel">
      <div className="rtabs">
        <button className={tab === "config" ? "rtab on" : "rtab"} onClick={() => show("config")}>
          Workspace
        </button>
        {runNames.length >= 2 && (
          <button className={tab === "leaderboard" ? "rtab on" : "rtab"} onClick={() => show("leaderboard")}>
            Leaderboard
          </button>
        )}
        {/* A job with no run of its own: an app fetching its bundle, or one that died during warm-up.
            It gets a tab of its own so it is reachable at all; it goes as soon as its real run comes
            alive under another name (the handover above follows it) or the job ends well. A job that
            DIED here keeps its tab: its console holds the only explanation of what went wrong, and
            dropping the tab on settle left the user in front of a bar with nothing to click. */}
        {stream.run &&
          !runNames.includes(stream.run) &&
          !handover &&
          (jobRunning || jobState(stream.status) === "err") && (
            <button
              className={tab === stream.run ? "rtab on" : "rtab"}
              onClick={() => show(stream.run)}
              title={`${stream.kind || "job"} · ${jobRunning ? "starting" : stream.status}`}
            >
              {stream.status && <StatusMark status={stream.status} />}
              {stream.run}
            </button>
          )}
        {runNames
          .filter((name) => !hiddenTabs.includes(name))
          .map((name) => {
            const newest = stream.runs.find((r) => r.run === name); // runs are newest first
            return (
              <button
                key={name}
                className={tab === name ? "rtab on" : "rtab"}
                onClick={() => show(name)}
                onContextMenu={(e) => {
                  e.preventDefault();
                  if (newest && deletable(newest.base)) deleteRun(name, "all", "the whole run (all outputs)");
                }}
                title={`${name}: ✕ closes the tab only; right-click deletes the whole run`}
              >
                {newest && <StatusMark status={newest.status} />}
                {name}
                <span
                  className="rtab-close"
                  title="Close this tab (nothing is deleted)"
                  onClick={(e) => {
                    e.stopPropagation();
                    setHiddenTabs((h) => [...h, name]);
                    if (tab === name) show("config");
                  }}
                >
                  ×
                </span>
              </button>
            );
          })}
        {hiddenTabs.length > 0 && (
          <button className="rtab" onClick={() => setHiddenTabs([])} title="Reopen the closed run tabs">
            +{hiddenTabs.length} closed
          </button>
        )}
        {/* The job-level Stop: bound to the session's job, not to a discovered run, so it is there while a
            job is still starting, and on every tab, run selected or not. */}
        {jobRunning && (
          <button
            className="stop-job"
            onClick={stopJob}
            disabled={stopping}
            title={`Stop the running job${stream.run ? ` (${stream.kind || "job"} · ${stream.run})` : ""}`}
          >
            {stopping ? "Stopping…" : "Stop job"}
          </button>
        )}
        {note && (
          <span className="rtab-note" onClick={() => setNote("")} title="Dismiss">
            {note}
          </span>
        )}
      </div>

      {/* Both panes stay mounted (hidden when inactive) so NiiVue keeps its volume and the workspace tree
          keeps its expansion/selection across tab switches. */}
      <div className={tab === "leaderboard" ? "rscroll feed" : "rscroll feed hidden"}>
        {tab === "leaderboard" && <Leaderboard session={session} />}
      </div>

      <div className={tab !== "config" && tab !== "leaderboard" ? "rscroll feed" : "rscroll feed hidden"}>
        {sel ? (
          <>
            <div className="feed-job-head">
              <div className="run-tabs">
                {kindsForName.map((r) => (
                  <button
                    key={r.key}
                    className={sel.key === r.key ? "run-tab on" : "run-tab"}
                    onClick={() => {
                      setAuto(false);
                      setSubKey(r.key);
                    }}
                    onContextMenu={(e) => {
                      e.preventDefault();
                      if (deletable(r.base)) deleteRun(r.run, r.kind, `the ${r.kind} outputs`);
                    }}
                    title={`${r.kind} · ${r.status}: right-click to delete`}
                  >
                    <StatusMark status={r.status} />
                    {r.kind}
                  </button>
                ))}
              </div>
              <span className="feed-job-actions">
                {sel.outputs.length > 1 ? (
                  // A transform with several chains wrote to several places: one Browse per destination,
                  // named by the group it wrote. A single chain keeps the one button every kind has.
                  sel.outputs.map((o) =>
                    inWorkspace(o.path) ? (
                      <button
                        key={`${o.group_dest}:${o.path}`}
                        className="tb-link"
                        onClick={() => browseRun(sel, o.path)}
                        title={`Open ${o.group_src || "?"} → ${o.group_dest} (${o.format || "?"}) in the Workspace tree`}
                      >
                        Browse {o.group_dest} ↦
                      </button>
                    ) : (
                      <span
                        key={`${o.group_dest}:${o.path}`}
                        className="tb-link tb-link-off"
                        title={`${o.path}: outside the session workspace, so the tree cannot open it`}
                      >
                        {o.group_dest} → {o.path}
                      </span>
                    ),
                  )
                ) : inWorkspace(sel.outputs.length === 1 ? sel.outputs[0].path : sel.data) ? (
                  <button className="tb-link" onClick={() => browseRun(sel)} title={`Open ${sel.kind} outputs in the Workspace tree`}>
                    Browse files ↦
                  </button>
                ) : (
                  <span
                    className="tb-link tb-link-off"
                    title={`${sel.outputs.length === 1 ? sel.outputs[0].path : sel.data}: outside the session workspace, so the tree cannot open it`}
                  >
                    Wrote {sel.outputs.length === 1 ? sel.outputs[0].path : sel.data} ↦
                  </span>
                )}
                <button className="tb-link" onClick={openTensorboard} title="Open the full TensorBoard for this experiment">
                  TensorBoard ↗
                </button>
                {isActiveSel && running && sel.kind === "train" && (
                  <>
                    <button
                      className="validate-job"
                      onClick={validateJob}
                      disabled={validating || stage === "validation"}
                      title="Run a validation pass now"
                    >
                      {stage === "validation" ? "Validating…" : validating ? "Requested…" : "Validate now"}
                    </button>
                    <TuneControls session={session} />
                  </>
                )}
                {/* Stopping is session-scoped (it cancels the job), so it lives once in the panel header. */}
                {!running && deletable(sel.base) && (
                  <button
                    className="delete-run"
                    onClick={() => deleteRun(sel.run, sel.kind, `the ${sel.kind} outputs`)}
                    title={`Delete this ${sel.kind}'s outputs (the run disappears once its last kind is gone)`}
                  >
                    Delete {sel.kind}
                  </button>
                )}
              </span>
            </div>
            <FeedBoundary resetKey={sel.live?.at ?? stream.doneNonce}>
              <RunSection
                key={sel.key}
                run={sel}
                showLabel={false}
                active={running && isActiveSel}
                onExpand={(runKey, seriesKey) => setExpand({ runKey, seriesKey })}
              />
              {sel.status === "error" && isActiveSel && stream.lines.length > 0 && (
                <ConsoleTail lines={stream.lines.slice(-16)} failed />
              )}
              {(sel.kind === "train" || sel.kind === "finetune") && (
                <Samples session={session} base={sel.base} refresh={stream.metricNonce + stream.doneNonce} onOpen={setLight} />
              )}
              {(sel.kind === "evaluation" || sel.kind === "uncertainty") && (
                <section className="feed-sec">
                  {hasEval && <div className="feed-label">Evaluation scores</div>}
                  <EvaluationView
                    session={session}
                    refresh={stream.doneNonce}
                    hideWhenEmpty
                    onHasData={setHasEval}
                    onOpenCase={(run, caseName) => {
                      // The prediction sits beside the evaluation under the same (possibly isolated) root.
                      const prefix = sel.base.replace(/Evaluations\/[^/]+$/, ""); // "" at session root, "<app>/…/" when isolated
                      setBrowsePath(`${prefix}Predictions/${run}/Dataset/${caseName}`);
                      setBrowseOpen("volume");
                      setAuto(false);
                      setTab("config");
                    }}
                  />
                </section>
              )}
            </FeedBoundary>
          </>
        ) : jobRunning || stream.lines.length > 0 ? (
          // A job with no run of its own yet: fetching its bundle, warming up, or dead before its first
          // iteration. Its console is the only thing it has produced, so that is what is shown, rather
          // than an invitation to launch the job that is already running.
          <>
            <div className="feed-job-head">
              <span className="feed-label">
                {stream.kind || "job"}
                {stream.run ? ` · ${stream.run}` : ""}
              </span>
              <span className="feed-note">{jobRunning ? "starting: no output yet" : stream.status || "finished"}</span>
            </div>
            <div className="feed-log">
              {stream.lines.slice(-60).map((line, i) => (
                <div key={i} className="ln">
                  {line}
                </div>
              ))}
            </div>
          </>
        ) : (
          <div className="feed-hint">Launch a job: its live curves, sample images and evaluation scores stream here.</div>
        )}
      </div>

      <div className={tab === "config" ? "rcfg" : "rcfg hidden"}>
        <ExperimentView
          session={session}
          volumePath={volumePath}
          onVolumePathChange={onVolumePathChange}
          comparePath={comparePath}
          onComparePathChange={onComparePathChange}
          refresh={stream.doneNonce}
          dataset={dataset}
          live={jobState(stream.status) === "run"}
          visible={tab === "config"}
          focusDir={browsePath}
          focusOpen={browseOpen}
        />
      </div>

      {expand &&
        (() => {
          const r = stream.runs.find((x) => x.key === expand.runKey);
          const s = r?.series[expand.seriesKey];
          return r && s ? (
            <CurveZoom id={chartId(expand.seriesKey, s.label)} series={s} runName={r.run} onClose={() => setExpand(null)} />
          ) : null;
        })()}

      {light && <ImageLightbox src={light} onClose={() => setLight(null)} />}
    </section>
  );
}
