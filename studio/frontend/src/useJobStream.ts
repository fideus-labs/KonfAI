import { useEffect, useRef, useState } from "react";

export type Point = { x: number; y: number };

// tqdm progress, parsed once by konfai-mcp (konfai_mcp.live_parse) and relayed structured — the front
// never re-parses a log line. Present during data-caching and training/validation/prediction/evaluation.
export type Progress = {
  percent: number;
  step: number;
  total: number;
  elapsed: string;
  remaining: string;
  rate: number;
  rate_unit: string;
};

// One metric's continuous curve across the whole run, on the global-iteration axis — the TensorBoard
// shape. `label` is the line's identity within its chart (Training vs Validation vs Metric TRAIN…). `at`
// is the wall-clock of the last point (when it was produced, for the feed timestamps).
export type Series = { label: string; stage: string; points: Point[]; at: number };

// The live "now" of a run — the one thing TensorBoard lacks: the current phase, its progress bar, and the
// host resources while it runs (RAM/CPU during caching, GPU memory + it/s during training).
export type LiveStatus = {
  label: string;
  stage: string;
  progress: Progress | null;
  metrics: Record<string, number>; // the current line's live metric values (losses/metrics, minus lr)
  memoryGb: number | null;
  memoryPercent: number | null;
  memoryGpuGb: number | null;
  memoryGpuPercent: number | null;
  cpuPercent: number | null;
  ram: Point[]; // RAM trace during the current caching phase
  at: number;
};

// One run of the experiment (a runtime-log directory: a training MR2CT_01, a prediction, an evaluation…).
// Runs of every kind accumulate and persist — launching a prediction never clears the training runs.
// Metrics accumulate into one continuous series per (label, metric) so a curve spans the whole run.
export type RunFeed = {
  key: string; // unique across kinds: `${run} ${kind}` (a train and a prediction can share a run name)
  run: string; // display name (train_name)
  kind: string;
  base: string; // run dir relative to the session root — "Statistics/<run>" or "<app_output>-<hash>/…/<run>"
  status: string;
  startedAt: number;
  series: Record<string, Series>; // keyed `${label}/${metric}`
  live: LiveStatus | null;
};

// One live subscription to the active task. The whole runtime log is replayed on connect, so the curves
// rebuild in full — the feed survives a reload or a server restart.
export type JobStream = {
  lines: string[]; // console output (errors only)
  runs: RunFeed[]; // every run of the experiment, newest first — one tab each
  activeRun: string; // key of the latest job's run (the tab to default to)
  run: string; // latest job's run name — for the header badge
  status: string;
  kind: string;
  metricNonce: number; // bumps the first time a metric arrives
  doneNonce: number; // bumps when a run reaches a terminal status
};

const POINT_CAP = 1200; // per series; the full history lives in TensorBoard (CurveModal), this is the live tail
const RUN_CAP = 8;

// Append a point, replacing the last one when it shares the x (a validation pass logs several lines at the
// same training iteration — keep the converged value, one marker per pass).
function appendPoint(points: Point[], x: number, y: number): Point[] {
  const last = points[points.length - 1];
  if (last && last.x === x) return [...points.slice(0, -1), { x, y }];
  return [...points.slice(-(POINT_CAP - 1)), { x, y }];
}

export function useJobStream(session: string, runNonce: number): JobStream {
  const [lines, setLines] = useState<string[]>([]);
  const [runs, setRuns] = useState<RunFeed[]>([]);
  const [activeRun, setActiveRun] = useState("");
  const [run, setRun] = useState("");
  const [status, setStatus] = useState("");
  const [kind, setKind] = useState("");
  const [metricNonce, setMetricNonce] = useState(0);
  const [doneNonce, setDoneNonce] = useState(0);
  const sawMetric = useRef(false);
  const activeKeyRef = useRef(""); // key of the latest job's run — keeps the header status fresh when it ends
  const itCount = useRef<Record<string, number>>({}); // per run: monotonic training-iteration counter (the curve x-axis)
  const ramTick = useRef<Record<string, number>>({}); // per run: monotonic x for the caching RAM trace

  useEffect(() => {
    setLines([]);
    setRuns([]);
    setActiveRun("");
    setRun("");
    setStatus("");
    setKind("");
    sawMetric.current = false;
    itCount.current = {};
    ramTick.current = {};
    const ctrl = new AbortController();
    let stopped = false;

    const rkey = (r: string, k: string) => `${r} ${k}`; // unique run identity across kinds

    // Find-or-create a run bucket by its (run, kind) key, apply `mutate` to a shallow copy. Runs persist —
    // buckets are only cleared when the subscription resets (a different experiment).
    const withRun = (r: string, k: string, mutate: (feed: RunFeed) => RunFeed) =>
      setRuns((prev) => {
        const key = rkey(r, k);
        const next = prev.slice();
        let i = next.findIndex((feed) => feed.key === key);
        if (i < 0) {
          next.push({ key, run: r, kind: k, base: "", status: "running", startedAt: Date.now(), series: {}, live: null });
          if (next.length > RUN_CAP) next.shift();
          i = next.findIndex((feed) => feed.key === key);
        }
        next[i] = mutate({ ...next[i] });
        return next;
      });

    type Host = {
      memory_gb?: number;
      memory_percent?: number;
      memory_gpu_gb?: number;
      memory_gpu_percent?: number;
      cpu_percent?: number;
    };
    const liveFrom = (
      prev: LiveStatus | null,
      label: string,
      stage: string,
      p: Progress | null,
      h: Host,
      metrics?: Record<string, number>,
    ): LiveStatus => {
      const changed = !prev || prev.label !== label;
      return {
        label,
        stage,
        progress: p,
        // keep the last metric values across a phase change (until the new phase logs its own) so the live
        // readout never flashes empty when training pauses for a validation pass
        metrics: metrics ?? prev?.metrics ?? {},
        memoryGb: h.memory_gb ?? prev?.memoryGb ?? null,
        memoryPercent: h.memory_percent ?? prev?.memoryPercent ?? null,
        memoryGpuGb: h.memory_gpu_gb ?? (changed ? null : prev?.memoryGpuGb ?? null),
        memoryGpuPercent: h.memory_gpu_percent ?? (changed ? null : prev?.memoryGpuPercent ?? null),
        cpuPercent: h.cpu_percent ?? prev?.cpuPercent ?? null,
        ram: changed ? [] : prev?.ram ?? [],
        at: Date.now(),
      };
    };

    async function pump() {
      const resp = await fetch(`/api/live?session=${encodeURIComponent(session)}`, { signal: ctrl.signal });
      const reader = resp.body!.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let idx: number;
        while ((idx = buf.indexOf("\n\n")) >= 0) {
          const chunk = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          if (!chunk.startsWith("data: ")) continue;
          const ev = JSON.parse(chunk.slice(6));

          if (ev.type === "job") {
            // The latest job — sets the header + the default tab. It never clears runs: a new job (a
            // prediction after a training) adds a run, it doesn't wipe the ones already followed.
            activeKeyRef.current = rkey(ev.run || "", ev.kind || "");
            setActiveRun(activeKeyRef.current);
            setRun(ev.run);
            setKind(ev.kind || "");
            setStatus(ev.status || "running");
            setLines([]); // console is per-job
          } else if (ev.type === "run") {
            // A run of the experiment was discovered — make sure its tab exists (even before any metric).
            withRun(ev.run, ev.kind || "", (r) => ({ ...r, status: ev.status || r.status, base: ev.base || r.base }));
          } else if (ev.type === "idle") {
            setStatus("");
          } else if (ev.type === "log") {
            setLines((p) => [...p.slice(-800), ev.line]);
          } else if (ev.type === "progress") {
            const label: string = ev.label || "Caching";
            const stage: string = ev.stage || "caching";
            const key = rkey(ev.run || "", ev.kind || "");
            withRun(ev.run || "", ev.kind || "", (r) => {
              const live = liveFrom(r.live, label, stage, ev.progress, ev);
              if (stage === "caching" && typeof ev.memory_gb === "number") {
                const tick = (ramTick.current[key] = (ramTick.current[key] ?? 0) + 1);
                live.ram = [...live.ram.slice(-(POINT_CAP - 1)), { x: tick, y: ev.memory_gb }];
              }
              return { ...r, live };
            });
          } else if (ev.type === "metric") {
            const label: string = ev.label || ev.stage;
            const stage: string = ev.stage;
            const key = rkey(ev.run || "", ev.kind || "");
            const p: Progress | null = ev.progress ?? null;
            if (p) {
              // The x-axis depends on the stage. Training: a monotonic iteration counter (the tqdm step
              // resets each epoch, so we count training lines instead). Validation of a training: pinned to
              // that training iteration, so its pass collapses to one point on the training axis. Evaluation
              // / prediction: the case index (progress.step) — a running curve over cases, one point each.
              let x: number;
              if (stage === "training") {
                x = itCount.current[key] = (itCount.current[key] ?? 0) + 1;
              } else if (stage === "validation") {
                x = itCount.current[key] ?? 0;
              } else {
                x = p.step;
              }
              const shown: Record<string, number> = {};
              withRun(ev.run || "", ev.kind || "", (r) => {
                const series = { ...r.series };
                for (const [name, val] of Object.entries(ev.values as Record<string, number>)) {
                  const seriesKey = `${label}/${name}`;
                  series[seriesKey] = { label, stage, points: appendPoint(series[seriesKey]?.points ?? [], x, val), at: Date.now() };
                  if (!name.endsWith(":lr")) shown[name] = val;
                }
                return { ...r, series, live: liveFrom(r.live, label, stage, p, ev, shown) };
              });
            }
            if (!sawMetric.current) {
              sawMetric.current = true;
              setMetricNonce((n) => n + 1);
            }
          } else if (ev.type === "status") {
            if (ev.run) {
              withRun(ev.run, ev.kind || "", (r) => ({ ...r, status: ev.status }));
              if (rkey(ev.run, ev.kind || "") === activeKeyRef.current) setStatus(ev.status);
            }
            setDoneNonce((n) => n + 1);
          }
        }
      }
    }

    (async () => {
      while (!stopped) {
        try {
          await pump();
        } catch {
          if (stopped) return; // aborted on unmount / superseded by a newer run
        }
        if (stopped) return;
        await new Promise((r) => setTimeout(r, 1500)); // stream dropped (server restart?) — reconnect
      }
    })();

    return () => {
      stopped = true;
      ctrl.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session, runNonce]);

  return { lines, runs, activeRun, run, status, kind, metricNonce, doneNonce };
}
