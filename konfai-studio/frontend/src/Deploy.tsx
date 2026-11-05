// SPDX-License-Identifier: Apache-2.0

// In-tab deployment of a KonfAI App on KonfAI Edge. The app's repository carries a `web/` folder: its
// `catalog.json` names, per app, the configs, the programs `konfai.portable` compiled from its Prediction.yml,
// their `.bpk` weights and the engine that runs them. The BFF serves those files; the engine (WebAssembly,
// the models on the tab's WebGPU device) decides what a run picks and runs it, and NiiVue renders the result.
// The volume never leaves the machine.
import { useEffect, useRef, useState } from "react";
import { Niivue } from "@niivue/niivue";
import type { StudioApp } from "./AppZoo";
import type * as KonfaiEdge from "./konfai-edge/konfai_edge";

type Engine = typeof KonfaiEdge;
type Variant = { weights: string; program?: string; uncertainty?: { program?: string; evaluation?: string } };
type Entry = {
  name: string;
  engine?: string;
  runtime: string;
  configs: { name: string; url: string }[];
  variants: Variant[];
  evaluations?: { program?: string }[];
};
type Chosen = { program: string; checkpoints: string[]; tta: number };
type Stage = "loading" | "ready" | "inferring" | "done" | "error" | "undeployable";

const fileUrl = (ref: string, path: string) =>
  `/api/apps/web/file?ref=${encodeURIComponent(ref)}&path=${encodeURIComponent(path)}`;

async function fetched(ref: string, path: string): Promise<Response> {
  const response = await fetch(fileUrl(ref, path));
  if (!response.ok) throw new Error(`web/${path}: the server said ${response.status}`);
  return response;
}

// One engine per wasm file, compiled once per tab, with its WebGPU device.
const engines = new Map<string, Promise<Engine>>();
function engineOf(ref: string, runtime: string): Promise<Engine> {
  const key = fileUrl(ref, `${runtime}/konfai_edge_bg.wasm.gz`);
  if (!engines.has(key))
    engines.set(
      key,
      (async () => {
        const glue = URL.createObjectURL(
          new Blob([await (await fetched(ref, `${runtime}/konfai_edge.js`)).text()], { type: "text/javascript" }),
        );
        let engine: Engine;
        try {
          engine = await import(/* @vite-ignore */ glue);
        } finally {
          URL.revokeObjectURL(glue);
        }
        const gzipped = (await fetched(ref, `${runtime}/konfai_edge_bg.wasm.gz`)).body!;
        const wasm = await new Response(gzipped.pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
        await engine.default({ module_or_path: wasm });
        await engine.init_webgpu();
        return engine;
      })().catch((e) => {
        engines.delete(key);
        throw e;
      }),
    );
  return engines.get(key)!;
}

// The engine's format token of a volume's file name.
const formatOf = (name: string) => name.match(/\.(nii\.gz|nii|mha|mhd|nrrd)$/i)?.[1].toLowerCase() ?? "";

export default function Deploy({ app, onClose }: { app: StudioApp; onClose: () => void }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const nvRef = useRef<Niivue | null>(null);
  const edgeRef = useRef<{ engine: Engine; entry: Entry; api: KonfaiEdge.App } | null>(null);
  const [models, setModels] = useState(0);
  const [ensemble, setEnsemble] = useState(0);
  const [result, setResult] = useState<{ url: string; name: string } | null>(null);
  // The blob URL of the result the Save link points at: freed when a new run replaces it or the pane closes.
  const resultUrl = useRef<string | null>(null);
  const [volume, setVolume] = useState<File | null>(null);
  const [stage, setStage] = useState<Stage>("loading");
  const [message, setMessage] = useState("reading the app's in-browser build…");
  const title = app.display_name || app.app_name || app.ref.split("/").pop();

  useEffect(() => {
    // No loading text: this canvas waits for a file the user has not picked yet, and NiiVue's default
    // paints "loading ..." across it, which reads as a pane that is stuck.
    const nv = new Niivue({ backColor: [0.035, 0.08, 0.09, 1], show3Dcrosshair: true, loadingText: "" });
    nv.attachToCanvas(canvasRef.current!);
    nvRef.current = nv;
    return () => {
      if (resultUrl.current) URL.revokeObjectURL(resultUrl.current);
    };
  }, []);

  // Escape leaves: this pane covers the whole app, and an app with no in-browser build offers nothing to
  // do in it, so a small ✕ was the only way back out.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // The app's entry in its repository's web/catalog.json, then its engine and the app as the engine reads it.
  useEffect(() => {
    let live = true;
    const refuse = (text: string) => Object.assign(new Error(text), { undeployable: true });
    (async () => {
      const response = await fetch(`/api/apps/web?ref=${encodeURIComponent(app.ref)}`);
      if (!response.ok)
        throw refuse((await response.json().then((b) => b.detail, () => "")) || `the server said ${response.status}`);
      const entry: Entry = await response.json();
      if (entry.engine !== "konfai-edge")
        throw refuse(`This app runs on ${entry.engine ?? "no"} engine; Studio runs the KonfAI Edge apps.`);
      setMessage("loading the KonfAI Edge engine…");
      const engine = await engineOf(app.ref, entry.runtime);
      const dir = `${entry.variants[0].weights}/`;
      const appJson = await (await fetched(app.ref, entry.configs.find((c) => c.name === "app.json")!.url)).text();
      const programs = entry.variants.flatMap((v) => [v.program, v.uncertainty?.program, v.uncertainty?.evaluation]);
      const files = [...programs, ...(entry.evaluations ?? []).map((e) => e.program)]
        .filter((p): p is string => !!p && p.startsWith(dir))
        .map((p) => p.slice(dir.length));
      const api = new engine.App(entry.name, appJson, files);
      if (!live) return;
      edgeRef.current = { engine, entry, api };
      setModels(JSON.parse(api.describe()).bounds?.models?.length ?? 0);
      setStage("ready");
      setMessage("pick a volume and run: nothing leaves your machine.");
    })().catch((e: Error & { undeployable?: boolean }) => {
      if (!live) return;
      setStage(e.undeployable ? "undeployable" : "error");
      setMessage(e.undeployable ? e.message : `error: ${e.message}`);
    });
    return () => {
      live = false;
    };
  }, [app.ref]);

  async function run() {
    const edge = edgeRef.current;
    if (!edge || !volume) return;
    const { engine, entry, api } = edge;
    let predictor: KonfaiEdge.Predictor | null = null;
    try {
      setStage("inferring");
      // What konfai-apps runs for these options: the program and the members it keeps.
      const chosen: Chosen = JSON.parse(api.run(JSON.stringify(ensemble ? { ensemble } : {})));
      const dir = entry.variants[0].weights;
      const programText = await (await fetched(app.ref, `${dir}/${chosen.program}`)).text();
      const program = JSON.parse(programText);
      setMessage("downloading the weights…");
      const names = engine.selected_weights(programText, chosen.checkpoints, chosen.tta);
      const weights = await Promise.all(
        names.map(async (n) => new Uint8Array(await (await fetched(app.ref, `${dir}/${n}`)).arrayBuffer())),
      );
      // The engine settles what the tab may hold from what it declares of its memory.
      const memory = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
      const { budget } = JSON.parse(
        engine.resolve_capabilities(
          JSON.stringify({ device_memory: memory ? memory * 2 ** 30 : undefined, address_space: 2 ** 32 }),
        ),
      );
      predictor = engine.Predictor.selected(
        programText, chosen.checkpoints, chosen.tta, names, weights, "webgpu", undefined, budget, undefined,
      );

      setMessage("running client-side inference…");
      const format: string = program.sources[0].format;
      const raw = new Uint8Array(await volume.arrayBuffer());
      const from = formatOf(volume.name);
      const data = from === format ? raw : engine.transcode_volume(raw, from, format, 2 ** 31);
      const written: { path: string; bytes: Uint8Array }[] = await predictor.predict(
        [`${program.cases[0]}/${program.input_group}.${format}`],
        [data],
      );
      const write = program.outputs[0].write;
      const output = written.find((w) => w.path.endsWith(`/${write.group}.${write.format}`));
      if (!output) throw new Error(`the run wrote no ${write.group}`);

      const nv = nvRef.current!;
      const input = URL.createObjectURL(volume);
      try {
        await nv.loadVolumes([{ url: input, name: volume.name }]);
      } finally {
        URL.revokeObjectURL(input);
      }
      const url = URL.createObjectURL(new Blob([output.bytes as BlobPart]));
      if (resultUrl.current) URL.revokeObjectURL(resultUrl.current);
      resultUrl.current = url;
      setResult({ url, name: output.path.split("/").pop()! });
      await nv.addVolumeFromUrl({
        url,
        name: output.path.split("/").pop()!,
        colormap: "warm",
        opacity: 0.6,
      });
      setStage("done");
      const kept = chosen.checkpoints.length;
      setMessage(`done: ${kept} model${kept > 1 ? "s" : ""}, rendered as overlay (WebGPU).`);
    } catch (e) {
      // A panic in the engine throws a bare RuntimeError: the message is the panic's.
      setStage("error");
      setMessage(`error: ${engine.last_panic() || (e as Error).message}`);
    } finally {
      predictor?.free();
      engine.release_device_memory();
    }
  }

  const busy = stage === "inferring" || stage === "loading";
  // An app with no in-browser build has nothing to run: it gets the reason and the way back, not a file
  // picker and a Run button that cannot do anything.
  const undeployable = stage === "undeployable";
  return (
    <div className="deploy-full">
      <header>
        <span className="deploy-eyebrow">Deploy · in-tab inference</span>
        <h2>{title}</h2>
        <span className="deploy-egress">zero egress · runs on your machine</span>
        <button className="close" onClick={onClose} aria-label="Close">✕</button>
      </header>
      <div className="deploy-body">
        <aside className="deploy-side">
          <div className="deploy-ref">{app.ref}</div>
          {!undeployable && (
            <>
              <label className="deploy-drop">
                <span>{volume ? volume.name : "Choose a volume (NIfTI / MHA / NRRD)"}</span>
                <input
                  type="file"
                  accept=".nii,.nii.gz,.mha,.mhd,.nrrd"
                  onChange={(e) => setVolume(e.target.files?.[0] ?? null)}
                />
              </label>
              {models > 1 && (
                <label className="deploy-ref">
                  Models in the ensemble{" "}
                  <select value={ensemble} onChange={(e) => setEnsemble(Number(e.target.value))} disabled={busy}>
                    <option value={0}>the app's default</option>
                    {Array.from({ length: models }, (_, i) => (
                      <option key={i + 1} value={i + 1}>
                        {i + 1}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              <button className="deploy-run" disabled={!volume || busy} onClick={run}>
                {stage === "inferring" ? "Inferring…" : "Run inference"}
              </button>
            </>
          )}
          <div className={`deploy-status ${stage}`}>
            {message}
          </div>
          {stage === "done" && result && (
            <a className="deploy-run" href={result.url} download={result.name}>
              Save the result
            </a>
          )}
          {undeployable && (
            <button className="deploy-run" onClick={onClose}>
              Back to the apps
            </button>
          )}
        </aside>
        <canvas ref={canvasRef} className="deploy-canvas" />
      </div>
    </div>
  );
}
