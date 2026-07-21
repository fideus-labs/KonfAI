// SPDX-License-Identifier: Apache-2.0

// In-tab deployment of a KonfAI App: the app's model.onnx + manifest.json (bundled by
// `konfai-apps bundle --onnx`) are served by the BFF; the user picks a volume and inference runs
// 100% in the browser -- the konfai-rs core compiled to WASM runs the pipeline (the SAME Rust as the
// native engine) and onnxruntime-web (WebGPU) runs the model, rendered with NiiVue. The data never
// leaves the machine -- the zero-egress side of the apps.
import { useEffect, useRef, useState } from "react";
import { Niivue } from "@niivue/niivue";
import * as ort from "onnxruntime-web/webgpu";
import type { StudioApp } from "./AppZoo";
import init, { infer_volume } from "./konfai-rs/konfai_rs.js";
import wasmUrl from "./konfai-rs/konfai_rs_bg.wasm?url";

// The pipeline (transforms/tiling/blend/resample) is the konfai-rs core compiled to WASM -- the SAME
// Rust as the native engine, no per-app JS. The model runs via ort-web (WebGPU), injected as run_patch.
let wasmReady: Promise<unknown> | null = null;
const ensureWasm = () => (wasmReady ??= init(wasmUrl));

type Stage = "loading" | "ready" | "inferring" | "done" | "error" | "undeployable";

export default function Deploy({ app, onClose }: { app: StudioApp; onClose: () => void }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const nvRef = useRef<Niivue | null>(null);
  const [manifest, setManifest] = useState<any>(null);
  const [volume, setVolume] = useState<File | null>(null);
  const [stage, setStage] = useState<Stage>("loading");
  const [backend, setBackend] = useState("");
  const [message, setMessage] = useState("fetching the app's ONNX bundle…");
  const title = app.display_name || app.app_name || app.ref.split("/").pop();

  useEffect(() => {
    const nv = new Niivue({ backColor: [0.035, 0.08, 0.09, 1], show3Dcrosshair: true });
    nv.attachToCanvas(canvasRef.current!);
    nvRef.current = nv;
  }, []);

  // The app must have been exported with an ONNX bundle; fetch its manifest to confirm it is deployable.
  useEffect(() => {
    fetch(`/api/apps/manifest?ref=${encodeURIComponent(app.ref)}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("no ONNX bundle for this app"))))
      .then((m) => {
        setManifest(m);
        setStage("ready");
        setMessage("pick a volume and run — nothing leaves your machine.");
      })
      .catch((e) => {
        setStage("undeployable");
        setMessage(`This app has no portable ONNX bundle yet (${e.message}). Export one with \`konfai-apps bundle --onnx\`.`);
      });
  }, [app.ref]);

  async function run() {
    if (!manifest || !volume) return;
    try {
      setStage("inferring");
      setMessage("parsing volume…");
      const nv = nvRef.current!;
      await nv.loadVolumes([{ url: URL.createObjectURL(volume), name: volume.name }]);
      const vol = nv.volumes[0];
      const shape = [vol.dims![3], vol.dims![2], vol.dims![1]]; // [Z, Y, X]
      const spacing = [vol.pixDims![1], vol.pixDims![2], vol.pixDims![3]]; // (x, y, z)
      const data = Float32Array.from(vol.img as ArrayLike<number>);

      setMessage("loading the model…");
      const modelBuf = await (await fetch(`/api/apps/model?ref=${encodeURIComponent(app.ref)}`)).arrayBuffer();
      let session: ort.InferenceSession;
      try {
        session = await ort.InferenceSession.create(modelBuf, { executionProviders: ["webgpu"] });
        setBackend("WebGPU");
      } catch {
        session = await ort.InferenceSession.create(modelBuf, { executionProviders: ["wasm"] });
        setBackend("WASM (CPU)");
      }
      const inName = session.inputNames[0];
      const outName = session.outputNames[0];
      const runPatch = async (patch: Float32Array, dims: Uint32Array) =>
        (await session.run({ [inName]: new ort.Tensor("float32", patch, Array.from(dims)) }))[outName].data as Float32Array;

      setMessage("running client-side inference…");
      await ensureWasm();
      const { data: result, shape: outShape, channels } = (await infer_volume(
        data,
        Uint32Array.from(shape),
        manifest.input?.channels ?? 1,
        Float64Array.from(spacing),
        JSON.stringify(manifest),
        runPatch,
      )) as { data: Float32Array; shape: Uint32Array; channels: number };

      const overlay = nv.volumes[0].clone();
      overlay.img = result.subarray(0, outShape[0] * outShape[1] * outShape[2]);
      overlay.setColormap?.("warm");
      overlay.opacity = 0.6;
      nv.addVolume(overlay);
      setStage("done");
      setMessage(`done — ${channels}×${outShape.join("×")} rendered as overlay (${backend}).`);
    } catch (e) {
      setStage("error");
      setMessage(`error: ${(e as Error).message}`);
    }
  }

  const busy = stage === "inferring" || stage === "loading";
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
          <label className="deploy-drop">
            <span>{volume ? volume.name : "Choose a volume (NIfTI / MHA / NRRD)"}</span>
            <input
              type="file"
              accept=".nii,.nii.gz,.mha,.mhd,.nrrd"
              disabled={stage === "undeployable"}
              onChange={(e) => setVolume(e.target.files?.[0] ?? null)}
            />
          </label>
          <button className="deploy-run" disabled={!manifest || !volume || busy} onClick={run}>
            {stage === "inferring" ? "Inferring…" : "Run inference"}
          </button>
          <div className={`deploy-status ${stage}`}>
            {backend && <span className="backend">{backend}</span>}
            {message}
          </div>
        </aside>
        <canvas ref={canvasRef} className="deploy-canvas" />
      </div>
    </div>
  );
}
