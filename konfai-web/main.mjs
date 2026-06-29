// SPDX-License-Identifier: Apache-2.0
// Browser demo: load the bundle and run one patch on WebGPU (falls back to wasm).
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.webgpu.min.mjs";
import { createSession, inferPatch } from "./src/runtime.mjs";

const log = (m) => (document.getElementById("log").textContent += m + "\n");

async function run(modelUrl, manifestUrl) {
  const manifest = await (await fetch(manifestUrl)).json();
  log(`manifest: ${manifest.input.channels}ch in, ${manifest.output.channels}ch out, patch ${manifest.patch.size}`);
  const providers = navigator.gpu ? ["webgpu"] : ["wasm"];
  log(`backend: ${providers[0]}${navigator.gpu ? " (WebGPU)" : " (no WebGPU -> CPU wasm)"}`);
  const session = await createSession(ort, modelUrl, { providers });
  const n = manifest.input.channels * manifest.patch.size.reduce((a, b) => a * b, 1);
  const patch = Float32Array.from({ length: n }, () => Math.random());
  const t0 = performance.now();
  const { dims } = await inferPatch(ort, session, manifest, patch);
  log(`output dims ${JSON.stringify(dims)} in ${(performance.now() - t0).toFixed(1)} ms`);
  log("OK — model ran in the browser, no Python.");
}

document.getElementById("run").onclick = () => run("assets/model.onnx", "assets/manifest.json").catch((e) => log("ERROR: " + e.message));
