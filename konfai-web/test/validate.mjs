// SPDX-License-Identifier: Apache-2.0
// Validate the konfai-web runtime with onnxruntime-node (same API as onnxruntime-web).
import ort from "onnxruntime-node";
import fs from "node:fs";
import { createSession, inferPatch } from "../src/runtime.mjs";

const manifest = JSON.parse(fs.readFileSync(new URL("../assets/manifest.json", import.meta.url)));
const model = fs.readFileSync(new URL("../assets/model.onnx", import.meta.url));

const session = await createSession(ort, model, { providers: ["cpu"] });
const n = manifest.input.channels * manifest.patch.size.reduce((a, b) => a * b, 1);
const patch = Float32Array.from({ length: n }, () => Math.random());
const { dims } = await inferPatch(ort, session, manifest, patch);

const expected = [1, manifest.output.channels, ...manifest.patch.size];
const ok = JSON.stringify(dims) === JSON.stringify(expected);
console.log(`konfai-web infer -> output dims ${JSON.stringify(dims)} (expected ${JSON.stringify(expected)})`);
console.log(ok ? "KONFAI_WEB_OK" : "KONFAI_WEB_FAIL");
process.exit(ok ? 0 : 1);
