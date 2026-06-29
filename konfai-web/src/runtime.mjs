// SPDX-License-Identifier: Apache-2.0
// konfai-web runtime: load a konfai-rs bundle (model.onnx + manifest.json) and run a
// patch through ONNX Runtime. Runtime-agnostic: pass the `ort` module (onnxruntime-web
// in the browser, onnxruntime-node for tests) — the InferenceSession API is identical.

/** Create an inference session from a model URL/path or an ArrayBuffer. */
export async function createSession(ort, model, options = {}) {
  return ort.InferenceSession.create(model, { executionProviders: options.providers ?? ["wasm"], ...options });
}

/** Run one fixed-size patch ([1, C, ...patch.size]) declared by the manifest. */
export async function inferPatch(ort, session, manifest, patchData) {
  if (manifest.patch.dim !== 2) throw new Error("konfai-web: only 2D/2.5D patches are supported");
  const dims = [1, manifest.input.channels, ...manifest.patch.size];
  const expected = dims.reduce((a, b) => a * b, 1);
  if (patchData.length !== expected) {
    throw new Error(`konfai-web: patch has ${patchData.length} values, expected ${expected} for ${dims}`);
  }
  const input = new ort.Tensor("float32", patchData, dims);
  const output = await session.run({ [manifest.input.name]: input });
  const tensor = output[manifest.output.name] ?? Object.values(output)[0];
  return { data: tensor.data, dims: tensor.dims };
}
