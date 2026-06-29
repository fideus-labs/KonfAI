# konfai-web

Portable, in-browser inference for KonfAI models via **ONNX Runtime Web + WebGPU**.
The author publishes a bundle (`model.onnx` + `manifest.json`, produced by
`konfai-apps bundle --onnx`); konfai-web runs **any** such model in the browser with no
per-model build and no Python — the generic counterpart to the native `konfai-rs` runtime.

## Try it
```
npm install          # onnxruntime-web (+ onnxruntime-node for the test)
npm test             # validate the runtime with onnxruntime-node
npm run serve        # serve index.html; open in a WebGPU browser and click "Run"
```
Drop a bundle's `model.onnx` + `manifest.json` into `assets/`.
