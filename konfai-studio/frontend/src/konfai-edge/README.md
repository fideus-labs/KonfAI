# KonfAI Edge, the engine of the Deploy pane

Studio's Deploy pane runs an app in the browser on KonfAI Edge: the app's Prediction.yml compiled to a
program (`konfai.portable.compile_prediction`), its models on Burn, on the tab's WebGPU device.

The engine is not part of this package. Each app's repository carries a `web/` folder whose `catalog.json`
names, per app, its `app.json`, its programs, their `<stem>.bpk` weights and the engine that runs them
(`<runtime>/konfai_edge.js` and `<runtime>/konfai_edge_bg.wasm.gz`). The BFF serves those files
(`/api/apps/web`, `/api/apps/web/file`), and the pane loads the engine it is handed. An app whose repository
has no `web/` folder is refused by the pane, with the reason.

`konfai_edge.d.ts` declares the part of the engine's API the pane calls, copied from the `konfai_edge.d.ts`
of KonfAI Edge 0c39de131 (wasm-bindgen 0.2.129, rustc 1.96.0). A rebuild of the engine that changes those
signatures updates this file and the revision above.
