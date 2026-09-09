# konfai-rs, the vendored portable engine

The four files beside this README (`konfai_rs.js`, `konfai_rs.d.ts`, `konfai_rs_bg.wasm`,
`konfai_rs_bg.wasm.d.ts`) are a `wasm-pack` build of the konfai-rs engine, the Rust runtime that
executes the ONNX export and its manifest (`konfai/export.py`, `konfai_apps.bundle`) in the browser.
Studio's deployment pane loads them; nothing in this repository builds them.

Provenance of the files checked in:

| File | sha256 (first 16) |
|---|---|
| `konfai_rs_bg.wasm` | `c65df12506108f3e` |
| `konfai_rs.js` | `bdba542d4d6bb34f` |

Source: the konfai-rs Rust workspace (Burn backend for the WASM target), developed on the
`feat/konfai-rs-burn` line of this repository's history; the engine's own sources are not part of
this package. A rebuild replaces the four files together and updates the hashes above with the source
revision and the toolchain (`wasm-pack`, `rustc`) that produced them.

Compatibility: the engine consumes the manifest and program the export writes (`manifest.json`,
`program.json`); the export's parity with the native predictor is what
`tests/unit/test_onnx_export.py` checks on the Python side. A parity fixture through the engine
itself (Python output against the WASM output on one exported model) is not in this repository.
