// SPDX-License-Identifier: Apache-2.0
//! Build-time ONNX -> Burn codegen. The model architecture is compiled into the
//! artifact (this is what enables WGPU/WASM portability + kernel fusion). Provide
//! `models/model.onnx` (produced by `konfai export`). One artifact per architecture.
use burn_onnx::ModelGen;
use std::path::Path;

fn main() {
    let onnx = "models/model.onnx";
    println!("cargo:rerun-if-changed={onnx}");
    if !Path::new(onnx).exists() {
        panic!(
            "konfai-rs: missing `{onnx}`. Export one with `konfai export` (Phase 1) and place \
             it at konfai-rs/models/model.onnx. burn-onnx bakes the architecture at build time."
        );
    }
    ModelGen::new().input(onnx).out_dir("model/").run_from_script();
}
