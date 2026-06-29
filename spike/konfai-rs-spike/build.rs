// Phase-0 spike: import the exported KonfAI UNet ONNX into Burn at build time.
// This is THE gate: does burn-onnx codegen Rust from a real KonfAI model graph?
use burn_onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("assets/model.onnx")
        .out_dir("model/")
        .run_from_script();
}
