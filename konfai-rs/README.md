# konfai-rs

Portable, Python-free inference for KonfAI models via **Burn + ONNX**, targeting a
native binary, an embeddable lib, and (later) WASM/WebGPU — from one crate.

**Build-time codegen.** `burn-onnx` bakes the model *architecture* into the artifact
at build time (this is what enables WGPU/WASM + kernel fusion). One artifact per
architecture; weights are baked from the ONNX (runtime weight-swap via `burn-store`
is a later step).

## Usage
```
# 1. produce the contract with `konfai export` (Phase 1):
#    -> model.onnx + manifest.json
cp model.onnx konfai-rs/models/model.onnx

# 2. build (bakes the architecture):
cargo build --release

# 3. run on one patch (raw little-endian f32):
konfai-rs infer --manifest manifest.json --input patch.bin --output out.bin --backend wgpu
```

Volume NIfTI/MHA I/O + sliding-window tiling + overlap blend are the next phase;
today the unit of work is a single fixed-size patch.
