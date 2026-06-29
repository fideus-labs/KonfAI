# konfai-rs — Phase-0 Spike Findings

> Result of the Phase-0 gate in [`RUST_BURN_IMPLEMENTATION_PLAN.md`](RUST_BURN_IMPLEMENTATION_PLAN.md).
> Run 2026-06-29 on the in-repo example 2D UNet. **Throwaway PoC** — sources under `spike/`, no product code.

## Verdict: ✅ GO (toolchain + KonfAI export path proven)

The full chain works **end-to-end with near-bit-exact parity**, no Python at inference:

```
KonfAI Network ──torch.onnx(dynamo)──▶ ONNX ──burn-onnx ModelGen──▶ Burn model ──forward──▶ output
   (example UNet, dim=2, 1,925,446 params)
```

| Hop | Result |
|---|---|
| KonfAI `UNet` → ONNX | parity vs torch **MAE 1.0e-8** |
| ONNX → `burn-onnx` `ModelGen` (build-time codegen) | imports + generates Rust; **all ops recognized** (Conv2d, ConvTranspose2d, MaxPool2d, Softmax, …) |
| Burn forward (NdArray / CPU backend) | parity vs onnxruntime **MAE 8.6e-9**, max 6e-8 |

## The recipe (reusable for Phase 1 — non-obvious bits the export must encode)

1. **Use the dynamo ONNX exporter.** A KonfAI `Network` overrides `state_dict()` with a custom signature (alias-based, no `destination`/`prefix`/`keep_vars` kwargs) that **breaks the legacy TorchScript exporter** (`torch.jit._unique_state_dict` → `TypeError`). `torch.onnx.export(..., dynamo=True)` traces via `torch.export.export(strict=False)` and sidesteps it. Needs `onnxscript`.
2. **Select the inference head explicitly.** `Network.forward` returns per-output-group results (empty without `.init()`). Reach the raw graph via `ModuleArgsDict.named_forward`, which yields `(dotted_name, tensor)` for every module, and wrap to return the **top-level full-res head** (`UNetBlock_0.Head.Softmax`) — skipping the multiple **deep-supervision heads** and the int64 `Argmax`.
3. **opset ≥ 18.** The dynamo exporter implements opset 18 and down-converts otherwise (warning).
4. **Make the ONNX self-contained.** The dynamo exporter writes weights as **external data** (`*.onnx.data` sidecar). `burn-onnx`'s `ModelGen` needs a single file → reload + re-save with weights embedded: `onnx.save(onnx.load(p), p, save_as_external_data=False)`.
5. **Rust deps (burn 0.21):** generated code uses the new **`burn-store`** crate (`BurnpackStore`, `ModuleSnapshot::load_from`). Required deps:
   ```toml
   [dependencies]
   burn = { version = "0.21", features = ["ndarray"] }   # default features ON (nn/record)
   burn-store = "0.21"
   [build-dependencies]
   burn-onnx = "0.21"
   ```
   `build.rs`: `burn_onnx::ModelGen::new().input("assets/model.onnx").out_dir("model/").run_from_script();`
   Load trained weights via the generated `Model::<B>::default()` (weights baked in); `Model::new(&device)` is random init.

## What is NOT yet proven (remaining Phase-0/1 validations)

- **Real target model `impact_synth`** (sCT, likely 3D + InstanceNorm/ReflectionPad/ConvTranspose). Only the **2D example UNet** is proven here. impact_synth needs an HF download (`KonfAIApp("VBoussot/ImpactSynth:…")`) and may hit different ops — **the op-coverage risk lives there, not in the chain itself.**
- **WGPU backend** (the browser-relevant one). Only **NdArray/CPU** is proven. WGPU is a backend swap in the same crate — next cheap validation.
- **WASM + WebGPU in a browser** — Phase 3.
- **Patch tiling + geometry-aware I/O** — Phase 4 (not exercised; this spike ran a single fixed patch).

## Reproduce

```
# Python (in pixi dev env; onnx deps are pip-installed in the same shell since pixi prunes them):
SPIKE_ASSETS=spike/konfai-rs-spike/assets pixi run --environment dev bash -c \
  "pip install -q onnx onnxruntime onnxscript && SPIKE_ASSETS=$SPIKE_ASSETS python spike/spike_export_unet.py"
# Rust:
cd spike/konfai-rs-spike && cargo run     # prints SPIKE_BURN_OK
```
