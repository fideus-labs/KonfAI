# konfai-rs — Phase-0 Spike Findings

> Result of the Phase-0 gate in [`RUST_BURN_IMPLEMENTATION_PLAN.md`](RUST_BURN_IMPLEMENTATION_PLAN.md).
> Run 2026-06-29. **Throwaway PoC** — sources under `spike/`, no product code.

## Verdict: ✅ GO — proven on BOTH the example UNet AND the real `impact_synth` model, on CPU AND WebGPU

The full chain works **end-to-end with near-bit-exact parity**, no Python at inference:

```
KonfAI Network ──torch.onnx(dynamo)──▶ ONNX ──burn-onnx ModelGen──▶ Burn model ──forward──▶ output
```

| Model | → ONNX (vs torch) | burn-onnx import | Burn CPU (vs ORT) | Burn WGPU (vs ORT) |
|---|---|---|---|---|
| Example UNet (2D, 1.9M params) | MAE **1.0e-8** | ✅ all ops | MAE **8.6e-9** | — |
| **`impact_synth` MR** (smp UNet++ resnet34, 2.5D 5-ch, **26M params**, Tanh head) | MAE **6.4e-6** | ✅ all ops | MAE **6.3e-6** | MAE **5.5e-6** |

**The #1 risk (op coverage) is retired for `impact_synth`.** Its ops (resnet34: Conv2d/BatchNorm/ReLU/MaxPool/Add; UNet++ decoder: Upsample/Concat/Conv; Tanh) all import and run. It is **2D / 2.5D (no 3D conv, no exotic ops)** — the favourable case. WGPU ran on an NVIDIA RTX PRO 5000 (Vulkan); since **WGPU is the same backend that compiles to WASM/WebGPU**, the browser compute path is strongly de-risked.

> `impact_synth` brings an external dep `segmentation_models_pytorch` (smp) on the Python/export side — needed to *build* the model for export, not at Rust inference. The exported ONNX is self-contained.

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

## Proven now (2026-06-29)

- ✅ Toolchain + KonfAI export path (example UNet).
- ✅ **Real `impact_synth` MR** model imports + runs with parity (CPU **and** WGPU).
- ✅ **WGPU backend** (NVIDIA RTX PRO 5000, Vulkan).

## What is NOT yet proven (remaining validations)

- **WASM + WebGPU in a browser** — Phase 3. (WGPU compute proven natively; remaining = the `wasm-bindgen` + WASM build + browser WebGPU device init.)
- **Patch tiling + geometry-aware I/O** — Phase 4 (this spike ran a single fixed patch, random-weight or real-weight; the per-volume sliding-window + overlap blend + NIfTI/geometry path is untouched).
- **Real trained weights end-to-end accuracy** — the spike used the architecture with the exported weights; a full sCT-vs-reference clinical parity (vs the Python `Predictor`, with the 5-fold ensemble) is Phase 2/4.
- **Other variants** (CBCT, MR_CBCT, Finetune) — same architecture, expected to behave identically.

## Reproduce

```
# Python (in pixi dev env; onnx deps are pip-installed in the same shell since pixi prunes them):
SPIKE_ASSETS=spike/konfai-rs-spike/assets pixi run --environment dev bash -c \
  "pip install -q onnx onnxruntime onnxscript && SPIKE_ASSETS=$SPIKE_ASSETS python spike/spike_export_unet.py"
# Rust:
cd spike/konfai-rs-spike && cargo run     # prints SPIKE_BURN_OK
```
