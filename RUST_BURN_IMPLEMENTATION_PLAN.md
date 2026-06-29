# konfai-rs — Implementation Plan (portable inference engine)

> Companion to [`RUST_BURN_ROADMAP.md`](RUST_BURN_ROADMAP.md) (the *why*). This is the *how*.
> Status: **plan only — no code written yet. Nothing is pushed.**
> **Base branch:** `pr/audit-followup`. **Work branch:** `konfai-rs`.

## 0. Goal & scope (decided)

Build **`konfai-rs`** — a **portable inference engine** (one Rust/Burn crate, a separate package sibling to `konfai-apps`/`konfai-mcp`) that runs the **feed-forward inference** of a KonfAI model from an exported `.onnx`, with **no Python at inference time**.

**The engine is one codebase; the browser is one compile target, not the identity.** From the same crate:

```
                 konfai-rs  (ONNX→Burn + patch tiling + I/O)
        ┌──────────────┬──────────────────┬────────────────────┐
   native binary     WASM + WebGPU     embeddable lib        (future)
   CUDA/Metal/CPU      browser           Slicer / edge        server/batch
```

- **First validated target:** **native** (CPU Flex + WGPU) — fastest to iterate and debug, and itself a useful deliverable (edge / offline clinic / Slicer / server).
- **Showcase target:** the **browser** (WASM + WebGPU), layered on top once the native core is proven.
- **First model:** **IMPACT-Synth** — synthetic-CT (sCT) generation from MR / CBCT (`VBoussot/ImpactSynth`, a supervised CNN generator, resolved today through `KonfAIApp`). Feed-forward → a valid candidate.
- **Parity metric:** synthesis → **MAE / PSNR on HU values** (+ SSIM), *not* Dice.
- **Out of scope (MVP):** training, TTA, ensembling, MC-dropout (`--tta/--ensemble/--mc`), registration/diffusion apps. Single deterministic forward only.

The KonfAI Python core is **not touched**. `konfai-rs` is additive and lives on its own branch/package.

## 1. The one architectural principle

**Push everything ONNX can express into the exported graph; keep Rust thin.**

| Stage | Where | Note (synthesis-specific) |
|---|---|---|
| Input normalization (MR/CBCT intensity → model range) | **ONNX graph** | data-dependent stats via ReduceMin/Max/Mean |
| Resample / resize | **ONNX graph** (`Resize`) | if the app resamples |
| Generator forward | **ONNX → Burn** (Flex/WGPU/CUDA/Metal) | the point |
| Output **HU denormalization + clamp/Tanh** | **ONNX graph** | sCT must come out in HU; fold the inverse-normalize in |
| File parse + geometry (Origin/Spacing/Direction) | **Rust** | not expressible in ONNX |
| 3D patch sliding-window + overlap blend | **Rust** | control flow over tiles |
| TTA / ensemble / MC | deferred | orchestration over many forwards |

This keeps the engine small and concentrates risk on **(a) burn-onnx import of the generator** and **(b) I/O + tiling** — identical regardless of native vs browser target.

## 2. Hard problems, named up front

1. **What to export.** IMPACT-Synth is a routed `ModuleArgsDict`; export the **inference forward path / prediction output module** the `Predictor` consumes, at a **fixed patch size**, opset 17.
2. **Synthesis op coverage in burn-onnx.** Generators often use **InstanceNorm** (reported supported), **ReflectionPad** (ONNX `Pad` mode=reflect — *verify*), **ConvTranspose / Resize upsampling**, **Tanh/clamp**. The spike must confirm these import.
3. **burn-onnx import reliability** (not op coverage) — the real failure mode (burn-onnx #18). **Phase 0 gates everything.**
4. **In-browser medical I/O** (later phase). The `nifti` Rust crate may not target WASM cleanly → may parse in JS and hand a typed array to WASM. DICOM deferred. (Native target has no such constraint.)
5. **WebGPU availability** (later phase). WGPU-in-WASM only where the browser ships WebGPU; a 3D generator on Flex-CPU-WASM is likely too slow → treat WebGPU as required in-browser, message unsupported browsers. (Native target uses CUDA/Metal/CPU freely.)
6. **Model provenance/licensing.** IMPACT-Synth weights live on HF; confirm redistribution terms before bundling an exported `.onnx` in a public demo.

## 3. Phased plan (native-first, browser layered on top)

Each phase: deliverable · key files · tests/benchmarks · **GO/NO-GO gate** · effort.

---

### Phase 0 — Spike / decision gate ⟵ START HERE
**Make-or-break. Build nothing permanent until this passes. Native only — the cheapest test of the riskiest assumption.**

- **Steps:**
  1. Resolve an IMPACT-Synth model via `KonfAIApp("VBoussot/ImpactSynth:<model>")`; read its `Prediction.yml` for `patch_size`, groups, normalization; confirm 2.5D vs 3D.
  2. `torch.onnx.export` the underlying generator `nn.Module` at that fixed `patch_size`, opset 17, **folding input-normalize + HU-denormalize into the graph**.
  3. `onnxruntime` parity vs torch (MAE ≤ 1 HU, high PSNR) — validates the export *before* blaming Burn.
  4. `burn-onnx` `ModelGen` (`build.rs`) → one native binary → run a patch on **Flex (CPU)** then **WGPU**; parity vs the onnxruntime reference.
- **Files (throwaway `spike/`, not committed to the product):** `spike/export_impact_synth.py`, `spike/konfai-rs-spike/{Cargo.toml,build.rs,src/main.rs}`.
- **GATE:**
  - **GO** → generator imports + compiles + matches torch (MAE ≤ 1 HU) on CPU **and** WGPU. Proceed to Phase 1.
  - **NO-GO** → import/compile failure (likely InstanceNorm/ReflectionPad/ConvTranspose) or numeric divergence unfixable in <1–2 days. **Park**, write up the failing op(s), revisit next Burn release.
- **Effort:** 1–3 days.

---

### Phase 1 — ONNX export (Python, no-regret value)
**Productionize the spike's export. Valuable even if Burn is dropped (general interop; ONNXRuntime-Web is a fallback deploy path).**

- **Files:** `konfai/export/__init__.py`, `konfai/export/onnx.py` — `export_to_onnx(model, patch_size, output_module, opset=17, fold_pre=True, fold_post=True)`; `konfai/main.py` — `EXPORT` CLI state (mirrors PREDICTION) or a `konfai-export` entrypoint; docs.
- **Tests:** `tests/unit/test_onnx_export.py` — onnxruntime-vs-torch parity (MAE/PSNR); assert opset ≥16, fixed dims, HU range preserved. Update `tests/unit/test_config.py` if config binding added.
- **Deps:** declare `onnx`/`onnxruntime` as an `export` extra in `pyproject.toml` **in the same commit** (AGENTS.md §13.4).
- **GATE:** parity test green in CI. Proceed to Phase 2.
- **Effort:** 3–5 days.

---

### Phase 2 — `konfai-rs` native core + CLI (the primary deliverable)
**A usable, Python-free inference binary. Separate package, off the wheel.**

- **Files:**
  - `konfai-rs/Cargo.toml` — deps `burn`, `burn-onnx` (build), feature flags `wgpu`/`cpu`(Flex)/`cuda`/`metal`; `crate-type` includes `cdylib` (for the later WASM target).
  - `konfai-rs/build.rs` — `ModelGen` over `konfai-rs/models/impact_synth.onnx`.
  - `konfai-rs/src/{lib.rs,model.rs,infer.rs,patch.rs,tensor_io.rs}` — `infer_patch`, sliding-window + overlap-blend (port of `patching.py`, deterministic order per AGENTS.md §10), `.safetensors`/`.npy` tensor I/O.
  - `konfai-rs/src/bin/cli.rs` — `konfai-rs infer --model … --input … --backend {cpu,wgpu,cuda}`.
  - `konfai-rs/tests/parity.rs` — Burn output vs committed onnxruntime reference (MAE/PSNR tolerance).
  - `konfai-rs/README.md`, CI `.github/workflows/konfai-rs.yml` (own workflow; **not** in the wheel build).
- **Benchmarks:** `konfai-rs/benches/` (criterion) — per-patch latency torch(CUDA) vs Burn(WGPU/CUDA); **binary size + cold-start**.
- **GATE:** native parity green on CPU + WGPU; benchmarks recorded. This is already shippable as an edge/Slicer/server binary. Proceed to Phase 3.
- **Effort:** ~1 week.

---

### Phase 3 — Browser target: WASM + WebGPU (the showcase)
**The same crate, compiled to WASM, running on WebGPU in a browser tab.**

- **Files:**
  - `konfai-rs/src/wasm.rs` — `wasm-bindgen` exports (`init`, `infer_patch(Float32Array, dims) -> Float32Array`); WGPU backend init from the browser GPU device.
  - `konfai-rs/web/{index.html,main.js,vite.config.*}` — load the WASM, feed a patch, render output; clear "WebGPU unavailable" message.
  - `konfai-rs/web/tests/` — Playwright smoke (model loads + runs one patch in headless WebGPU).
- **GATE:** model runs one patch in a real WebGPU browser; output matches the native reference; WASM size + first-inference latency recorded. Proceed to Phase 4.
- **Effort:** 1–2 weeks.

---

### Phase 4 — Full pipeline (native first, then browser)
**End-to-end: a volume in → sCT out, with no server.**

- **Files:**
  - `konfai-rs/src/io/{mod.rs,nifti.rs}` (+ `mha.rs`) — parse to typed array + geometry; native uses the Rust crate, browser decides Rust/WASM vs JS by what compiles.
  - `konfai-rs/src/pipeline.rs` — `parse → (graph pre) → 3D tile → per-patch forward → overlap blend → (graph post) → sCT volume`.
  - `konfai-rs/web/` — upload UI, slice viewer (canvas), sCT download (NIfTI) / overlay.
- **GATE:** full-volume sCT (native, then browser) matches the Python `Predictor` output within tolerance (MAE/PSNR) on a fixture case.
- **Effort:** 2–3 weeks (tiling + geometry + viewer).

---

### Phase 5 — Package polish & reuse (stretch)
- Package the deliverable (native binaries per platform; static web build / npm), demo page, docs.
- Embeddable lib reuse: Slicer plugin / edge / batch server from the same crate.
- Optionally re-add **TTA / ensemble / MC-dropout** as multi-forward orchestration in Rust.
- **Effort:** 1–2 weeks.

## 4. Cross-cutting conventions

- **Separate package, off the wheel, own branch.** `konfai-rs` mirrors `konfai-mcp`/`konfai-apps`: own `Cargo.toml`/build, own tests + CI, never imported by core, depends only on the public API + the exported `.onnx`. Lives on the **`konfai-rs` branch (based on `pr/audit-followup`)**, **local-only for now (no push)** — same posture as the `konfai-mcp` branch.
- **Parity-first.** Every phase validates against a torch/onnxruntime reference within a fixed tolerance; parity dumps are committed fixtures.
- **No new core runtime deps** (AGENTS.md §13.4). The Python side adds only an ONNX exporter; `onnx`/`onnxruntime` become an `export` extra declared in the same commit.
- **Commits:** Conventional Commits, no AI branding/trailers (AGENTS.md §12). **No push without an explicit ask.**

## 4b. konfai-apps integration — the seam, runtime selector, and manifest contract

Mapped from the konfai-apps source. konfai-rs is a **runtime**, not a surface; it plugs into konfai-apps at exactly **one seam** and leaves everything else untouched.

**The run flow & the seam.** `KonfAIApp.infer()` (`konfai-apps/konfai_apps/app.py`) runs under `@run_distributed_app` (isolated temp workspace) and:
1. `_write_inputs_to_dataset()` → symlinks inputs into `./Dataset/P{idx}/Volume_{i}`,
2. `app_repository.install_inference()` → downloads `.pt` + `.yml` + `.py`, pip-installs `requirements.txt`,
3. **`konfai.predictor.predict(models_path, …, Prediction.yml)`** ← **THE SEAM** (`app.py:900`),
4. torch forward (`predictor.py:565` → `network.py:710`), writes `./Predictions/`,
5. `copytree(./Predictions → output)`.

**Runtime selector.** Branch at `app.py:900` on an `app.json` `runtime` field:
```
runtime: pytorch  → konfai.predictor.predict(.pt, Prediction.yml)            # today
runtime: portable → konfai_rs(model.onnx, manifest.json, ./Dataset → ./Predictions)
```
A `--runtime {pytorch,portable}` flag is added in `cli.py:add_common_konfai_apps`; the dispatch branch sits just before the `predict()` call.

**What stays in konfai-apps (Python) vs moves to konfai-rs (Rust):**

| Stays in konfai-apps | Moves to konfai-rs |
|---|---|
| App resolution Local/HF/Remote; download/cache | Load `model.onnx` |
| `./Dataset/P*/Volume_*` layout + input symlinking | Read `./Dataset`, geometry-aware |
| `./Predictions` collection → output | Patch sliding-window + overlap blend; write `./Predictions` (same layout) |
| Surfaces: CLI, FastAPI server, remote client, SSE | Forward + reduction (Mean/Median for TTA/ensemble) |
| VRAM-aware config mutation; `app.json` schema | Pre/post folded into the ONNX graph |

**The contract = `model.onnx` + `manifest.json`.** Proposed minimal manifest (produced by the Phase-1 exporter, consumed by konfai-rs):
```json
{
  "konfai_rs_manifest": 1,
  "model": "model.onnx",
  "input":  { "name": "input",  "group": "Volume_0", "channels": 5, "dtype": "f32" },
  "output": { "name": "output", "group": "sCT",      "channels": 1, "dtype": "f32" },
  "patch":  { "size": [256, 256], "overlap": [0, 0], "dim": 2, "extend_slice": 2 },
  "preprocess_in_graph": true,
  "postprocess_in_graph": true,
  "reduction": "mean",
  "geometry": "preserve_from_input"
}
```

**Distribution.** `app_repository.install_inference` gains a `portable` branch: download `model.onnx` + `manifest.json` instead of `.pt`. The bundle on HF/Local/Remote carries both, so an app can ship *both* runtimes and the selector picks.

**Trust win.** The `portable` path performs **no `pip install`, no `torch.load(weights_only=False)` pickle, no arbitrary `.py` import** — the model is just an ONNX graph + a fixed Rust runtime (`AUDIT.md` §4b).

**Browser.** The FastAPI server (`app_server.py`) can `StaticFiles`-mount the konfai-rs WASM bundle + `model.onnx`; execution is 100% client-side — konfai-apps distributes, konfai-rs executes.

## 5. First concrete actions (next steps)

1. **Toolchain** — installing now: Rust (`rustup`/`cargo`) + `wasm32` target; `onnx`/`onnxruntime` in the Pixi dev env. (Pixi dev already has torch 2.12 + konfai.)
2. **Run Phase 0 spike** against an IMPACT-Synth model (download via `KonfAIApp`) → GO/NO-GO report appended to the roadmap.
3. If GO → open Phase 1 (`feat/onnx-export`) as the first real, no-regret commit on this branch.

> We do **not** scaffold `konfai-rs` until Phase 0 is GO. The spike is the cheapest test of the riskiest assumption (does the IMPACT-Synth generator import into burn-onnx and run?).

## 6. Open decisions to confirm

1. **IMPACT-Synth patch geometry:** 2.5D or 3D? (Read from the model's `Prediction.yml` in Phase 0 — drives tiling.)
2. **Browser input format (Phase 4):** NIfTI upload (likely); DICOM in-browser deferred? 2D slice vs full 3D volume?
3. **Output UX:** download sCT as NIfTI, render slices in-canvas, or both?
4. **Weights/license:** are IMPACT-Synth weights redistributable as an exported `.onnx` for a public demo?
5. **First shippable surface to prioritize after the spike:** native edge binary, Slicer plugin, or browser demo? (All come from the same crate; this is about ordering Phases 3–5.)
