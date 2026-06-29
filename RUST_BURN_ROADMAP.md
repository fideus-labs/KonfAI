# KonfAI × Rust / Burn — Strategy & Roadmap

> Status: **RFC / decision document.** No code in the core is changed by this proposal.
> Companion to [`AUDIT.md`](AUDIT.md) and [`AGENTS.md`](AGENTS.md). Read those first for the architecture.

This document answers a single strategic question raised internally (Matt → Burn, `https://burn.dev`):

**Should KonfAI move toward Rust / Burn — a full version, a Rust layer, or a Burn integration for some parts — or stay Python/PyTorch?**

The analysis below is grounded in the *actual* KonfAI code (file:line references) and in dated facts about Burn (mid-2026), not in intuition. It is deliberately critical.

---

## TL;DR

**Keep the core (training + research) in Python/PyTorch. Burn is not a PyTorch replacement today. But Burn's portability (WebGPU/WASM, single self-contained binary, one codebase across CPU/CUDA/Metal/browser) maps onto a real KonfAI gap: *deployment of inference* — and specifically the `konfai-apps` packaged-model layer (§4.1), where today running one model requires a full Python + torch + imaging stack and the execution of arbitrary downloaded code. The correct entry point is neither a Burn backend inside the network core, nor a Rust preprocessing layer, but a separate `konfai-rs` crate dedicated to portable inference of the feed-forward subset of already-trained models — gated behind one decisive spike: does a KonfAI 3D U-Net survive `torch → ONNX → burn-onnx` import?**

---

## 1. What the code actually says (not intuition)

Three findings reframe the entire question.

### 1.1 Preprocessing is already torch, not slow Python

| Module | `torch.` | `np.` | `sitk.` |
|---|---|---|---|
| `konfai/data/transform.py` | 147 | 10 | 12 |
| `konfai/data/augmentation.py` | 142 | 8 | — |
| `konfai/data/patching.py` | 29 | 4 | — |

`Resample` already runs `F.interpolate` (trilinear) **on GPU** when the tensor is on device (`konfai/data/transform.py:386-399`). The math is already delegated to torch's C++/CUDA kernels.

➡️ **The "Rust for fast preprocessing" thesis is dead on arrival.** The only Python overhead left is per-patch *orchestration* (the `DataLoader.__getitem__` loop), which is solved with `num_workers` / streaming — not by rewriting tensor math in Rust. The real bottleneck of a medical run is the **GPU forward/backward**, not the CPU.

### 1.2 I/O is already delegated to mature C/C++ libraries

`SimpleITK`, `h5py`, `zarr`, `pydicom` do the heavy lifting. The only genuinely pure-Python hot spots:

- DICOM series sort/read — non-vectorised Python loops, redundant series re-discovery on sliced reads (`konfai/utils/dicom.py:163-179`, `:310-369`, `:397-427`).
- `Attribute` stringify/reparse per access (`konfai/utils/dataset.py:68-95`).

These are micro-optimisations, not a transformation. A Rust DICOM crate would duplicate GDCM/pydicom with **worse** coverage.

### 1.3 The model/training core is 100% torch-coupled — not "swappable"

- `ModuleArgsDict(torch.nn.Module, ABC)` at `konfai/network/network.py:478` — the routed graph **is** an `nn.Module`. Routing metadata is a layer *on top of* torch; it does **not** abstract the backend.
- `named_forward` is built on `torch.utils.checkpoint`, `isinstance(module, torch.nn.Module)` dispatch, manual `.to()` device moves (`network.py:663-722`).
- KonfAI's distinctive features — `outputs_criterions` attached to **named** module outputs, `alias`-based pretrained-weight remapping, deep supervision — depend on torch `state_dict`/`named_parameters` introspection (`network.py:860-944`).
- `GradScaler`, `autocast`, DDP, `torch.optim`: all torch (`network.py:1104`, `:1310-1328`; `trainer.py:307-312`).

➡️ **Putting a Burn backend "under" the `Network` is not an abstraction, it's a rewrite** that would have to reinvent named routing + alias remapping + checkpointing.

### 1.4 The MCP/agent layer is backend-agnostic

`konfai-mcp/konfai_mcp/server.py` only ever sees YAML strings, subprocess jobs, and JSON metrics. The reflection engine (`apply_config`, `inspect.signature`) runs *inside the runtime*, after the job subprocess starts — not in the agent layer.

➡️ **Burn neither helps nor hurts the agentic loop.** It would just need a CLI + a config surface. The MCP loop is neutral with respect to the compute backend.

### 1.5 There is no ONNX/TorchScript export of KonfAI's own models today

The only `torch.jit.load` usage loads *third-party* perceptual/IMPACT loss models (`konfai/metric/measure.py`). Any "deploy via Burn" path must **build the export first**.

---

## 2. Burn reality check (dated, mid-2026)

| Dimension | Reality | Verdict |
|---|---|---|
| **Overall maturity** | v0.21.0 (2026-05-07), **still pre-1.0** after ~3 years. API not stabilised; churn between minors. Backing is real though: Tracel AI ($3M seed), ~15.5k GitHub stars, 3 substantive releases (0.19→0.21) in ~7 months targeting training/distributed gaps. | ⚠️ Young but well-funded, steep trajectory |
| **Backends** | CUDA, ROCm, Metal, Vulkan, **WebGPU**, LibTorch, Cpu(CubeCL), **Flex** (replaces ndarray; WASM/no_std). Autodiff + Fusion + Remote = *composable* backend decorators. **CubeCL** = "write the kernel once, run everywhere". | ✅ **Excellent — the real asset** |
| **Training** | Mature autodiff, `LearnerBuilder` (checkpointing, early-stopping, grad-accum, multi-device). Optimizers SGD/Adam/AdamW/RmsProp/Adan; schedulers cosine/Noam/step/linear. | ⚠️ OK, but **distributed = least-settled area** (0.19+), no LAMB/OneCycle/ReduceLROnPlateau, thin third-party ecosystem |
| **ONNX import** | New `burn-onnx` crate (~May 2026; old `burn-import` ONNX path now *legacy*). **Build-time codegen**, not a runtime loader. **Opset 16+ required.** Op coverage is now **~75-80% (~230/300)** — GridSample, Einsum, InstanceNorm, LRN, **3D Conv/ConvTranspose**, Resize are all marked supported as of mid-2026. **So missing ops are no longer the main blocker.** The real risk is the gap between *op-on-paper* and *model-actually-imports*: mainstream CNNs (ResNet-50, MobileNet) still fail to import (burn-onnx issue #18). Dynamic shapes & control-flow (If/Loop) remain structural problems for a static-codegen importer. | ⚠️ **Risk #1 for 3D medical deployment — but it's import-pipeline reliability, not op coverage** |
| **PyTorch weight import** | `PyTorchFileRecorder`/safetensors = **weights only**; you must have **hand-written the model in Rust** already. | ⚠️ Manual labour |
| **WebGPU / WASM** | **Real and first-class.** Official in-browser demos (`mnist-inference-web`, `image-classification-web`). WGPU-in-WASM where the browser ships WebGPU; otherwise Flex CPU. "Same code across backends" portability genuinely holds. | ✅ **The killer feature** |
| **Complex models** | Transformer primitives (MHA, encoder/decoder, RoPE, GQA) — but *low-level building blocks*, not an HF catalog. **SAM**: a single-dev hobby port (`karelnagel/sam-rs`), unmaintained; **no SAM2/SAM3**. **No first-class LoRA/PEFT** (hand-rolled). Thin official zoo (ResNet, RoBERTa, YOLOX). **Zero medical models, no nnU-Net, no MONAI.** Also: Burn's `grid_sample` is **2D-only** (`grid_sample_2d`, 0.20.0) → KonfAI's **3D VoxelMorph/registration cannot run on Burn** at all. | ⛔ **Exactly where medical research lives, Burn is empty** |

### The three optimistic claims, refuted

1. *"Burn can replace PyTorch for KonfAI training"* → **REFUTED.** Pre-1.0, unstable distributed, no medical/PEFT ecosystem, no pretrained weights. Researchers' time is the scarce resource; Burn would burn it.
2. *"A train-in-PyTorch / deploy-in-Burn pipeline is reliable today"* → **REFUTED for "reliable enough to ship today".** The path exists and op coverage is now ~75-80% (3D conv, GridSample, Resize supported), so the blocker has **shifted from missing ops to import-pipeline reliability** — mainstream CNNs (ResNet-50, MobileNet) still fail to import (burn-onnx #18). Note too: "data on GPU end-to-end" is only literally true for the model *interior* — inputs still marshal through WASM linear memory. **This is exactly what the spike must test.**
3. *"Rust would speed up preprocessing/IO beyond numpy+SITK+zarr"* → **MIXED, leaning refuted.** Already C/C++/torch — marginal gains, high friction. One genuine measured exception: **OME-Zarr decompression** via the Rust `zarrs`/`zarrs-py` crate (~3× on zarr reads), but contingent on actually using OME-Zarr (KonfAI's default + all examples use `.mha` → no win). And free-threaded Python (3.13/3.14) + a multithreaded DataLoader is a lower-effort path to the same GIL relief.

---

## 3. Options evaluated

| Option | Effort | Real benefit | Risk / tech debt | Impact on current users | Verdict |
|---|---|---|---|---|---|
| **A — Do nothing** | 0 | Keeps velocity | None | None | ✅ **For the core** |
| **B — Rust for some bricks** (preproc/IO/patch/metrics) | Medium-high | **Low** (already torch/C++) | FFI, two languages, complex build | Install friction | ⛔ Poor ROI (one narrow niche: DICOM decode, and even that competes with C libs) |
| **C — Optional Burn backend inside `Network`** | **Very high** | Low | **Huge**: reinvent named routing + alias + checkpointing; two paths to keep in sync | Breakage risk | ⛔ A rewrite in disguise |
| **D — Separate `konfai-rs` sub-project (inference)** | Medium (bounded) | **High** (portability, deployment) | Contained (isolated crate, like `konfai-mcp`) | **None** (does not touch the core) | ✅ **The right target** |
| **E — Ambitious rewrite** | Massive | Negative short/medium term | Disperses the project; Burn pre-1.0 | Regressions | ⛔ Premature |

**Recommendation: A for the core + D as an isolated strategic bet.** B, C, E are traps.

---

## 4. Target architecture

```
┌─────────────────────── PYTHON / PyTorch (unchanged) ───────────────────────┐
│  Research · training · fine-tuning · challenges · MCP/agents · YAML configs │
│  Dataset/transform/patching/augmentation (already torch, GPU-capable)       │
│                                                                             │
│   new, pure Python, useful on its own:  konfai export → ONNX (opset 16+)     │
└────────────────────────────────┬────────────────────────────────────────────┘
                                  │  trained model → .onnx (clean boundary)
                                  ▼
┌──────────────── konfai-rs (separate Rust crate, excluded from wheel) ───────┐
│  burn-onnx (build-time codegen) → Burn Network → chosen backend:            │
│     Flex (CPU/WASM) · WGPU (browser/GPU) · CUDA/Metal (desktop/edge)        │
│  + ports the portable KonfAI logic: sliding-window patching + overlap blend │
│  Targets: CLI binary · browser WASM module · Slicer plugin · edge server    │
└──────────────────────────────────────────────────────────────────────────────┘
```

### What stays Python / what could move / what should use Burn

- **Stays Python (do not touch):** declarative configs + reflection, datasets, transforms, training-side patching, models, losses, training metrics, training, fine-tuning, train CLI, MCP/agents, OME-Zarr/DICOM/NIfTI readers, challenge workflows.
- **Could move to Rust/Burn:** *only* inference of frozen models + patch reassembly + post-processing, **plus** a new ONNX export (Python-side, but serves Burn).
- **Does NOT use Burn:** training, KonfAI autograd, perceptual/IMPACT losses (TorchScript), anything touching SAM/LoRA.
- **Not worth rewriting:** Rust preprocessing, a Burn backend inside `Network`, the config reflection engine.

### 4.1 The pulling use case: `konfai-apps` portability

The strongest concrete target is **`konfai-apps`** — the packaged-model / deployment layer, not the research core. This is where Burn's portability becomes a product advantage rather than a science project.

Today a packaged app is `app.json` + `Prediction.yml` + custom `.py` + `.pt` weights, resolved from HuggingFace / local / remote, and run by **copying the app's `.py` into the working dir, importing it, and `pip install`-ing its `requirements.txt`** — i.e. a downloaded app **runs arbitrary code and arbitrary dependency installs** (see `AUDIT.md` §4b trust model), and needs a full Python + PyTorch + SimpleITK/h5py/zarr/pydicom environment to run *one inference*.

A `konfai-rs` inference runtime would let a class of apps ship as a **self-contained artifact** instead:

| `konfai-apps` today (Python) | `konfai-apps` + `konfai-rs` (portable) |
|---|---|
| Python + torch + imaging deps must be installed | Single static binary, or a WASM module |
| Downloads & imports arbitrary `.py`, `pip install`s `requirements.txt` | **No arbitrary code execution** — model is the ONNX graph + fixed Rust runtime |
| Server-side / workstation only | Browser, Slicer plugin, edge device, offline kiosk |
| GPU optional but heavy stack | WGPU on whatever GPU is present; CPU fallback via Flex |

This reframes the whole effort: the deliverable is **a portable inference target for the *feed-forward subset* of packaged apps** (segmentation/synthesis U-Nets — exactly the migratable subset in `AGENTS.md` §7). Apps whose `.py` carries genuine custom Python logic (diffusion sampling, registration grid math, custom training loops) stay on the Python runtime.

**Delivery shape (decided):** rather than modifying `konfai-apps`, this ships as **its own separate package** — **`konfai-rs`**, a sibling to `konfai-apps`/`konfai-mcp` with its own build, tests/CI, and branch, depending only on the public API + an exported `.onnx`. It is a **portable inference *engine*, not a web project**: one Rust/Burn crate that compiles to a **native binary** (CUDA/Metal/CPU — edge/clinic/Slicer/server) *and* to **WASM + WebGPU** for the browser, from the same code. First validated target is **native** (fastest to iterate); the **browser is the showcase target**, layered on top. First model: the **IMPACT-Synth** sCT generator (`VBoussot/ImpactSynth`). See [`RUST_BURN_IMPLEMENTATION_PLAN.md`](RUST_BURN_IMPLEMENTATION_PLAN.md).

> Caveat that keeps this honest: only apps whose model survives the ONNX → `burn-onnx` spike qualify, and apps with custom-`.py` inference logic do **not** become portable just by swapping the tensor backend. So this is a *subset* portability story, not "all apps become a binary".

---

## 5. PR roadmap

### PR #1 (smart, minimal) — the decision + the decisive spike
- **Branch:** `docs/rust-burn-strategy`
- **Creates:** this `RUST_BURN_ROADMAP.md` (sibling to `AUDIT.md` for agent discoverability) + a spike protocol note.
- **The spike (~1-2 days):** export the example U-Net (`examples/Segmentation`) to ONNX → `burn-onnx` codegen → run 1 patch → compare against the torch reference (tol 1e-3) → benchmark latency + binary size. **This is the go/no-go for the entire thesis.** Note the mid-2026 failure mode is *import-pipeline bugs* (does the generated Rust compile and run?), **not** missing ops — see burn-onnx issue #18 where mainstream CNNs still fail to import despite their ops being "supported".

### PR #2 (pure Python, value independent of Burn) — ONNX export
- **Branch:** `feat/onnx-export`
- **Modifies:** `konfai/predictor.py` or a new `EXPORT` CLI state in `konfai/main.py`.
- **Adds:** `konfai/export/onnx.py` (export a frozen `Network`, opset ≥16, handle named outputs).
- **Tests:** `tests/unit/test_onnx_export.py` — onnxruntime-vs-torch parity on the example U-Net.
- This PR has value **even if Burn is dropped** (ONNX = general interop). The best "no-regret" first step.

### PR #3 (large but bounded, *only if the spike is GO*) — `konfai-rs`
- **Branch:** `feat/konfai-rs-inference-poc`
- **Creates an isolated crate (excluded from the wheel, like `konfai-mcp`):**
  - `konfai-rs/Cargo.toml`, `konfai-rs/build.rs` (burn-onnx `ModelGen`), `konfai-rs/src/{main.rs,patch.rs,io.rs}` (CLI + sliding-window + overlap reassembly ported), `konfai-rs/README.md`
  - `konfai-rs/benches/inference.rs` (criterion: torch CPU/CUDA vs Burn Flex/WGPU)
  - `konfai-rs/tests/parity.rs` (Burn output vs ONNX reference, fixed tolerance)
  - **Stretch:** `konfai-rs/web/` — WASM/WebGPU in-browser demo of the model.

### Benchmarks to run
- Single-patch latency: torch (CPU/CUDA) vs Burn (Flex/WGPU).
- **Binary size + cold-start** (the deployment metric that matters).
- Op coverage: how much of the 3D U-Net imports without manual patching.
- Browser: does the model run *at all* via WGPU+WASM.

### GO / NO-GO criteria
- **GO if:** a KonfAI 3D U-Net survives `torch → ONNX → burn-onnx` **and** compiles **and** output ≈ torch (1e-3) **and** WGPU runs **and** the binary is meaningfully simpler to deploy than Python+torch+imaging.
- **NO-GO / PARK if:** burn-onnx stumbles on conv3d / trilinear resample / dynamic shapes without manual patching; or numerical divergence; or maintaining two inference paths costs more than the deployment pain it removes. **Re-test at Burn 1.0.**

---

## 6. Principal risks

1. **burn-onnx can't digest 3D segmentation** (conv3d/trilinear/dynamic) → the spike reveals this fast and cheap. *The right risk to take early.*
2. **Project dispersion** — two languages, two implementations to keep in sync. Mitigated by *isolating* in an out-of-wheel crate, never in the core.
3. **Pre-1.0 churn** — Burn API moves between versions. Mitigated by staying on inference (small surface).
4. **Effort without a user** — building `konfai-rs` nobody uses. Mitigated by requiring a *pulling* use case (Slicer? browser? a specific Fideus partner?) before PR #3.
5. **ONNX as a fragile boundary** — KonfAI named outputs ↔ ONNX. PR #2 handles this cleanly.

---

## 7. Questions to align with Matt / Fideus

1. **Pulling use case — likely `konfai-apps` portability (§4.1).** Confirm: is the goal to ship packaged apps as a self-contained binary/WASM (no Python, no arbitrary-code trust issue), and for which delivery surface first — Slicer plugin? browser demo? edge/offline clinic? Which specific app(s) should be the first portable target?
2. **Inference only — agreed?** Or does he actually picture Burn *training* (then the gaps must be shown: PEFT, distributed, medical zoo)?
3. **"Data on GPU end-to-end":** preprocessing included, or only network→viz? (KonfAI preprocessing is already torch/GPU — so largely already achieved on the Python side.)
4. **Which models?** If it's SAM/SAM3/LoRA, Burn is a "no" today — better to know up front.
5. **Who maintains the Rust?** Bus factor: an orphaned Rust crate is debt. Long-term commitment?
6. **Tolerance for pre-1.0?** Do we accept Burn breaking changes every ~3 months on a clinical deployment path?

---

## 8. Bottom line

- **Good idea?** For **portable inference**: yes. For training/preprocessing/core: no.
- **Which parts exactly?** ONNX export (Python) + a `konfai-rs` Burn inference crate (WGPU/WASM/binary).
- **Why not the rest?** Preprocessing is already torch; the core is 100% torch-coupled; Burn is pre-1.0 with no medical/PEFT ecosystem.
- **Best entry point:** the **ONNX → burn-onnx spike on the 3D U-Net** — decisive and cheap.
- **First clean PR:** this roadmap + the spike; then **ONNX export** (no-regret value).
- **Roadmap:** doc+spike → ONNX export → (if GO) `konfai-rs` inference crate → WASM/Slicer demo.
- **Strategic upside for Fideus?** Potentially real and now concrete: making **`konfai-apps`** packaged models ship as a portable binary/WASM (§4.1) — no Python stack, no arbitrary-code trust issue, runnable in browser/Slicer/edge. PyTorch can't easily match this. **Conditional on the spike result and on apps whose models are feed-forward (custom-`.py` apps stay on the Python runtime).**
