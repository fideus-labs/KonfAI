# KonfAI — Agent Guide

Canonical reference for humans and AI agents. Read it before changing code. User-facing detail lives in
`docs/` and `examples/`.

## 1. What KonfAI is

KonfAI is a modular, fully-configurable deep-learning framework for medical imaging (Boussot & Dillenseger, 2025 — arXiv:2508.09823). A model, its data pipeline, losses/metrics, optimizer/schedulers, augmentations, and the whole train/predict/evaluate workflow are described in **YAML** and mapped onto Python objects by a reflection engine — *without editing code*. The config is a complete, reproducible record of the experiment. KonfAI has produced top-ranking MICCAI-challenge results (SynthRAD, TrackRAD, CURVAS, PANTHER) across segmentation, registration, and synthesis.

Three pillars run through the codebase:

1. **Config-by-reflection.** `apply_config(path)` reads a callable's signature and builds its arguments from the YAML subtree it owns (`@config("Key")`), recursing into nested `@config` objects. Resolved defaults are written *back* to the file, so a run leaves a fully-resolved config on disk. **Reading a config mutates it.**
2. **Lazy, patch-based imaging.** Volumes are never loaded whole into RAM: data is read as overlapping patches (optionally streamed) and predictions are reassembled with overlap blending. **Mandatory invariant.**
3. **Declarative models.** Networks are routed `add_module` graphs — written as Python classes in `konfai/models/`, or entirely as a `.yml` via the YAML model builder.

## 2. Repository layout

| Path | Role |
|---|---|
| `konfai/` | Core package (config, data, network, metric, workflows, utils) |
| `konfai-apps/` | **Independent** package `konfai_apps` (app management, HF repos, FastAPI server) — own `pyproject.toml`, deps, and CI |
| `konfai-mcp/` | **Independent** package `konfai_mcp` (FastMCP server exposing KonfAI to LLM agents) — own `pyproject.toml`, tests, CI. Lives on the `konfai-mcp` branch (published to `origin/konfai-mcp`); do not push without explicit confirmation. |
| `apps/` | Ready-to-use model app bundles (excluded from the `konfai` wheel) |
| `examples/` | Runnable `Segmentation` / `Synthesis` workflows (assume CWD = the example dir) |
| `docs/` · `tests/` | Sphinx site · core test suite (`tests/unit`, `tests/integration`) |

**Core modules worth knowing:** `utils/config.py` (the reflection engine — read before any config change); `utils/dataset.py` (storage backends `SitkFile`/`H5File`/`OmeZarrFile`/`DicomFile` + the `Attribute` geometry sidecar); `data/data_manager.py` + `data/patching.py` (lazy patch index, DDP sharding, overlap-blended reassembly); `network/network.py` (`ModuleArgsDict`/`Network`/`ModelLoader`/`Measure` — the heart of the model system); `metric/measure.py` (`Criterion` = losses + metrics); `data/{transform,augmentation}.py`; `trainer.py`/`predictor.py`/`evaluator.py` (the pipelines); `main.py` (CLI); `utils/{errors,runtime,model_builder}.py`.

## 3. How it fits together

**Commands → config files.** KonfAI is command-driven; four CLI states map to three YAML files:

| Command | File | Root key | Purpose |
|---|---|---|---|
| `TRAIN` / `RESUME` | `Config.yml` | `Trainer:` | Model + dataset + losses + augmentations + optimizer/schedulers + training params |
| `PREDICTION` | `Prediction.yml` | `Predictor:` | Load model(s), patch/TTA/ensemble inference, output post-processing |
| `EVALUATION` | `Evaluation.yml` | `Evaluator:` | Predictions vs ground truth → per-case + aggregate metric JSON |

Each run writes a **workspace** keyed by `train_name`: `Checkpoints/`, `Statistics/` (TensorBoard + the resolved-config snapshot), `Predictions/`, `Evaluations/` (metric JSON). `Dataset/` is the *input* data directory, not a run output.

**Conventions.** Arrays are **channel-first** `[C,(Z),Y,X]`; geometry/spacing is **`(x,y,z)`** (SimpleITK). `Attribute` geometry keys are `Origin`/`Spacing`/`Direction`.

**Network graph.** `add_module(name, module, in_branch=[...], out_branch=[...], alias=...)` wires a string-keyed branch register (branch `'0'` = input; execution = insertion order). **Named module outputs are referenceable in YAML** — e.g. an `outputs_criterions` key is a module's dotted path like `UNetBlock_0:Head:Softmax` (the `:`/`.` separators are load-bearing). `out_branch:[-1]` marks a terminal/deep-supervision head; `alias` lists are positional and load-bearing for pretrained-weight remapping.

**Runtime.** Workflows run under `run_distributed_app` (`utils/runtime.py`): it builds the configured `DistributedObject`, sets the `KONFAI_*` env vars, forces `KONFAI_CONFIG_MODE='Done'`, and spawns one process per GPU (or submits to SLURM via `submitit`). Disk/log side effects are gated on `global_rank == 0`.

For the full config-key catalogue and a concrete end-to-end trace, read the `docs/` config guides and `examples/`.

## 4. Extending KonfAI

Every extension point is **"subclass a base, reference it by classpath in YAML"** — no core edits:

- **Model:** subclass `network.Network`, build the graph in `__init__` via `add_module`. Reference `classpath: module.MyNet`, a local `Model:MyNet`, or a `.yml`.
- **Loss / metric:** subclass `metric.measure.Criterion`; `forward` returns a `Tensor` (loss) or a `(value, dict)` tuple (metric — consumers `isinstance`-branch). Attach under `outputs_criterions`/`metrics` to a **named module output**. Optional-dep criteria import lazily via `_require_optional(...)` and raise an actionable `MeasureError` — never a bare top-level import.
- **Transform:** subclass `data.transform.Transform`; implement `__call__` **and** `transform_shape()` (must predict the output spatial shape *exactly* — patch planning depends on it). Pair `inverse()` if `apply_inverse`.
- **Augmentation:** subclass `data.augmentation.DataAugmentation`; `_state_init` (sample params per case index) + `_compute` (apply lazily). Only `Mask`/`Permute` may change shape.
- **Imaging format:** add a `Dataset.AbstractFile` backend, dispatch it in `File.__enter__`, register aliases in `SUPPORTED_EXTENSIONS`; import-guard the heavy lib.

**Classpaths:** a bare name (e.g. `Dice`) resolves inside that kind's package; `module:Class` imports *any* module — a local file (`Loss:MyWrapper`) or an installed library (`monai.losses:DiceLoss`, `torch:nn:L1Loss`).

**YAML model builder** (`utils/model_builder.py`): builds a `Network` from a `.yml`, **safe by construction** (node types must come from two curated registries — no `eval`/import injection). It *complements* `models/` today and can replace the feed-forward subset once the registry grows (`UNet`/`NestedUNet`/`ResNet` are migratable; custom-`forward` models like DDPM/DiffusionGAN/ConvNeXt are not).

## 5. Apps (`konfai-apps`)

A separate package layered on KonfAI's **public** API (core never imports it). An "app" bundles a config + custom `.py` + `.pt` weights, resolved from a Local dir, a HuggingFace repo, or a Remote server; the `apps/*` bundles are thin CLI wrappers.

> ⚠️ **Trust model.** Resolving an app **copies and imports its `.py` files** → it **runs arbitrary code**. It also **pip-installs its `requirements.txt` by default** (only missing/mismatched packages; core packages like `torch`/`konfai` are never touched; opt out with `KONFAI_APPS_INSTALL_REQUIREMENTS=0`). **Only resolve apps from sources you trust.**

## 5b. MCP server (`konfai-mcp`)

A third independent package (depends only on KonfAI's public API) exposing a **FastMCP** server so an LLM agent can inspect a dataset → author YAML → run train/predict/evaluate → monitor jobs → compare runs → iterate. Lives on the `konfai-mcp` branch (published to `origin/konfai-mcp`; do not push without explicit confirmation). Jobs run in a **`spawn`** subprocess (training may init CUDA); `validate_config_semantics` and `run_component_smoke_test` run in a **spawn subprocess** (never in the server process), are side-effect-free (config bytes snapshotted/restored), and re-import edited workspace code; discovery is via `list_components` / `describe_extension_points` / `describe_config_schema` / `check_external_dependency`. Tests: `pip install -e ./konfai-mcp` then `python -m pytest konfai-mcp/tests` (the segmentation E2E needs the imaging extra).

**Working on the MCP server — how to validate a change:**

- **Synthetic fixtures:** `pixi run --environment dev python audit/make_fixtures.py` builds a segmentation
  dataset, a registration pair with a known translation, a synthesis pair, a 3-level OME-Zarr store, and
  corrupted/unsupported inputs under `audit/fixtures/` (procedural, no patient data). Reuse these, do not
  invent ad-hoc data in `/tmp`.
- **Drive it black-box first, not by tool name.** Formulate a real objective ("segment these CT volumes"),
  then exercise the loop through a `fastmcp.Client` exactly as `test_mcp_server_segmentation_pipeline.py`
  does. A new tool is not "done" because it returns without an exception.
- **Verify outputs, not return codes.** After `run_*` + `wait_for_job`, assert the job `status=="done"`
  (never trust a green `validate_*` alone — its default level `instantiate` runs no train step; only
  `level='train_step'` runs one forward+backward), then open the
  produced files: `read_session_file` the config, check `Predictions/<name>/Dataset` exists, and read the
  `Metric_*.json` via `get_run_metrics`. Confirm the numbers correspond to the requested task.
- **Validating a new tool:** (1) its `next_actions` must be registered tool names — the anti-drift test in
  `test_mcp_server_tool_index.py` enforces this for job payloads only, so if the tool emits `next_actions`
  in its own payload, assert it in the tool's pytest; (2) if it takes a workspace path, route it through
  `resolve_workspace_relative_path` (jail); (3) if it imports/executes app or workspace code, run it in the
  spawn subprocess (`run_api_in_subprocess`) and gate it behind `allow_untrusted_code` where applicable;
  (4) document per-parameter meaning via `Annotated[..., Field(description=...)]`, not only prose; (5) add a
  pytest that inspects the output.
- **Adding a workflow kind touches ~7 registries** (WORKFLOWS, WORKFLOW_CONFIG_FILES/ROOT_KEYS, runner
  command map, capabilities `_WORKFLOW_ROOTS`, `Job.kind` Literal + retry map, GUIDE). Prefer one descriptor
  table consumed everywhere over editing each.
- **Safety invariants to preserve:** validation/smoke-tests never execute in the server process; only
  `read/write_session_file` are path-jailed (dataset tools read arbitrary host paths by design — keep it
  that way only for the trusted-local deployment, and never widen writes). `cancel_job` now reaps the whole
  process group — the job runs `os.setsid()` and cancel sends the signal via `os.killpg`, so `mp.spawn` DDP
  grandchildren are killed with the middle process (regression test:
  `test_cancel_reaps_the_whole_process_group_including_grandchildren`).
- **Regenerate derived docs:** after changing a tool's description run
  `python konfai-mcp/scripts/generate_tool_reference.py` (the committed skill reference is generated).

## 6. Running things

```bash
pixi run check                                                    # lint + format-check + test (run before finalising)
pixi run test                                                     # core unit + integration (tests/)
pixi run --environment dev typecheck                              # mypy konfai
pip install -e ./konfai-apps && pixi run --environment dev python -m pytest konfai-apps/tests   # apps suite (separate)
```

The Pixi `dev` env carries the imaging extras; a bare `pip install .[dev]` does not. `pixi run test` does **not** run `konfai-apps/tests` — install that package first (it pulls its own runtime deps), exactly as its CI does. Install runtime extras with `pip install konfai[<extra>]` (`itk`, `hdf5`, `dicom`, `omezarr`, `imaging`, `tensorboard`, `lpips`, `ssim`, `fid`, `cluster`, …).

## 7. Invariants — do NOT break

- **Never load a full volume into RAM.** Use lazy/patch/streaming access (`can_stream_patch`, `read_data_slice`).
- **Channel-first `[C,(Z),Y,X]`; spacing `(x,y,z)`.** `Attribute` stringifies every value and reparses geometry via `np.fromstring` — only flat scalars / 1-D arrays round-trip. Read via `__getitem__`/`get_np_array`.
- **`KONFAI_config_file` + `KONFAI_CONFIG_MODE` must be set before any `Config()`** (tests must `monkeypatch.setenv` both); workflows require `KONFAI_CONFIG_MODE='Done'`. Reading a config rewrites it on disk.
- **Patch ordering** must match between read (`disassemble`) and write (`Accumulator`); for PREDICTION/EVALUATION all patches of a case stay on the same DDP rank.
- **`outputs_criterions` keys equal a module's dotted path**; the `:`/`.` separators are load-bearing.
- **`state_dict` load/save does not recurse into nested `Network`s** (each owns its optimizer/state); alias lists are positional.
- **The YAML model builder is the trusted/untrusted boundary** — only registry types; module names contain no `.`.
- **`konfai-apps` is a separate package**; `apps/` is excluded from the `konfai` wheel.

## 8. Conventions & rules

- **Code:** line length 120 (Ruff); type annotations on new public functions; Apache-2.0 SPDX header on every new source file; prefer `pathlib.Path`; use the error classes in `utils/errors.py` (do not invent exceptions); import-guard heavy optional deps (`SimpleITK`/`h5py`/`pydicom`/`zarr`) — fail at point-of-use with an install hint, not at import.
- **Commits:** Conventional Commits (`cz check`): `type(scope): subject`, imperative, < 72 chars. No AI-agent branding (`claude`/`codex`/"generated by/with") and no AI co-author trailers.
- **For agents:** read before editing; keep diffs small (one logical change per PR, no unrelated reformats); run `pixi run check` (and the apps suite if you touched `konfai-apps`) before finalising; no new runtime dependency without an explicit request + a matching `pyproject.toml` update in the same commit; update docs and `tests/unit/test_config.py` when changing config binding; do not skip pre-commit with `--no-verify`.
