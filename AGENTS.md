# KonfAI Agent Guide

Canonical reference for humans and AI agents. Read it before changing code. User-facing detail lives in
`docs/` and `examples/`.

## 1. What KonfAI is

KonfAI is a modular, fully-configurable deep-learning framework for medical imaging (Boussot & Dillenseger, 2025, arXiv:2508.09823). A model, its data pipeline, losses/metrics, optimizer/schedulers, augmentations, and the whole train/predict/evaluate workflow are described in **YAML** and mapped onto Python objects by a reflection engine, *without editing code*. The same engine drives a fifth workflow, **dataset preparation** (`TRANSFORM`): read a dataset, apply a chain, write it back out. The config is a complete, reproducible record of the experiment. KonfAI has produced top-ranking MICCAI-challenge results (SynthRAD, TrackRAD, CURVAS, PANTHER) across segmentation, registration, and synthesis.

Three pillars run through the codebase:

1. **Config-by-reflection.** `apply_config(path)` reads a callable's signature and builds its arguments from the YAML subtree it owns (`@config("Key")`), recursing into nested `@config` objects. Resolved defaults are written *back* to the file, so a run leaves a fully-resolved config on disk. **Reading a config mutates it.** Each workflow builder reads its file inside `strict_config(root)`: a key nothing bound reads is reported by path (TRANSFORM refuses, the other three warn), so everything a workflow reads from its file must be bound at construction, not lazily.
2. **Lazy, patch-based imaging.** A volume is never loaded whole into RAM on a route that can stream: data is read as overlapping patches (optionally streamed) and predictions are reassembled with overlap blending. **Mandatory invariant.** The one declared exception is TRANSFORM's whole-volume route, taken only when a stage cannot serve a region: the plan names it `WHOLE-VOLUME`, sizes it against the memory budget (`Transform.working_multiple`) and refuses the case when it does not fit. Never an implicit `read_data()`.
3. **Declarative models.** Networks are routed `add_module` graphs, written as Python classes in `konfai/models/`, or entirely as a `.yml` via the YAML model builder.

## 2. Repository layout

| Path | Role |
|---|---|
| `konfai/` | Core package (config, data, network, metric, workflows, utils) |
| `konfai-apps/` | **Independent** package `konfai_apps` (app management, HF repos, FastAPI server) with its own `pyproject.toml`, deps, and CI |
| `konfai-mcp/` | **Independent** package `konfai_mcp` (FastMCP server exposing KonfAI to LLM agents) with its own `pyproject.toml`, tests, CI. On `main` since v1.6.0; published to PyPI by the release workflow. |
| `studio/` | **Independent** package `konfai_studio` (FastAPI BFF + built React front: a chatbot web UI over `konfai-mcp`). Own `pyproject.toml`; the front (`konfai_studio/web/*`) is a CI `npm` build, not in git; wheel-only. |
| `apps/` | Ready-to-use model app bundles (excluded from the `konfai` wheel) |
| `examples/` | Runnable `Segmentation` / `Synthesis` / `Registration` workflows (assume CWD = the example dir) |
| `docs/` · `tests/` | Sphinx site · core test suite (`tests/unit`, `tests/integration`) |

`konfai/models/` has two halves: **`models/python/<kind>/`** (Python `Network` subclasses) and
**`models/yaml/`** (the shipped declarative catalog, 14 models). `models/python` has no `__init__.py` on
purpose: it is a PEP 420 namespace package, and the wheel ships it via `include = ["konfai", "konfai.*"]`
with `namespaces=true`; the catalog `*.yml` ship via `package-data`. Changing either breaks the wheel
silently, so verify with a clean **non-editable** install, not an editable one.

**Core modules worth knowing:** `utils/config.py` (the reflection engine: read before any config change); `utils/dataset.py` (storage backends `SitkFile`/`H5File`/`OmeZarrFile`/`DicomFile`/`ItkTransformFile` + the `Attribute` geometry sidecar + `DataStream` region writes); `data/data_manager.py` + `data/patching.py` (lazy patch index, DDP sharding, overlap-blended reassembly, `SlabRegionStream`) + `data/materialize.py` (`CaseMaterializer`: the TRANSFORM materialization engine over the manager's plan/replay API); `data/geometry.py` (grids/boxes/affine maps in world coordinates: pure numpy, crosses the `mp.spawn` pickle boundary) + `data/sampling.py` (coordinate producers + the torch gather; no SimpleITK); `network/network.py` (`ModuleArgsDict`/`Network`/`ModelLoader`/`Measure`: the heart of the model system); `metric/measure.py` (`Criterion` = losses + metrics); `data/{transform,augmentation}.py`; `data/reduction.py` (`Reduction`/`Mean`/`Median`/`Std`/`Vote`/`Concat`: **moved out of `predictor.py`, which re-exports `Mean`/`Median`/`Concat` so `konfai.predictor.Median` keeps resolving for published configs**) + `data/case_reduction.py` (the N-cases-to-one engine); `trainer.py`/`predictor.py`/`evaluator.py`/`transformer.py` (the pipelines; `transformer.py` is the dataset-preparation one); `api.py` (the workflows as Python callables); `export.py` (ONNX export, extra `export`); `main.py` (CLI); `utils/{errors,runtime,model_builder}.py`.

## 3. How it fits together

**Commands → config files.** KonfAI is command-driven; five CLI states map to four YAML files:

| Command | File | Root key | Purpose |
|---|---|---|---|
| `TRAIN` / `RESUME` | `Config.yml` | `Trainer:` | Model + dataset + losses + augmentations + optimizer/schedulers + training params |
| `PREDICTION` | `Prediction.yml` | `Predictor:` | Load model(s), patch/TTA/ensemble inference, output post-processing |
| `EVALUATION` | `Evaluation.yml` | `Evaluator:` | Predictions vs ground truth → per-case + aggregate metric JSON |
| `TRANSFORM` | `Transform.yml` | `Transformer:` | **Dataset preparation.** Read a dataset, run a chain per case, `Write` the result. Plan first; an existing output is skipped, so a run resumes |

Each run writes a **workspace** keyed by `train_name` (`name` for TRANSFORM): `Checkpoints/`, `Statistics/` (TensorBoard + the resolved-config snapshot), `Predictions/`, `Evaluations/` (metric JSON), `Transforms/` (per-rank logs opening with the plan, the config copy, `outputs.json`, but **not the transformed data**, which lands wherever each `Write:` points, in the user's own tree). `Dataset/` is the *input* data directory, not a run output.

**Python API.** The workflows are also callables: `konfai.transform` / `plan_transform` / `evaluate` / `predict` / `train` (`api.py`, lazily re-exported from `konfai`). Same engine, two spellings: a chain is a list of live stage objects or the equivalent dict tree. The contract: a designed refusal raises `KonfAIError` (only the CLI catches), results come back structured (`TransformResult`/`EvaluationResult`), the `KONFAI_*` environment is restored around every call, one workflow per process (lock), and **a caller's config FILE is copied to scratch, so the write-back never lands on it**.

**Conventions.** Arrays are **channel-first** `[C,(Z),Y,X]`; geometry/spacing is **`(x,y,z)`** (SimpleITK). `Attribute` geometry keys are `Origin`/`Spacing`/`Direction`. In `Resample`, a `spacing`/`shape` value **≤ 0 is the keep-this-axis sentinel** (that axis keeps the source's own density/extent).

**Network graph.** `add_module(name, module, in_branch=[...], out_branch=[...], alias=...)` wires a string-keyed branch register (branch `'0'` = input; execution = insertion order). **Named module outputs are referenceable in YAML**: an `outputs_criterions` key is a module's dotted path like `UNetBlock_0:Head:Softmax` (the `:`/`.` separators are load-bearing). `out_branch:[-1]` marks a terminal/deep-supervision head; `alias` lists are positional and load-bearing for pretrained-weight remapping.

**Runtime.** Workflows run under `run_distributed_app` (`utils/runtime.py`): it builds the configured `DistributedObject`, sets the `KONFAI_*` env vars, forces `KONFAI_CONFIG_MODE='Done'`, and spawns one process per GPU (or submits to SLURM via `submitit`). Disk/log side effects are gated on `global_rank == 0`.

For the full config-key catalogue and a concrete end-to-end trace, read the `docs/` config guides and `examples/`.

## 4. Extending KonfAI

Every extension point is **"subclass a base, reference it by classpath in YAML"**, with no core edits:

- **Model:** subclass `network.Network`, build the graph in `__init__` via `add_module`. Reference `classpath: module.MyNet`, a local `Model:MyNet`, a `.yml`, or `default|<Name>.yml` for the shipped catalog.
- **Pretrained weights:** `utils/pretrained.py:transfer_weights_by_execution_order` pairs weighted leaves in forward-execution order (no key map). It fills **every** target tensor or raises: a tensor held by a parent module (`torch.nn.MultiheadAttention` owns `in_proj_weight` beside its `out_proj` child) or by a submodule the forward skips cannot be paired. Unreached *source* branches (nnU-Net deep-supervision heads) are ignored on purpose.
- **Loss / metric:** subclass `metric.measure.Criterion`; `forward` returns a `Tensor` (loss) or a `(value, dict)` tuple (metric; consumers `isinstance`-branch). Attach under `outputs_criterions`/`metrics` to a **named module output**. Optional-dep criteria import lazily via `_require_optional(...)` and raise an actionable `MeasureError`, never a bare top-level import.
- **Transform:** subclass `data.transform.Transform`; implement `__call__` **and** `transform_shape()` (must predict the output spatial shape *exactly*, because patch planning depends on it). Declare `working_multiple` (class attribute) if the whole-volume call allocates beyond its input and output, in volumes-worth of the case (`Resample` 3.0 for its sampling grid, `Gradient` 8.0): the plan sizes the whole-volume fallback with it. Declare `patch_locality()` (a `LocalityKind`: `POINTWISE`/`HALO`/`ORIENTATION`/`CROP`/`GLOBAL_STAT`/`REGRID`/`SLAB`/`WHOLE_VOLUME`) or the base default makes it `WHOLE_VOLUME`; a `WHOLE_VOLUME` that is a property of the *configuration* rather than of the stage must carry `reason=`, which the plan prints; without it the reader has nothing to change. A per-voxel stage that reads a companion volume (a mask) stays `POINTWISE` and overrides `stream_region(name, tensor, context, attribute)`: both dispatchers (the reader's patches and regions, the writer's slabs) hand it where its region sits; `SLAB`+`stream_slab` is the write-side-only contract for a side effect needing the output's slabs in order (`InferenceStack`), and the reader treats it as `WHOLE_VOLUME`. A `REGRID` stage owns both halves: `stream_region_source()` (the pull map) and `stream_region()`. Pair `inverse()` if `apply_inverse`; override `prepare(konfai_args)` only when the stage builds a sub-object from configuration of its own (`Reduce` → its operator).
- **Augmentation:** subclass `data.augmentation.DataAugmentation`; `_state_init` (sample params per case index) + `_compute` (apply lazily). Only `Mask`/`Permute` may change shape. Declare `_patch_locality` like a transform so the copies stream (`Rotate` quarter turns are `ORIENTATION`, free angles and `Scale` `REGRID` through the affine's own pull box, `Noise`/`CutOUT` `POINTWISE` because their field and box are functions of the voxel's position); a draw that depends on the place overrides `_stream_region(name, index, a, tensor, context)`. A copy's draws are keyed by the case NAME (`_drawn_from(seed, name, kind, occurrence)`), never by its index. A draw is also a **chain stage**: `TransformLoader` resolves a bare name against `data.transform` first and `data.augmentation` second, and **reverses that order past an `Expand` marker** (`prefer_augmentation`), where the chain is the copies' draws: so `Flip`, `Mask`, `Permute` and `Foreign`, which both packages define, are the transform before the marker and the draw after it (the module-qualified spelling forces either). That is how a `transforms:` block interleaves draws and transforms.
- **Reduction:** subclass `data.reduction.Reduction`; implement `__call__(list[Tensor]) -> Tensor` over the `[1, K, C, *spatial]` layout both engines hand over. Two consumers, one vocabulary: the predictor folds one case's copies (ensemble/TTA), `data.transform.Reduce` folds N **cases** into one. Declare `voxel_local = True` only if every output voxel reads the same voxel of each input (**a wrong `True` corrupts a streamed output**: the gate checks nothing else), `incremental = True` if `accumulate` can fold one at a time, `working_multiple` = the buffers-worth the operator allocates ON TOP of what it is handed (the plan multiplies it into the peak it sizes regions against). An operator whose route depends on the COUNT overrides `working_multiple_for(cases)` and the attribute stays its worst case: `Median` selects the middle by a network of element-wise min/max up to five members (measured 1.0 at three, 2.5 at four, 1.5 at five) and sorts the stack past that (4.0), and override `output_channels(channels, cases)` when the fold changes the channel count (`Concat` does). `Reduce` refuses a non-`voxel_local` operator outright.
- **Imaging format:** add a `Dataset.AbstractFile` backend, dispatch it in `File.__enter__`, register aliases in `SUPPORTED_EXTENSIONS`, or in `SUPPORTED_BACKEND_FORMATS` when the token is not a suffix any file carries (`:itktransform` writes `<group>.h5`), since only the extensions are probed on disk; import-guard the heavy lib and raise a `DatasetManagerError` naming the extra rather than degrading in silence.

**Classpaths:** a bare name (e.g. `Dice`) resolves inside that kind's package; `module:Class` imports *any* module: a local file (`Loss:MyWrapper`) or an installed library (`monai.losses:DiceLoss`, `torch:nn:L1Loss`). Model classpaths resolve against `konfai.models.python`. The pre-1.6.0 absolute form `konfai.models.<kind>.<file>:<Class>` still resolves via a rewrite + `DeprecationWarning`; new code uses the relative or `default|` form.

**YAML model builder** (`utils/model_builder.py`): builds a `Network` from a `.yml`, **safe by construction** (node types must come from two curated registries, no `eval`/import injection). The shipped catalog (`models/yaml/`, 14 models incl. `UNet`/`NestedUNet`/`ResNet`/`UNETR`/`ViT`/`VNet`) now covers the feed-forward subset; custom-`forward` models (DDPM/DiffusionGAN/ConvNeXt) stay Python. `default|<Name>.yml` addresses the flat catalog only; a name with a path separator is refused.

## 5. Apps (`konfai-apps`)

A separate package layered on KonfAI's **public** API (core never imports it). An "app" bundles a config + custom `.py` + `.pt` weights, resolved from a Local dir, a HuggingFace repo, or a Remote server; the `apps/*` bundles are thin CLI wrappers.

> ⚠️ **Trust model.** Resolving an app **copies and imports its `.py` files** → it **runs arbitrary code**. It also **pip-installs its `requirements.txt` by default** (only missing/mismatched packages; core packages like `torch`/`konfai` are never touched; opt out with `KONFAI_APPS_INSTALL_REQUIREMENTS=0`). **Only resolve apps from sources you trust.**

## 5b. MCP server (`konfai-mcp`)

A third independent package (depends only on KonfAI's public API) exposing a **FastMCP** server so an LLM agent can inspect a dataset → author YAML → run train/predict/evaluate → monitor jobs → compare runs → iterate. On `main` since v1.6.0 and published to PyPI by the release workflow. Jobs run in a **`spawn`** subprocess (training may init CUDA); `validate_config_semantics` and `run_component_smoke_test` run in a **spawn subprocess** (never in the server process), are side-effect-free (config bytes are snapshotted/restored **in the parent**, so the restore survives a subprocess timeout kill), and re-import edited workspace code; discovery is via `list_components` / `describe_extension_points` / `describe_config_schema` / `check_external_dependency`. Tests: `pip install -e ./konfai-mcp` then `python -m pytest konfai-mcp/tests` (the segmentation E2E needs the imaging extra).

**Working on the MCP server, how to validate a change:**

- **Synthetic fixtures:** `pixi run --environment dev python audit/make_fixtures.py` (note: `audit/` is
  currently local-only/untracked, so commit it or regenerate it before relying on this flow elsewhere) builds a segmentation
  dataset, a registration pair with a known translation, a synthesis pair, a 3-level OME-Zarr store, and
  corrupted/unsupported inputs under `audit/fixtures/` (procedural, no patient data). Reuse these, do not
  invent ad-hoc data in `/tmp`.
- **Drive it black-box first, not by tool name.** Formulate a real objective ("segment these CT volumes"),
  then exercise the loop through a `fastmcp.Client` exactly as `test_mcp_server_segmentation_pipeline.py`
  does. A new tool is not "done" because it returns without an exception.
- **Verify outputs, not return codes.** After `run_*` + `wait_for_job`, assert the job `status=="done"`
  (never trust a green `validate_*` alone: its default level `instantiate` runs no train step; only
  `level='train_step'` runs one forward+backward), then open the
  produced files: `read_session_file` the config, check `Predictions/<name>/Dataset` exists, and read the
  `Metric_*.json` via `get_run_metrics`. Confirm the numbers correspond to the requested task.
- **Validating a new tool:** (1) its `next_actions` must be registered tool names; the anti-drift test in
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
  `read/write_session_file` are path-jailed (dataset tools read arbitrary host paths by design, so keep it
  that way only for the trusted-local deployment, and never widen writes). `cancel_job` now reaps the whole
  process group: the job runs `os.setsid()` and cancel sends the signal via `os.killpg`, so `mp.spawn` DDP
  grandchildren are killed with the middle process (regression test:
  `test_cancel_reaps_the_whole_process_group_including_grandchildren`).
- **Regenerate derived docs:** after changing a tool's description run
  `python konfai-mcp/scripts/generate_tool_reference.py` (the committed skill reference is generated).

## 6. Running things

```bash
pixi run check                                                    # lint + format-check + core tests + apps tests (run ONCE before finalising)
pixi run test                                                     # core unit + integration (tests/), ~6 min (pytest-xdist), not an iteration loop
pixi run test-fast                                                # dev loop (~1m40): skips slow oracle + integration tests
pixi run --environment dev typecheck                              # mypy konfai
pip install -e ./konfai-apps && pixi run --environment dev python -m pytest konfai-apps/tests   # apps suite (separate)
pip install -e ./konfai-mcp  && pixi run --environment dev python -m pytest konfai-mcp/tests    # mcp suite (separate)
```

The Pixi `dev` env carries the imaging extras; a bare `pip install .[dev]` does not. `pixi run test` does **not** run `konfai-apps/tests` or `konfai-mcp/tests`; install those packages first (they pull their own runtime deps), exactly as their CI does. Install runtime extras with `pip install konfai[<extra>]` (`itk`, `hdf5`, `dicom`, `omezarr`, `imaging`, `tensorboard`, `lpips`, `ssim`, `fid`, `cluster`, `export`, …).

## 6b. Releasing

Versions are **tag-derived** (`setuptools_scm`, `^v(?P<version>.*)$`) for every package, so never hand-edit
a version. Pushing a `v*` tag runs `.github/workflows/publish.yml`: test (core + apps + mcp + studio) → build
the 9-package matrix (konfai, konfai-apps, konfai-mcp, konfai-studio, and the 5 `apps/*` bundles) → publish to
PyPI via OIDC → build the Docker image once `konfai` is visible on PyPI. `apps/*` bundles pin `konfai==` and
`konfai-apps==`, and `konfai-studio` pins `konfai-mcp==` (via its `setup.py`), all the same version, so the
whole matrix releases in lockstep. `konfai-studio` is the one exception to the pure-Python build: the build
job runs `npm ci && npm run build` first (its React front is git-ignored) and then `python -m build --wheel`
(wheel-only, because the sdist file-finder would drop the built `web/`).

Before tagging: `pixi run check` green; both sibling suites green; and, because the test job only exercises
the **source tree**, confirm the built wheel still ships `konfai/models/python/**` and `konfai/models/yaml/*.yml`
by installing it **non-editable** in a clean venv (an editable install hides PEP 420 / `package-data` breakage).

**The changelog is written before the tag, not after.** `CHANGELOG.md` is generated from the commits and the
publish workflow renders the same section into the GitHub Release, so the file must already contain the version
being tagged:

```bash
uvx --from commitizen cz changelog --unreleased-version vX.Y.Z --start-rev v1.5.8   # 1. write the section
git commit -am "ci: changelog for vX.Y.Z"                                          # 2. commit it
git tag -s vX.Y.Z -m "vX.Y.Z" && git push origin vX.Y.Z                            # 3. sign, tag, push
```

`-s` and not `-a`: releases are signed (`v1.7.0` is), and `tag.gpgsign` already makes `-a` sign on a machine
that has it configured, which is exactly why `-s` is spelled out. On a machine that does not, `-a` publishes an
unsigned release in silence where `-s` refuses. Commitizen's own `gpg_sign` is deliberately NOT set: it would
only apply to `cz bump`, which does not tag here, so it would be a setting that describes an intention nothing
acts on.

`cz bump` computes the right increment (`cz bump --dry-run` is a good second opinion on major/minor/patch) but
does **not** tag here: with `version_provider = "scm"` there is no version file to write, so when the changelog
is already current it has nothing to commit and stops silently. Tag by hand. `--start-rev v1.5.8` is required:
Conventional Commits only took hold at `v1.5.9`, and rendering further back emits empty version headings.

## 7. Invariants, do NOT break

- **Never load a full volume into RAM** on a route that can stream. Use lazy/patch/streaming access (`can_stream_patch`, `read_data_slice`). TRANSFORM's `WHOLE-VOLUME` verdict is the one exception, and it is a route the plan declares and prices, never a read a stage takes on its own.
- **Channel-first `[C,(Z),Y,X]`; spacing `(x,y,z)`.** `Attribute` stringifies every value and reparses geometry via `np.fromstring`, so only flat scalars / 1-D arrays round-trip. Read via `__getitem__`/`get_np_array`.
- **`KONFAI_config_file` + `KONFAI_CONFIG_MODE` must be set before any `Config()`** (tests must `monkeypatch.setenv` both); workflows require `KONFAI_CONFIG_MODE='Done'`. Reading a config rewrites it on disk.
- **Patch ordering** must match between read (`disassemble`) and write (`Accumulator`); for PREDICTION/EVALUATION all patches of a case stay on the same DDP rank.
- **`outputs_criterions` keys equal a module's dotted path**; the `:`/`.` separators are load-bearing.
- **`state_dict` load/save does not recurse into nested `Network`s** (each owns its optimizer/state); alias lists are positional.
- **The YAML model builder is the trusted/untrusted boundary**: only registry types, and module names contain no `.`.
- **`konfai-apps` is a separate package**; `apps/` is excluded from the `konfai` wheel. Core must never import
  `konfai_apps` **at module level**. Known exception: `data/transform.py` `KonfAIInference.infer_entry` does a
  lazy, guarded import: a layering inversion pending an owner decision (see `REFACTORING.md` §C); do not add more.
- **The pretrained bridge fills every target tensor or raises**; never report a partial load as success.
- **The config write is atomic** (temp + `os.replace`); a reader must never see a truncated config and bind all-defaults.

## 7b. Contracts with the ecosystem

- **SlicerKonfAI** (separate repo) drives `konfai-apps` by CLI + JSON. It imports, from
  `konfai_apps.app_repository`: `current_free_vram(devices, remote_server=None)`, `get_app_repository_info`,
  `is_app_repo`, `LocalAppRepositoryFromDirectory`, `LocalAppRepositoryFromHF`, `AppRepositoryError`, plus the
  `get_parameters() -> {values, constraints}` params primitive that drives its advanced-params dialog.
  **Renaming or dropping any of these breaks Slicer silently** (it happened once: PR#33 dropped
  `current_free_vram`, PR#35 restored it). Grep the Slicer checkout before touching that surface.
- **HF bundles** (`hf_bundles/*`) carry `app.json` + config + `requirements.txt` + `.pt` + custom `.py`. Their
  configs use **bundle-relative** classpaths (`ResidualEncoderUNet.yml`, `model:Unet_TS_CT`,
  `Model:RegistrationNet`), never KonfAI's internal module paths, which is why the models→`models.python`
  move did not break them. Validate a bundle by loading its config on a **copy** (reading mutates).

## 7c. Security boundaries

Three, and only three, places decide trust. Keep them honest:
1. **YAML model builder**: registry types only; the `default|` catalog name must stay a bare filename.
2. **`konfai-apps` app resolution**: runs the app's `.py` **and** pip-installs its `requirements.txt` by
   default (`KONFAI_APPS_INSTALL_REQUIREMENTS=0` opts out). Protected core packages are matched by **PEP 503
   canonical name** (`konfai_apps` ≡ `konfai-apps`); transitive deps are *not* policed, so say so and don't overclaim.
3. **`konfai-mcp`**: validation/smoke-tests run only in a spawn subprocess (never the server process);
   `read/write_session_file` are path-jailed; dataset tools may *read* arbitrary host paths by design, but
   **writes must never widen** (any tool that composes a write target must reject path separators);
   `cancel_job` reaps the whole process group.

## 7d. Traps that have bitten before

- An **editable install hides packaging breakage** (`models/python` is PEP 420; catalog `.yml` is package-data).
- **A green `validate_*` proves little**: its default level `instantiate` runs no train step; only
  `level='train_step'` does a real forward+backward.
- **`transform_shape()` must be exact**: patch planning trusts it, a wrong prediction corrupts reassembly.
- **Reading a config mutates it**, so snapshot bytes before any validation that builds a workflow.
- **Adding a workflow kind touches ~12 registries + ~8 `Literal`s**; prefer one descriptor table over editing each.
- **Union coercion in the config binder is declaration-order-sensitive**: `overlap: 0.25` once bound `0`
  (lossy `int` won). Fixed; pinned by
  `test_config.py::test_apply_config_union_keeps_the_value_type_over_lossy_coercion`. Any new union-typed
  config key still needs a test.
- **Nested-`Network` save/load use different key coordinates**: `checkpoint_save` writes dotted paths,
  `Network.load` looks up bare class names, so composite models (GAN family) silently lose optimizer/scheduler
  state on RESUME until fixed. Any change near `get_networks()`/`load` must keep the two in agreement.
- **Per-epoch augmentation redraws never reach persistent DataLoader workers**: `persistent_workers=True`
  (the `num_workers>0` default) freezes inline augmentations at their first-epoch draw.
- **The train/val split is drawn from the unseeded global RNG at `Trainer.__init__`** (before per-rank
  seeding), so `manual_seed` does not cover it and RESUME re-splits.
- **Nothing checks that `Prediction.yml` preprocesses a group the way `Config.yml` did**, so a
  train/predict transform mismatch is silent: the run succeeds and only the output is wrong. The
  Synthesis example shipped with `Standardize(mask: None)` in training against `Standardize(mask: MASK)`
  plus an extra input `Mask` in prediction: same checkpoint, **409 HU of MAE instead of 98**. The tell
  is an evaluated metric that contradicts the training loss by an order of magnitude (training MAE
  0.018 in `[-1, 1]` ≈ 36 HU); diff the two transform stacks before blaming the model or the epochs.

## 8. Conventions & rules

- **Code:** line length 120 (Ruff); type annotations on new public functions; Apache-2.0 SPDX header on every new source file; prefer `pathlib.Path`; use the error classes in `utils/errors.py` (do not invent exceptions); import-guard heavy optional deps (`SimpleITK`/`h5py`/`pydicom`/`zarr`), failing at point-of-use with an install hint, not at import.
- **Commits:** Conventional Commits (`cz check`): `type(scope): subject`, imperative, < 72 chars. No AI-agent branding (`claude`/`codex`/"generated by/with") and no AI co-author trailers.
- **For agents:** read before editing; keep diffs small (one logical change per PR, no unrelated reformats); **iterate cheaply, verify once**: while developing run only the per-module test file you touched (`pytest tests/unit/test_<module>.py`), then `pixi run test-fast` (~1m40); reserve the full `pixi run check` (and the apps suite if you touched `konfai-apps`) for one final pass before finalising, because the full suite takes ~6 min and re-running it per edit adds up fast; no new runtime dependency without an explicit request + a matching `pyproject.toml` update in the same commit; update docs and `tests/unit/test_config.py` when changing config binding; do not skip pre-commit with `--no-verify`.
