# Architecture (for contributors)

```text
cli.py                    argparse front end: list, show, register, eval, uncertainty, apply
  │
impact_reg.py             ImpactRegKonfAIApp: the orchestration, one method per command
  │   register  ─► validate inputs, presets and --set, before anything runs
  │             ─► per preset: a plan from its declared cost (konfai.utils.vram.max_voxels), then whole
  │                (_infer_preset) or in two passes (_register_tiled: global pass, resampled by KonfAI →
  │                _prewarp → tile pass, tiled by KonfAI → _compose)
  │             ─► ensemble (_ensemble_mean) ─► moved image (_derive_moved) ─► register.json and the summary
  │   evaluate / uncertainty / apply: KonfAI transforms (Resample, Reduce, Write) and the evaluation configs
  │                                   shipped in evaluation/
  ▼
konfai-apps infer         one subprocess per preset (and per pass): resolves the preset from Hugging Face or a local
  │                       directory, installs its requirements, runs KonfAI's PREDICTION on the pair
  ▼
models/<engine>.py        the preset's model class (Prediction.yml names it): RegistrationNet, a KonfAI network whose
  │                       single module registers the pair and returns the displacement field
  ├── elastix.py, elastix_engine.py, elastix_install.py   parameter maps, the elastix subprocess, the binary
  ├── fireants.py                                          FireANTs stages and their IMPACT loss (KonfAI's extraction)
  ├── convexadam.py                                        itk-impact's coarse and fine stages
  ├── impact_loss.py                                       the IMPACT settings the three engines share
  └── intensity.py, orientation.py                         shared: the graph module running an engine, clamped
                                                           intensities, masks, voxel-axis order
```

## Contracts

- **A preset's only output is the displacement field**, on the fixed grid, pull-back convention, physical components
  (see [How it works](how-it-works.md)). The engines return a channel-first array on the fixed image they were given;
  KonfAI writes it as `Transform.h5` through its `:itktransform` backend. Everything else (moved image, ensemble,
  evaluation) is the orchestrator's, computed from the field.
- **One subprocess per preset.** KonfAI's configuration and runtime state are per process, so each preset (and each
  pass of a tiled run) is its own `konfai-apps infer`. The orchestrator stages inputs and collects outputs in a work
  directory (`--tmp-dir`, or hidden beside `--output`), kept when a run fails.
- **Streaming.** Every volume-sized step of the orchestrator (resample, compose, average, derive) is a KonfAI transform
  that reads and writes regions. Only the engines hold a whole pair (or a whole tile) in memory.
- **Presets are data.** Engine code lives in this package, its configuration in the preset (`Prediction.yml`,
  parameter maps), its cost in `app.json` (`vram_bytes_per_voxel`, `ram_bytes_per_voxel`, `tiling`). A preset's
  `requirements.txt` names the package versions it needs, and `requirements_no_deps` in `app.json` those installed
  without their dependencies (FireANTs); konfai-apps never replaces an installed core package to satisfy them.
- **SlicerImpactReg drives the CLI** (`register --keep-fields`, `eval`, `uncertainty`, the output layout above):
  keep those arguments and files stable.
- **KonfAI Studio drives it through konfai-mcp**: `run_app` (action `infer`) on an app whose `app.json` says `task:
  registration` calls `ImpactRegKonfAIApp.register` in a job (the presets repository taken from the app reference), and
  `run_registration_evaluate` calls `evaluate`; `run_app` (action `evaluate`) refuses such an app, whose bundled
  evaluation configs read the moving data without the transform.

## Adding an engine

1. A module in `models/` with a `RegistrationNet(network.Network)` taking its settings as typed, documented arguments
   (KonfAI binds them from `Prediction.yml` and shows them in `show NAME`), and a module returning the field.
2. A preset folder: `app.json`, `Prediction.yml` naming `impact_reg_konfai.models.<module>:RegistrationNet`,
   `requirements.txt`, and optionally `Prediction_tile.yml` with the `tiling` and cost declarations.
3. A case in `tests/integration/test_engines_cpu.py` (a blob translated by a known vector, built through the preset's
   own `RegistrationNet`, whose field must be that vector on the fixed grid), and the preset in
   `tests/integration/test_preset_contract.py` (every published preset binds to this package's models).

## Tests

```bash
cd apps/impact_reg
CUDA_VISIBLE_DEVICES= python -m pytest tests/unit          # CPU, seconds
IMPACT_REG_ENGINE_TESTS=1 python -m pytest tests/integration   # real engines on tiny pairs, CPU; needs the engines
```

Unit tests cover the orchestration (tiling plans, composition order, validation, output layout), the CLI, the engines'
field contract with fakes, the elastix installer and parameter maps, and eval/uncertainty on small real images. The
integration tests run each engine for real (itk-impact, fireants and the elastix-IMPACT binary installed, network for
the models); the CI lane that installs the engines sets `IMPACT_REG_ENGINE_TESTS=1`, and there a missing or broken
engine fails instead of skipping.
