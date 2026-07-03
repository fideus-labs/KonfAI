# Stability & maturity

KonfAI ships a large component zoo. Not all of it is equally mature: some pieces
are production-grade and CI-tested, some are usable but unexercised, and a few are
research scaffolding that **crashes with its default arguments**. This page is the
honest, at-a-glance maturity map so you can tell the two apart before you build a
config on top of something.

```{note}
Maturity here is assessed from the source (tests, examples, and code that raises
by design) — see the per-kind pages under {doc}`components/index` for details and
constructor arguments. When in doubt, prefer the **Stable** rows and the shipped
{doc}`../examples/index`.
```

## Legend

| Label | Meaning |
| --- | --- |
| ✅ **Stable** | Implemented and exercised by tests and/or the shipped examples. |
| 🟡 **Usable** | Implemented and functional, but not covered by tests/examples. |
| 🧪 **Experimental** | Research / undocumented; may need a specific config; interfaces can change. |
| ⛔ **Broken by default** | The default constructor raises or fails; usable only with specific arguments (or not yet). |

## Core framework

| Area | Status | Notes |
| --- | --- | --- |
| Config-by-reflection engine (`utils/config.py`) | ✅ Stable | The `interactive` and `remove` config modes are read but never activated by the shipped CLI — treat as dormant. |
| Data layer: datasets, patching, streaming, DDP sharding | ✅ Stable | The "case stays on one rank" (predict/eval) and shard-truncation (train) rules are load-bearing invariants. |
| Distributed runtime (`utils/runtime.py`), local multi-GPU | ✅ Stable | SLURM submission via `submitit` is optional (`konfai[cluster]`). |
| Three workflows: TRAIN / RESUME / PREDICTION / EVALUATION | ✅ Stable | — |
| Storage backends `SitkFile` / `H5File` | ✅ Stable | — |
| Storage backends `OmeZarrFile` / `DicomFile` | 🟡 Usable | Recent; DICOM write is scalar-array only. `dcm` ≠ `dicom` (see {doc}`components/storage-backends`). |
| YAML model builder | ✅ Stable (small registry) | Safe by construction, but the node registry is small and **cannot express custom-`forward` models** (diffusion/StyleGAN/ConvNeXt/VoxelMorph). |

## Models

| Model | Status | Notes |
| --- | --- | --- |
| `UNet` | ✅ Stable | Tested + the Segmentation example. |
| `NestedUNet` | 🟡 Usable | Implemented; blocks reused elsewhere. |
| `VAE` (deterministic AE), `LinearVAE` | ✅ Stable | `LinearVAE` is tested. |
| `ResNet` | 🟡 Usable | Implemented; not tested in-repo. |
| `UNetpp` | 🧪 Experimental | Untested; hardcoded activations/norms. |
| `Gan` / `Generator` / `Discriminator` | 🧪 Experimental | Research; `SYNCBATCH` needs DDP. |
| `DiffusionGan*`, `CycleGan*`, `cStyleGan.Generator` | 🧪 Experimental | Undocumented research code. |
| `ConvNeXt` | ⛔ Broken by default | 3D (its default) raises; 2D works. Finding N6. |
| `DDPM` | ⛔ Broken by default | `__init__` raises `NotImplementedError`. |
| `VoxelMorph` | ⛔ Broken by default | Raises unless `dim: 2` (default is 3). Prefer the IMPACT-Reg app. |
| `Representation` | 🧪 Experimental | Example-grade; 3D-hardcoded feature dims. |

## Losses & metrics

| Group | Status | Notes |
| --- | --- | --- |
| `MSE`, `MAE`, `ME`, `PSNR`, `Dice`, `CrossEntropyLoss`, `TRE`, `GradientImages`, `KLDivergence`, `Variance`, `Mean`, `PatchGanLoss` | ✅ Stable | Tested and/or used in examples. |
| `SSIM` | ✅ Stable | Needs `konfai[ssim]`. |
| `BCE`, `WGP`, `Gram`, `FocalLoss`, `Accuracy`, `MutualInformationLoss`, `TripletLoss`, `L1LossRepresentation` | 🟡 Usable | `FocalLoss.alpha` is a length-5 list; `Accuracy` never resets — see {doc}`components/losses-metrics`. |
| `LPIPS`, `FID` | 🟡 Usable | Need `konfai[lpips]` / `konfai[fid]`; pinned to CUDA GPU 0. |
| `IMPACTReg`, `IMPACTSynth`, `SAM_Perceptual` | 🟡 Usable | Download TorchScript models from Hugging Face (network required); `SAM_Perceptual` is 2D-only. |
| `PerceptualLoss` | 🧪 Experimental | No-op preprocessing; placeholder default checkpoint. |
| `MAESaveMap`, `DiceSaveMap` | 🧪 Experimental | Return a 3-tuple the normal loss path can't unpack. |

## Transforms & augmentations

| Group | Status | Notes |
| --- | --- | --- |
| `Clip`, `Standardize`, `Normalize`, `TensorCast`, `ResampleToResolution`, `ResampleToShape`, `Padding`, `Mask`, `Statistics` | ✅ Stable | Tests + examples. |
| Most other transforms/augmentations | 🟡 Usable | Functional; not covered by tests/examples. |
| `Flip` (augmentation) | ✅ Stable | The only augmentation used in the shipped train configs. |
| `ResampleTransform.inverse`, `Save`, `Elastix._inverse`, `Mask._inverse` | ⛔ Stubs | No-op / `pass`; see the component pages. |
| `KonfAIInference` (transform) | 🧪 Experimental | Heavy; needs `konfai-apps`; `num_workers: 0`. |

## Tooling & ecosystem

| Piece | Status | Notes |
| --- | --- | --- |
| `konfai-apps` CLI / server / Python API | ✅ Stable | Published to PyPI; own CI. See {doc}`python-api` and {doc}`app-server-api`. |
| App bundles `impact_synth`, `impact_seg`, `mrsegmentator`, `totalsegmentator` | ✅ Stable | Thin CLI wrappers; weights on Hugging Face. |
| App bundle `impact_reg` | 🟡 Usable, heaviest | A full registration orchestrator, not a thin wrapper; the most moving parts. |
| ONNX export (`konfai/export.py`) | 🧪 Experimental | Python-API-only (no CLI); single static-shape head; feed-forward models only. |
| MC-dropout in apps | ⛔ Not applied | Plumbed end-to-end but not yet applied to the prediction config. |
| Remote `--patch-size` / `--batch-size` | ⛔ Dropped remotely | Honored locally; the HTTP endpoints don't accept them. |
| `konfai-mcp` (agent server) | 🧪 Experimental | Working and tested, but unpublished (private branch). |
| SlicerKonfAI | 🧪 External | Downstream GUI client with a brittle byte-level contract to `konfai-apps`. |

## See also

- {doc}`components/index` — the full component catalogue with arguments
- {doc}`../ecosystem/index` — how the surrounding packages relate
- `AUDIT_KONFAI.md` (repo root) — the tracked backlog of known issues
