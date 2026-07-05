# Ecosystem

KonfAI is the core, but several packages and tools sit around it. This page maps
how they relate and — importantly — **what is shipped versus what is still
in-progress or external**, so you know what you can rely on today.

## The map

```text
                        ┌─────────────────────────────┐
   YAML configs  ─────► │   konfai (core, PyPI)        │  TRAIN / PREDICTION / EVALUATION
                        │   train · predict · evaluate │
                        └──────────────┬──────────────┘
                                       │ public API
                        ┌──────────────▼──────────────┐
                        │   konfai-apps (PyPI)         │  package a workflow as an "app"
                        │   CLI · HTTP server · Python │
                        └───┬───────────┬───────────┬──┘
                            │           │           │
             apps/* wrappers│   HTTP/SSE│    stdio  │
                            ▼           ▼           ▼
                  impact_synth,   SlicerKonfAI   konfai-mcp
                  impact_seg,     (3D Slicer GUI) (agent server)
                  totalseg, …       [external]     [private]
```

## What ships

| Piece | Status | What it is |
| --- | --- | --- |
| **`konfai`** | ✅ Shipped (PyPI) | The core framework: config-by-reflection, lazy patch-based data, model graphs, and the three YAML workflows. This is what the rest of this documentation is about. |
| **`konfai-apps`** | ✅ Shipped (PyPI, own CI) | Packages a mature workflow as a reusable **app** — see {doc}`../usage/apps`. |
| **App bundles** (`apps/`) | ✅ Shipped (thin wrappers) | Ready-to-use CLI shims: `impact-synth-konfai`, `impact-seg-konfai`, `mrsegmentator-konfai`, `totalsegmentator-konfai`. Config + weights live on Hugging Face and download on first run. |
| **`impact-reg-konfai`** | 🟡 Shipped, heaviest | A full multi-preset registration orchestrator (Elastix + the IMPACT semantic metric), **not** a thin wrapper. The most moving parts of the five. |
| **Demo data & models (HF)** | ✅ Published | `VBoussot/konfai-demo` (demo dataset), plus per-app model repos (`ImpactSynth`, `ImpactSeg`, `TotalSegmentator-KonfAI`, `MRSegmentator-KonfAI`, `ImpactReg`) and `impact-torchscript-models` (the SAM2.1 backbone behind `IMPACTSynth`). |
| **Challenge repos** | ✅ External | Top-ranking MICCAI-challenge projects built on KonfAI (SynthRAD 2025 T1/T2, TrackRAD 2025, Panther, CURVAS, CURVAS-PDACVI). Referenced from the README; not in this tree. |

## Ready-to-use apps

| App CLI | Task | Modality | Models |
| --- | --- | --- | --- |
| `impact-synth-konfai synthesize` | Synthetic CT (sCT) | MR → CT, CBCT → CT | `MR`, `CBCT` |
| `impact-seg-konfai segment` | Multimodal body segmentation (11 labels) | CBCT / MR / CT | `body` |
| `mrsegmentator-konfai segment` | Whole-body MRI segmentation | MRI | folds 1–5 |
| `totalsegmentator-konfai segment` | Whole-body CT/MRI segmentation | CT / MRI | `total`, `total_mr`, 3 mm variants |
| `impact-reg-konfai register` | Multimodal deformable registration | MR/CT, CBCT/CT | presets |

Each also exposes `eval` and (except TotalSegmentator) `uncertainty`; the thin
wrappers add `pipeline`. See {doc}`../usage/apps` for how to run them and
{doc}`../reference/cli` for the full flag reference.

## What is in-progress or external

| Piece | Status | Notes |
| --- | --- | --- |
| **`konfai-mcp`** | 🧪 Working, not yet published | A `fastmcp` server that exposes KonfAI to LLM agents: dataset inspection, config authoring/validation, launching training/prediction/evaluation runs, and reading live metrics. Working and tested, but not yet published. |
| **SlicerKonfAI** | 🧪 External GUI | A [3D Slicer](https://github.com/vboussot/SlicerKonfAI) client of the `konfai-apps` CLI/server. Real and useful, but coupled to `konfai-apps` through a **brittle byte-level contract** (progress format, `InferenceStack.mha`, `app.json`, checkpoint schema) that can break on a release without a contract test. |
| **ONNX export → `konfai-rs`** | 🧪 Experimental | `konfai/export.py` produces ONNX + a manifest for a planned portable (native/WASM) inference engine. Python-API-only, single static-shape head, feed-forward models only. See {doc}`../reference/python-api`. |

```{note}
**Trust model.** Resolving a `konfai-apps` app copies and imports its `.py`
files (arbitrary code) and pip-installs its `requirements.txt` by default
(opt out with `KONFAI_APPS_INSTALL_REQUIREMENTS=0`), so only resolve apps
from sources you trust — see {doc}`../reference/python-api`.
```

## See also

- {doc}`../usage/apps` — what an app is and how to run apps from the CLI
- {doc}`../reference/cli` — the full flag reference for the app CLIs
