[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/vboussot/KonfAI/blob/main/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/konfai)](https://pypi.org/project/konfai/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/vboussot/KonfAI/actions/workflows/konfai_ci.yml/badge.svg)](https://github.com/vboussot/KonfAI/actions/workflows/konfai_ci.yml)
[![Documentation Status](https://readthedocs.org/projects/konfai/badge/?version=latest)](https://konfai.readthedocs.io/en/latest/?badge=latest)
[![Paper](https://img.shields.io/badge/📌%20Paper-KonfAI-blue)](https://www.arxiv.org/abs/2508.09823)
[![Agent-ready (MCP)](https://img.shields.io/badge/🤖%20Agent--ready-MCP-8A2BE2)](https://konfai.readthedocs.io/en/latest/usage/mcp.html)

# 🧠 KonfAI

<img src="https://raw.githubusercontent.com/vboussot/KonfAI/main/logo.png" alt="KonfAI Logo" width="230" align="right"/>

**KonfAI is a modular, YAML-driven deep learning framework for medical imaging,
built on PyTorch.** You describe an entire pipeline — data loading, model, losses,
metrics, augmentations, optimizer, and the train / predict / evaluate workflow —
in **configuration**, not orchestration scripts. The config *is* the experiment:
a complete, reproducible record you can diff, share, and re-run.

```yaml
Trainer:
  Model:
    classpath: UNet.yml          # a model, referenced by name
  Dataset:
    groups_src: { CT: {...}, SEG: {...} }   # channel-first, lazy, patch-based
  epochs: 100
```

```bash
konfai TRAIN -c Config.yml --gpu 0     # then PREDICTION, then EVALUATION
```

KonfAI has powered **top-ranking MICCAI-challenge results** across segmentation,
registration, and synthesis:
[SynthRAD2025 T1](https://github.com/vboussot/Synthrad2025_Task_1) ·
[SynthRAD2025 T2](https://github.com/vboussot/Synthrad2025_Task_2) ·
[CURVAS PDACVI](https://github.com/vboussot/CurvasPDACVI) ·
[TrackRAD2025](https://github.com/vboussot/TrackRAD2025) ·
[Panther](https://github.com/vboussot/Panther) ·
[CURVAS](https://github.com/vboussot/CURVAS)

> 📄 **Paper:** [KonfAI: A Modular and Fully Configurable Framework for Deep Learning in Medical Imaging](https://www.arxiv.org/abs/2508.09823) (Boussot & Dillenseger, 2025)

> 🤖 **Agent-native.** KonfAI ships an **[MCP server](https://konfai.readthedocs.io/en/latest/usage/mcp.html)**
> so an LLM agent can drive the *entire* experiment loop — inspect a dataset, author & validate YAML,
> launch train / predict / evaluate, monitor jobs, and compare runs — always grounded in the same
> reproducible configs a human would run. → **[Agents & MCP](https://konfai.readthedocs.io/en/latest/usage/mcp.html)**

---

## Why KonfAI?

Most frameworks focus on *models*. **KonfAI focuses on *pipelines*.**

- 🧩 **Compose** full workflows from modular, named components — no glue code.
- 🔁 **Iterate** by editing YAML, not rewriting Python.
- 🔬 **Reproduce** every run: KonfAI resolves and writes back the full config.
- 🩻 **Scale to real volumes**: volumes are read as patches, and a preprocessing chain that allows it is [streamed](https://konfai.readthedocs.io/en/latest/concepts/streaming.html) from disk rather than loaded.
- 📦 **Ship** a mature workflow as a reusable [**KonfAI App**](https://konfai.readthedocs.io/en/latest/usage/apps.html) (CLI, HTTP server, 3D Slicer).
- 🤖 **Drive it with an agent**: KonfAI-MCP lets an LLM inspect data, author configs, and launch runs.

---

## Install

```bash
pip install "konfai[imaging]"     # core + all imaging backends (recommended)
pip install konfai                # core only (bring your own data reader)
```

`[imaging]` pulls SimpleITK / h5py / pydicom / zarr — needed to read `.mha`,
`.nii.gz`, DICOM, and OME-Zarr. For the full extras matrix (`ssim`, `fid`,
`lpips`, `export`, `cluster`, …) and a reproducible Pixi setup, see the
[installation guide](https://konfai.readthedocs.io/en/latest/getting-started/installation.html).

---

## Three workflows, three configs

KonfAI is command-driven; each CLI state maps to one YAML file:

| Command | Config | Does |
| --- | --- | --- |
| `konfai TRAIN` / `RESUME` | `Config.yml` (`Trainer:`) | fit a model |
| `konfai PREDICTION` | `Prediction.yml` (`Predictor:`) | patch/TTA/ensemble inference → datasets |
| `konfai EVALUATION` | `Evaluation.yml` (`Evaluator:`) | metrics on saved predictions |

Full CLI reference (flags, `konfai-cluster`, `konfai-apps`):
[docs/reference/cli](https://konfai.readthedocs.io/en/latest/reference/cli.html).

---

## Quickstart (5-minute teaser)

```bash
git clone https://github.com/vboussot/KonfAI.git && cd KonfAI
pip install -e ".[imaging]"
cd examples/Segmentation

# download the small public demo dataset
pip install -U "huggingface_hub[cli]"
hf download VBoussot/konfai-demo --repo-type dataset --include "Segmentation/**" --local-dir Dataset
mv Dataset/Segmentation/* Dataset/ && rmdir Dataset/Segmentation && rm -rf Dataset/.cache

konfai TRAIN -y --gpu 0 --config Config.yml     # use --cpu 1 if you have no GPU
```

> 💡 After a run, `Config.yml` will contain the resolved defaults KonfAI
> materialised — that's expected, and it's what makes runs reproducible.

The full walkthrough (predict, evaluate, what to inspect, common first issues,
notebook entry points) lives in the
[**Quickstart**](https://konfai.readthedocs.io/en/latest/quickstart.html).

---

## 🩻 How volumes are read

Volumes are read as patches. Whether the volume is *also* held in RAM depends on
`use_cache` and on whether your preprocessing chain can be streamed — KonfAI
derives streamability from the transforms you declared:

| Regime | When | Memory held |
| --- | --- | --- |
| **Cache** | `use_cache: true` (training default) | every case, resident for the whole run |
| **Stream** | `use_cache: false`, chain is streamable | one patch |
| **Buffer** | `use_cache: false`, chain is not streamable | a FIFO of `max(batch_size + 1, shuffle_window)` cases |

A chain streams when every step declares the region it needs: the exact patch
(`OneHot`), a halo (`Dilate`), a remap (`Flip`), a resample (`ResampleToShape`),
or a whole-volume statistic read once from disk (`Normalize`). On the stream
path, a 16 GiB uncompressed `.mha` trains at patch 64³ under an 8 GiB memory cap
with a peak resident set of 0.46 GiB.

→ [**Patch streaming**](https://konfai.readthedocs.io/en/latest/concepts/streaming.html) — what streams, what does not, and why.

---

## What's in the box

Everything below is referenceable by name in YAML. See the
[**built-in component catalogue**](https://konfai.readthedocs.io/en/latest/reference/components/index.html)
for classpaths and constructor arguments.

| Kind | Examples | Catalogue |
| --- | --- | --- |
| **Models** | `UNet`, `NestedUNet`, `ResNet`, `VAE`, `VoxelMorph`, GAN/diffusion families | [models](https://konfai.readthedocs.io/en/latest/reference/components/models.html) |
| **Losses & metrics** | `Dice`, `MAE`, `PSNR`, `SSIM`, `LPIPS`, `FID`, `CrossEntropyLoss`, `TRE`, `IMPACTReg`, `IMPACTSynth` | [losses-metrics](https://konfai.readthedocs.io/en/latest/reference/components/losses-metrics.html) |
| **Transforms** | `Standardize`, `Normalize`, `Clip`, `Resample*`, `OneHot`, `Crop` (~40) | [transforms](https://konfai.readthedocs.io/en/latest/reference/components/transforms.html) |
| **Augmentations** | `Flip`, `Rotate`, `Elastix`, `Noise`, `CutOUT` (~15) | [augmentations](https://konfai.readthedocs.io/en/latest/reference/components/augmentations.html) |
| **Schedulers** | weight (`Constant`, `CosineAnnealing`) + LR (`PolyLRScheduler`, `Warmup`, any torch) | [schedulers](https://konfai.readthedocs.io/en/latest/reference/components/schedulers.html) |
| **Storage backends** | ITK, HDF5, DICOM series, OME-Zarr | [storage-backends](https://konfai.readthedocs.io/en/latest/reference/components/storage-backends.html) |

Not limited to these: any importable class (`monai.losses:DiceLoss`,
`torch:nn:L1Loss`, or a local `Model:MyNet`) works via the `module:Class` form.

---

## 🤖 Agent-ready by design

KonfAI is built to serve as a **deterministic backend for LLM-driven
experimentation**. Through the **KonfAI-MCP server**, an agent can:

- 🔎 inspect datasets and infer their structure
- 📝 generate and validate YAML configurations
- 🚀 launch training / prediction / evaluation runs
- 📈 read live metrics, compare runs, and iterate

Every execution stays **reproducible, structured, and grounded in the same YAML
workflows** a human would run — bridging LLM reasoning and real experimental
execution. See the [ecosystem map](https://konfai.readthedocs.io/en/latest/ecosystem/index.html)
for the current status.

---

## Ecosystem

| Package | What it is |
| --- | --- |
| **`konfai`** | the core framework (this repo) |
| **`konfai-apps`** | package a workflow as an app — [CLI](https://konfai.readthedocs.io/en/latest/reference/cli.html), [HTTP server](https://konfai.readthedocs.io/en/latest/reference/app-server-api.html), [Python API](https://konfai.readthedocs.io/en/latest/reference/python-api.html) |
| **App bundles** (`apps/`) | ready-to-run: `impact-synth`, `impact-seg`, `mrsegmentator`, `totalsegmentator`, `impact-reg` |
| **[SlicerKonfAI](https://github.com/vboussot/SlicerKonfAI)** | run KonfAI apps from a 3D Slicer GUI |
| **KonfAI-MCP** | expose KonfAI to LLM agents — inspect data, author configs, launch and monitor runs |

See the [ecosystem map](https://konfai.readthedocs.io/en/latest/ecosystem/index.html)
for what is shipped vs. in-progress.

---

## Documentation

📚 **Full docs: <https://konfai.readthedocs.io/en/latest/>**

- [Quickstart](https://konfai.readthedocs.io/en/latest/quickstart.html) — first end-to-end run
- [Core concepts](https://konfai.readthedocs.io/en/latest/concepts/index.html) — how YAML becomes Python objects
- [Component catalogue](https://konfai.readthedocs.io/en/latest/reference/components/index.html) — everything you can configure
- [Examples](https://konfai.readthedocs.io/en/latest/examples/index.html) — runnable Segmentation & Synthesis workflows

🐳 **Docker:** `vboussot/konfai` —
[guide](https://konfai.readthedocs.io/en/latest/usage/docker.html).

---

## Development & contributing

```bash
git clone https://github.com/vboussot/KonfAI.git && cd KonfAI
pixi install
pixi run test      # run the test suite
pixi run check     # lint + format-check + test (run before pushing)
```

Contributions are welcome — improve examples, clarify docs, add tests, or extend
models / transforms / apps. See the
[developer guide](https://konfai.readthedocs.io/en/latest/development.html).

**AI coding agents:** start with [`AGENTS.md`](AGENTS.md) — the canonical
reference for conventions, commands, and repository rules.

---

## Citation

```bibtex
@article{boussot2025konfai,
  title   = {KonfAI: A Modular and Fully Configurable Framework for Deep Learning in Medical Imaging},
  author  = {Boussot, Valentin and Dillenseger, Jean-Louis},
  journal = {arXiv preprint arXiv:2508.09823},
  year    = {2025}
}
```

Licensed under [Apache-2.0](LICENSE).
