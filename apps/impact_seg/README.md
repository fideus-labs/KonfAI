[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![PyPI version](https://img.shields.io/pypi/v/impact_seg_konfai.svg?color=blue)](https://pypi.org/project/impact_seg_konfai/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/vboussot/KonfAI/actions/workflows/konfai_ci.yml/badge.svg)](https://github.com/vboussot/KonfAI/actions/workflows/konfai_ci.yml)
[![Paper](https://img.shields.io/badge/📌%20Paper-KonfAI-blue)](https://www.arxiv.org/abs/2508.09823)

# IMPACT-Seg-KonfAI

**Fast and lightweight CLI for multimodal anatomical segmentation using IMPACT-Seg models within the KonfAI framework.**

---

## 🧩 Overview

**IMPACT-Seg-KonfAI** is a lightweight **command-line interface (CLI)** for running **IMPACT-Seg** models through the
[KonfAI](https://github.com/vboussot/KonfAI) deep learning framework.

It provides a simple way to perform **anatomical segmentation inference**, **evaluation**, **ensembling**, and
**uncertainty estimation** on medical image volumes.

The underlying **IMPACT-Seg** models are **multimodal anatomical segmentation** networks built around a **2.5D U-Net
with a residual encoder**. A single model segments **CBCT, MR, and CT** scans into a **consistent label space**,
enabling cross-modality workflows without per-modality retraining.

Pretrained models are automatically downloaded from
[Hugging Face Hub](https://huggingface.co/VBoussot/ImpactSeg).

---

## 🧠 Features

- ⚡ **Fast inference** powered by [KonfAI](https://github.com/vboussot/KonfAI)
- 🤗 **Automatic model download** from Hugging Face
- 🧩 **Multi-model ensembling** and **test-time augmentation (TTA)**
- 🧠 **Supports evaluation workflows with reference data, and uncertainty estimation without reference**
- 🧾 **Multi-format compatibility:** supports all major medical image formats handled by ITK

---

## 🗂️ Available models

Model identifiers are resolved dynamically from the [`VBoussot/ImpactSeg`](https://huggingface.co/VBoussot/ImpactSeg)
repository and passed as the first positional argument.

| Model | Modalities | Labels | Training cohort |
|-------|------------|--------|-----------------|
| `body` | CBCT · MR · CT | 11 | 232 CBCT + 282 MR + 955 CT |

The **`body`** model predicts **11 anatomical labels** spanning soft tissues, cavities, bones, and central structures:

| # | Label | # | Label |
|---|-------|---|-------|
| 1 | subcutaneous_tissue | 7 | pericardium |
| 2 | muscle | 8 | prosthetic_breast_implant |
| 3 | abdominal_cavity | 9 | mediastinum |
| 4 | thoracic_cavity | 10 | spinal_cord |
| 5 | bones | 11 | brain |
| 6 | gland_structure | | |

---

## 🚀 Installation

From PyPI:

```bash
python -m pip install impact-seg-konfai
```

From source:

```bash
git clone https://github.com/vboussot/KonfAI.git
python -m pip install -e apps/impact_seg
```

---

## ⚙️ Usage

The CLI is organised into sub-commands, mirroring the KonfAI Apps operations:

| Sub-command | Purpose |
|---|---|
| `segment` | Run the segmentation (inference). |
| `eval` | Evaluate a segmentation against a reference. |
| `uncertainty` | Estimate uncertainty (TTA / MC-dropout / ensemble spread). |
| `pipeline` | Segment, then evaluate and estimate uncertainty in one command. |

Run anatomical segmentation on an input volume:

```bash
impact-seg-konfai segment body -i path/to/image.nii.gz -o ./Output/
```

Evaluate against a reference, or run everything at once:

```bash
impact-seg-konfai eval body -i image.nii.gz --gt reference_mask.nii.gz -o ./Output/
impact-seg-konfai pipeline body -i image.nii.gz --gt reference_mask.nii.gz --mask eval_mask.nii.gz --gpu 0 --tta 2 -uncertainty
```

### Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `MODEL` | Model name on `VBoussot/ImpactSeg` (e.g. `body`) — determines what is predicted | *required* |
| `-i`, `--inputs` | Input file(s) or a dataset directory | *required* |
| `-o`, `--output` | Output directory | `./Output/` |
| `--ensemble` | Number of models to ensemble (`segment` / `pipeline`) | `0` |
| `--tta` | Number of test-time augmentations (`segment` / `pipeline`) | `0` |
| `--mc` | Monte Carlo dropout samples (`segment` / `pipeline`) | `0` |
| `-uncertainty` | Also write the inference stack (`segment` / `pipeline`) | `False` |
| `--gt` | Reference labels — required by `eval`, optional in `pipeline` | *unset* |
| `--mask` | Evaluation mask(s) (`eval` / `pipeline`) | *unset* |
| `--gpu` | GPU id(s), e.g. `0` or `0 1` | CPU if unset |
| `--cpu` | Number of CPU worker processes | *unset* |
| `-q`, `--quiet` | Suppress console output | `False` |

> When `--ensemble`, `--tta`, and `--mc` are left at `0`, the values declared in the app bundle (`app.json`) are used.

See the full help of any sub-command with:

```bash
impact-seg-konfai segment --help
```

---

## 📚 References

If you use **IMPACT-Seg-KonfAI** in your work, please cite KonfAI along with the IMPACT-Seg materials associated with
the model release you use.

- Boussot, V., & Dillenseger, J.-L. (2025).
  **KonfAI: A Modular and Fully Configurable Framework for Deep Learning in Medical Imaging.**
  arXiv preprint [arXiv:2508.09823](https://arxiv.org/abs/2508.09823)

---

## ⚡ Performance & VRAM

Benchmarked on an **NVIDIA RTX PRO 5000 (24 GB)**, synthetic data, patch `[1, 192, 192]`, single model (`Mean`). The app **auto-selects the batch size from your free GPU VRAM** (`vram_plan`); override it in SlicerKonfAI (⚙ **Advanced**) or on the CLI with `--patch-size` / `--batch-size`.

| Free VRAM | Batch (auto) | Peak VRAM |
|:--|:--|:--|
| 8 GB  | 160 | ~7 GB |
| 16 GB | 320 | ~14 GB |
| 24 GB | 512 | ~22 GB |

Measured peak VRAM: batch 64 → 3.1 GB · 128 → 5.8 GB · 256 → 8.5 GB · 512 → 22.4 GB. Inference ≈ **16 s / case** on the benchmark volume (scales with the case size).

---

## 🔗 Links

- 🤗 **Model Hub:** [huggingface.co/VBoussot/ImpactSeg](https://huggingface.co/VBoussot/ImpactSeg)
- 📦 **PyPI Package:** [pypi.org/project/impact_seg_konfai](https://pypi.org/project/impact_seg_konfai)
- 🧠 **KonfAI Repository:** [github.com/vboussot/KonfAI](https://github.com/vboussot/KonfAI)
