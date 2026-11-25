[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![PyPI version](https://img.shields.io/pypi/v/impact_seg_konfai.svg?color=blue)](https://pypi.org/project/impact_seg_konfai/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/fideus-labs/KonfAI/actions/workflows/konfai_ci.yml/badge.svg)](https://github.com/fideus-labs/KonfAI/actions/workflows/konfai_ci.yml)
[![Paper](https://img.shields.io/badge/📌%20Paper-KonfAI-blue)](https://www.arxiv.org/abs/2508.09823)

# IMPACT-Seg-KonfAI

**Fast and lightweight CLI for multimodal anatomical segmentation using IMPACT-Seg models within the KonfAI framework.**

---

## 🧩 Overview

**IMPACT-Seg-KonfAI** is a lightweight **command-line interface (CLI)** for running **IMPACT-Seg** models through the
[KonfAI](https://github.com/fideus-labs/KonfAI) deep learning framework.

It provides a simple way to perform **anatomical segmentation inference**, **evaluation**, **ensembling**, and
**uncertainty estimation** on medical image volumes.

The underlying **IMPACT-Seg** models are **multimodal anatomical segmentation** networks: **residual-encoder
U-Nets, 2.5D or 3D**. A single model segments **CBCT, MR, and CT** scans into a **consistent label space**,
enabling cross-modality workflows without per-modality retraining.

Pretrained models are automatically downloaded from
[Hugging Face Hub](https://huggingface.co/HCedric/IMPACTSeg).

---

## 🧠 Features

- ⚡ **Fast inference** powered by [KonfAI](https://github.com/fideus-labs/KonfAI)
- 🤗 **Automatic model download** from Hugging Face
- 🧩 **Multi-model ensembling** and **test-time augmentation (TTA)**
- 🧠 **Supports evaluation workflows with reference data, and uncertainty estimation without reference**
- 🧾 **Multi-format compatibility:** supports all major medical image formats handled by ITK

---

## 🗂️ Available models

Model identifiers are resolved dynamically from the [`HCedric/IMPACTSeg`](https://huggingface.co/HCedric/IMPACTSeg)
repository and passed as the first positional argument.

| Model | Modalities | Labels | Network | Weights |
|-------|------------|--------|---------|---------|
| `fine-3d` | CBCT · MR · CT | 126 | 3D, patch 128 × 128 × 128 at 1.5 × 1.5 × 2 mm | 5 folds |
| `fine-2.5d` | CBCT · MR · CT | 126 | 2.5D, 5 slices of 288 × 288 at 1.5 × 1.5 × 2 mm | 5 folds |
| `coarse` | CBCT · MR · CT | 11 | 3D, patch 128 × 128 × 128 at 3 mm | 1 model |

The models are [IMPACT-Seg](https://github.com/HEMONCedric/IMPACTSeg)'s, with its network and preprocessing; its
patch fusion, the label of the surest patch at each voxel, is the apps' own `patch_combine`, a class of their bundle for
KonfAI and an extension for KonfAI Edge.

The **fine** models predict **126 structures**: abdominal and thoracic organs, the five lung lobes, heart chambers and
great vessels, every vertebra and rib by level, pelvic and limb bones, trunk muscles and head-and-neck organs at risk.
`fine-3d` is the more accurate of the two, `fine-2.5d` the lighter. Both run their five folds as an ensemble;
`--ensemble 1` runs a single fold, five times faster. The label names are in the `terminology` of each app's
`app.json`.

The **`coarse`** model predicts **11 anatomical labels** spanning soft tissues, cavities, bones, and central
structures:

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
git clone https://github.com/fideus-labs/KonfAI.git
cd KonfAI
# konfai and konfai-apps must come from the same checkout: this app pins both to its own
# setuptools_scm version, which only exists on PyPI at a release tag.
python -m pip install -e . -e konfai-apps -e apps/impact_seg
```

---

## ⚙️ Usage

The CLI is organised into sub-commands, mirroring the KonfAI Apps operations:

| Sub-command | Purpose |
|---|---|
| `segment` | Run the segmentation (inference). |
| `eval` | Evaluate a segmentation against a reference. |
| `uncertainty` | Estimate uncertainty (TTA / ensemble spread). |
| `pipeline` | Segment, then evaluate and estimate uncertainty in one command. |

Run anatomical segmentation on an input volume:

```bash
impact-seg-konfai segment fine-3d -i path/to/image.nii.gz -o ./Output/ --gpu 0
impact-seg-konfai segment fine-3d -i path/to/image.nii.gz -o ./Output/ --gpu 0 --ensemble 1   # one fold
impact-seg-konfai segment coarse -i path/to/image.nii.gz -o ./Output/
```

Evaluate against a reference, or run everything at once:

```bash
impact-seg-konfai eval coarse -i image.nii.gz --gt reference_mask.nii.gz -o ./Output/
impact-seg-konfai pipeline fine-3d -i image.nii.gz --gt reference_mask.nii.gz --mask eval_mask.nii.gz --gpu 0 -uncertainty
```

### Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `MODEL` | Model name on `HCedric/IMPACTSeg` (e.g. `fine-3d`), which determines what is predicted | *required* |
| `-i`, `--inputs` | Input file(s) or a dataset directory | *required* |
| `-o`, `--output` | Output directory | `./Output/` |
| `--ensemble` | Number of models to ensemble (`segment` / `pipeline`) | `0` |
| `--tta` | Number of test-time augmentations (`segment` / `pipeline`) | `0` |
| `-uncertainty` | Also write the inference stack (`segment` / `pipeline`) | `False` |
| `--gt` | Reference labels: required by `eval`, optional in `pipeline` | *unset* |
| `--mask` | Evaluation mask(s) (`eval` / `pipeline`) | *unset* |
| `--gpu` | GPU id(s), e.g. `0` or `0 1` | CPU if unset |
| `--cpu` | Number of CPU worker processes | *unset* |
| `-q`, `--quiet` | Suppress console output | `False` |

> When `--ensemble` and `--tta` are left at `0`, the values declared in the app bundle (`app.json`) are used.
> The models take no `--tta`: their labels are sided (left and right kidney, ribs), which a
> mirrored copy swaps.

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

The models on a whole-body CT (512 × 512 × 531, 1.5 × 1.5 × 2 mm), single
**NVIDIA RTX PRO 5000 (24 GB)** (2026-11-20). Their batch size is the one each app's `Prediction.yml` sets.

| Model | Folds | Time | Peak RAM | Peak VRAM |
|:--|:--:|:--|:--|:--|
| `coarse` | 1 | **6 s** | 2.6 GB | 1.7 GB |
| `fine-2.5d` | 1 | **26 s** | 3.0 GB | 4.6 GB |
| `fine-2.5d` | 5 | **123 s** | 3.8 GB | 5.5 GB |
| `fine-3d` | 1 | **48 s** | 4.0 GB | 5.7 GB |
| `fine-3d` | 5 | **212 s** | 4.0 GB | 8.6 GB |

---

## 🔗 Links

- 🤗 **Model Hub:** [huggingface.co/HCedric/IMPACTSeg](https://huggingface.co/HCedric/IMPACTSeg)
- 📦 **PyPI Package:** [pypi.org/project/impact_seg_konfai](https://pypi.org/project/impact_seg_konfai)
- 🧠 **KonfAI Repository:** [github.com/fideus-labs/KonfAI](https://github.com/fideus-labs/KonfAI)
