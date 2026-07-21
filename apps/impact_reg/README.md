[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![PyPI version](https://img.shields.io/pypi/v/impact_reg_konfai.svg?color=blue)](https://pypi.org/project/impact_reg_konfai/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/fideus-labs/KonfAI/actions/workflows/konfai_ci.yml/badge.svg)](https://github.com/fideus-labs/KonfAI/actions/workflows/konfai_ci.yml)
[![Paper](https://img.shields.io/badge/📌%20Paper-KonfAI-blue)](https://www.arxiv.org/abs/2508.09823)

<p align="center">
  <img src="Logo.png" alt="IMPACT-Reg logo" width="220">
</p>

# IMPACT-Reg-KonfAI

**Fast and lightweight CLI for multimodal medical image registration using IMPACT-Reg presets within the KonfAI framework.**

---

## 🧩 Overview

**IMPACT-Reg-KonfAI** is the **command-line interface (CLI)** for running **IMPACT-Reg** registration presets published
in the [`VBoussot/ImpactReg`](https://huggingface.co/VBoussot/ImpactReg) Hugging Face repository, through the
[KonfAI](https://github.com/fideus-labs/KonfAI) deep learning framework.

**IMPACT-Reg** introduces a **semantic similarity metric** for **multimodal registration**, driven by deep features
extracted from large pretrained segmentation and foundation models (MIND, TotalSegmentator, MRSegmentator). It plugs
into an Elastix-based multi-resolution deformable pipeline to achieve robust cross-modality alignment while keeping
deformations smooth and physically plausible.

A registration run combines:

- **fixed** and **moving** images
- one or more **registration presets** resolved from the published preset database (each preset is a KonfAI app)
- optional **image**, **segmentation**, or **landmark** references (with an optional mask) for evaluation

---

## 🧠 Features

- ⚡ **Fast registration** powered by [KonfAI](https://github.com/fideus-labs/KonfAI)
- 🤗 **Automatic preset, parameter-map, and model download** from Hugging Face
- 🧩 **Multi-preset ensembling** (transforms averaged into a single displacement field)
- 🧠 **Semantic IMPACT metric** on deep features from pretrained segmentation / foundation models
- 📐 **Evaluation workflows** against image, segmentation, and landmark references
- 🧾 **Multi-format compatibility:** supports all major medical image formats handled by ITK

---

## 🗂️ Available presets

Presets are resolved dynamically from the published preset database (`PresetDatabase.json`) and passed as the first
positional argument(s). Current presets include generic rigid / rigid + BSpline strategies and IMPACT-driven
deformable presets tuned per modality pair (MR/CT, CBCT/CT) and anatomy (generic, head & neck).

List the presets exposed by your installation with:

```bash
impact-reg-konfai register --help
```

---

## 🚀 Installation

From PyPI:

```bash
python -m pip install impact-reg-konfai
```

From source:

```bash
git clone https://github.com/fideus-labs/KonfAI.git
python -m pip install -e apps/impact_reg
```

---

## ⚙️ Usage

The CLI is organised into sub-commands, matching the registration workflow:

| Sub-command | Purpose |
|---|---|
| `register` | Register a moving image onto a fixed image with one or more presets. Several presets are ensembled (their displacement fields are averaged). Writes the moved image, the displacement field (`DVF`), the transform, and the per-preset fields (kept for `uncertainty`). |
| `eval` | Evaluate a registration on any subset of modalities — image (MAE), segmentation (Dice), landmarks (TRE). At least one modality is required. |
| `uncertainty` | Voxel-wise spread map from an ensemble of displacement fields. |

Register a moving image onto a fixed image (ensemble several presets by listing them):

```bash
impact-reg-konfai register <PRESET> [<PRESET_2> ...] -f fixed.nii.gz -m moving.nii.gz -o ./Output --gpu 0
```

Evaluate a registration — any subset of modalities; the transform comes from a prior `register`:

```bash
impact-reg-konfai eval \
  --transform ./Output/P000/Transform.h5 \
  -f fixed.nii.gz -m moving.nii.gz --mask roi.nii.gz \
  --gt-fixed-seg fixed_seg.nii.gz --gt-moving-seg moving_seg.nii.gz \
  --gt-fixed-fid fixed.fcsv --gt-moving-fid moving.fcsv \
  -o ./Output --gpu 0
```

Estimate uncertainty from the per-preset displacement fields written by `register`:

```bash
impact-reg-konfai uncertainty --dvf ./Output/P000/Ensemble/*.mha -o ./Output/P000
```

### `register` arguments

| Flag | Description | Default |
|------|-------------|---------|
| `PRESET` | One or more presets from the published preset database (several are ensembled) | *required* |
| `-f`, `--fixed-images` | Fixed image(s), or a dataset directory | *required* |
| `-m`, `--moving-images` | Moving image(s), or a dataset directory | *required* |
| `-o`, `--output` | Output directory | `./Output/` |
| `--gpu` / `--cpu` | GPU id(s) / CPU worker processes | CPU if unset |
| `-q`, `--quiet` | Suppress console output | `False` |

### `eval` arguments — at least one modality required

| Flag | Description | Default |
|------|-------------|---------|
| `--transform` | Transform(s) from a prior `register` (identity if omitted) | *unset* |
| `-f`, `-m` | Fixed / moving images — image modality (MAE) | *unset* |
| `--gt-fixed-seg`, `--gt-moving-seg` | Fixed / moving segmentations — seg modality (Dice) | *unset* |
| `--gt-fixed-fid`, `--gt-moving-fid` | Fixed / moving landmarks — fid modality (TRE) | *unset* |
| `--mask` | Evaluation mask(s) for the image modality | *unset* |
| `--preset` | Preset providing the evaluation configs | first available |

### `uncertainty` arguments

| Flag | Description | Default |
|------|-------------|---------|
| `--dvf` | Two or more ensemble displacement fields (e.g. the per-preset fields from `register`) | *required* |
| `-o`, `--output` | Output directory | `./Output/` |

See the full help of any sub-command with:

```bash
impact-reg-konfai register --help
```

---

## 📦 Notes

- Available presets are resolved dynamically from the published IMPACT-Reg preset database.
- Multiple presets can be provided in one command; their displacement fields are averaged into a single field.
- The wrapper orchestrates the preset KonfAI apps (model inference), then ensembles, evaluates, and estimates uncertainty on their outputs.

---

## 📚 References

If you use **IMPACT-Reg-KonfAI** in your work, please cite KonfAI and the IMPACT-Reg paper.

- Boussot, V., Hémon, C., Nunes, J.-C., Dowling, J., Rouzé, S., Lafond, C., Barateau, A., & Dillenseger, J.-L.
  **IMPACT-Reg: A Generic Semantic Loss for Multimodal Medical Image Registration.**

- Boussot, V., & Dillenseger, J.-L. (2025).
  **KonfAI: A Modular and Fully Configurable Framework for Deep Learning in Medical Imaging.**
  arXiv preprint [arXiv:2508.09823](https://arxiv.org/abs/2508.09823)

---

## 🔗 Links

- 🤗 **Model Hub:** [huggingface.co/VBoussot/ImpactReg](https://huggingface.co/VBoussot/ImpactReg)
- 📦 **PyPI Package:** [pypi.org/project/impact_reg_konfai](https://pypi.org/project/impact_reg_konfai)
- 🧠 **KonfAI Repository:** [github.com/fideus-labs/KonfAI](https://github.com/fideus-labs/KonfAI)
