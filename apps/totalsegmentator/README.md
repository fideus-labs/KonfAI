[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![PyPI version](https://img.shields.io/pypi/v/totalsegmentator-konfai.svg?color=blue)](https://pypi.org/project/totalsegmentator-konfai/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/vboussot/KonfAI/actions/workflows/konfai_ci.yml/badge.svg)](https://github.com/vboussot/KonfAI/actions/workflows/konfai_ci.yml)
[![CI](https://github.com/vboussot/KonfAI/actions/workflows/konfai_apps_ci.yml/badge.svg)](https://github.com/vboussot/KonfAI/actions/workflows/konfai_apps_ci.yml)
[![Paper](https://img.shields.io/badge/📌%20Paper-KonfAI-blue)](https://www.arxiv.org/abs/2508.09823)

# TotalSegmentator-KonfAI

**Fast and lightweight CLI for whole-body CT or MRI segmentation using TotalSegmentator models within the KonfAI framework.**

---

## 🧩 Overview

**TotalSegmentator-KonfAI** is a lightweight **command-line interface (CLI)** for running **[TotalSegmentator](https://github.com/wasserth/TotalSegmentator)** models for **multi-organ medical image segmentation**, through the [KonfAI](https://github.com/vboussot/KonfAI) deep learning framework.

It provides **fast and efficient inference** for segmentation tasks, including on low-resource hardware. Pretrained models are automatically downloaded from [Hugging Face Hub](https://huggingface.co/VBoussot/TotalSegmentator-KonfAI).

---

## ⭐ Key Advantages

### 📦 Lightweight model distribution

- **~125 MB per model** 1.5 mm model
- 🔁 Compared to **~234 MB** per model for the original TotalSegmentator  
- **~66.2 MB** 3 mm models
- 🔁 Compared to **~135 MB** (original)

➡️ **Faster setup, smaller disk footprint**

---

## ⚡ Efficient inference

### 🔬 Performance comparison (single CT volume)

**Experimental setup**
- **Input volume size:** `512 × 512 × 366`
- **GPU:** NVIDIA RTX 6000
- **CPU:** Intel® Xeon® w5-3425

---

### Original TotalSegmentator

| Configuration | Time | Peak RAM | Peak VRAM |
|---------------|------|----------|------------|
| **Total – 5 models** | 82.37 s | 33.1 GB | ~4.7 GB |
| **Total 3 mm – 1 model** | 30.07 s | 30.6 GB | ~3.4 GB |

---

### TotalSegmentator-KonfAI

| Configuration | Time | Peak RAM | Peak VRAM |
|---------------|------|----------|------------|
| **Total – 5 models** | 61.55 s | 32.5 GB | ~4.3 GB |
| **Total 3 mm – 1 model** | 22.85 s | 10.5 GB | ~3.4 GB |

---

### 📈 Key observations

- **Faster inference times** compared to the original TotalSegmentator  
- **Significantly lower RAM usage for 3 mm models** (≈ 10.5 GB vs ≈ 30.6 GB)

---

## 🧠 Features

- ⚡ **Fast inference** powered by [KonfAI](https://github.com/vboussot/KonfAI)
- 🤗 **Automatic model download** from Hugging Face
- 🧠 **Supports evaluation workflows with reference data**
- 🧾 **Multi-format compatibility:** supports all major medical image formats handled by ITK

## 🚀 Installation

From PyPI:
```bash
python -m pip install totalsegmentator-konfai
```

From source:
```bash
git clone https://github.com/vboussot/KonfAI.git
python -m pip install -e apps/totalsegmentator
```

---

## ⚙️ Usage

The CLI is organised into sub-commands, mirroring the KonfAI Apps operations:

| Sub-command | Purpose |
|---|---|
| `segment` | Run the segmentation (inference). |
| `eval` | Evaluate a segmentation against a reference. |
| `pipeline` | Segment, then evaluate in one command. |

Perform segmentation on an input volume:

```bash
totalsegmentator-konfai segment total -i path/to/image.nii.gz -o ./Output/
```

Evaluate against a reference, or run both at once:

```bash
totalsegmentator-konfai eval total -i image.nii.gz --gt reference.nii.gz -o ./Output/
totalsegmentator-konfai pipeline total -i image.nii.gz --gt reference.nii.gz --gpu 0
```

### Arguments

| Flag | Description | Default |
|------|--------------|----------|
| `TASK` | Model on Hugging Face (`total`, `total_mr`, `total_3mm`, `total_mr_3mm`) — determines what is predicted | *required* |
| `-i`, `--inputs` | Input medical image(s) or a dataset directory | *required* |
| `-o`, `--output` | Output directory | `./Output/` |
| `--models` | Explicit model identifiers/paths to ensemble (`segment` / `pipeline`) | *unset* |
| `--gt` | Reference segmentation(s) — required by `eval`, optional in `pipeline` | *unset* |
| `--mask` | Evaluation mask(s) (`eval` / `pipeline`) | *unset* |
| `--gpu` | GPU id(s), e.g. `0` or `0 1` | CPU if unset |
| `--cpu` | Number of CPU worker processes | *unset* |
| `-q`, `--quiet` | Suppress console output | `False` |

> **Note:** TotalSegmentator models do not expose an uncertainty workflow, so there is no `uncertainty` sub-command.

---

## 📖 Reference

If you use **TotalSegmentator-KonfAI** in your work, please cite the original TotalSegmentator work in addition to this CLI tool.

- Wasserthal, J. *et al.* (2023).  
  **TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images.**  
  *Radiology: Artificial Intelligence*, 5(5). https://doi.org/10.1148/ryai.230024

- Akinci D’Antonoli, T. *et al.* (2025).  
  **TotalSegmentator MRI: Robust Sequence-independent Segmentation of Multiple Anatomic Structures in MRI.**  
  *Radiology*, 314(2). https://doi.org/10.1148/radiol.241613

- Boussot, V., & Dillenseger, J.-L. (2025).  
  **KonfAI: A Modular and Fully Configurable Framework for Deep Learning in Medical Imaging**.  
  arXiv preprint [arXiv:2508.09823](https://arxiv.org/abs/2508.09823)

---

## ⚡ Performance & VRAM

Benchmarked on an **NVIDIA RTX PRO 5000 (24 GB)**, synthetic data, patch `[96, 128, 160]`, 5-model ensemble (`Concat`), half precision (autocast). The app **auto-selects the batch size from your free GPU VRAM** (`vram_plan`); override it in SlicerKonfAI (⚙ **Advanced**) or on the CLI with `--patch-size` / `--batch-size`.

| Free VRAM | Batch (auto) | Peak VRAM |
|:--|:--|:--|
| 8 GB  | 4  | ~8 GB |
| 16 GB | 8  | ~15 GB |
| 24 GB | 12 | ~23 GB |

Measured peak VRAM (single model, half): batch 4 → 5.6 GB · 8 → 10.6 GB · 12 → 15.6 GB — the full 5-model ensemble adds ~15 %. Inference ≈ 20 s/model → **~110 s / case** for the full ensemble (scales with the case size).

---

## 🔗 Links

- 🧠 **Original TotalSegmentator:** [github.com/wasserth/TotalSegmentator](https://github.com/wasserth/TotalSegmentator)  
- 🤗 **Model Hub:** [huggingface.co/VBoussot/TotalSegmentator-KonfAI](https://huggingface.co/VBoussot/TotalSegmentator-KonfAI)  
- 📦 **PyPI Package:** [pypi.org/project/totalsegmentator-konfai](https://pypi.org/project/totalsegmentator-konfai)

---
