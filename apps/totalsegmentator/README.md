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

### 🔬 Performance comparison

**Setup**
- **Input:** real whole-body CT, `295 × 259 × 219` (2 mm)
- **GPU:** single NVIDIA RTX PRO 5000 (24 GB)

| Tool (`total`, 5-model) | Time | Peak RAM | Peak VRAM |
|---|------|----------|-----------|
| **TotalSegmentator-KonfAI** | **~42 s** | ~19 GB | ~20 GB |
| Original TotalSegmentator | ~76 s | ~9 GB | ~7 GB |

### 📈 Key observations

- **~1.8× faster** whole-body inference (`total`, 5-model ensemble)
- The 117-class head keeps the accumulator on the host, so KonfAI trades higher system RAM for the speed-up

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

Benchmarked on a single **NVIDIA RTX PRO 5000 (24 GB)** with a real whole-body CT (295 × 259 × 219, 2 mm), patch `[96, 128, 160]`, 5-model ensemble (`total`), half precision (autocast). The app **auto-selects the batch size from your free GPU VRAM** (`vram_plan`); override it in SlicerKonfAI (⚙ **Advanced**) or on the CLI with `--patch-size` / `--batch-size`.

| Free VRAM | Batch (auto) | Peak VRAM | Time / case |
|:--|:--|:--|:--|
| 8 GB  | 2 | — | — |
| 16 GB | 4 | — | — |
| 24 GB | 4 | ~20 GB | **~42 s** |

The 5-model `total` head (117 classes) needs **~20 GB** for its forward, so the ensemble targets a **24 GB card** — on smaller cards use **`total-3mm`** (1 model, 3 mm). Its whole-volume accumulator is too large for the GPU, so reassembly runs on the host (~19 GB RAM). A larger batch saturates the card and *slows* inference. Inference scales with the case size.

---

## 🔗 Links

- 🧠 **Original TotalSegmentator:** [github.com/wasserth/TotalSegmentator](https://github.com/wasserth/TotalSegmentator)  
- 🤗 **Model Hub:** [huggingface.co/VBoussot/TotalSegmentator-KonfAI](https://huggingface.co/VBoussot/TotalSegmentator-KonfAI)  
- 📦 **PyPI Package:** [pypi.org/project/totalsegmentator-konfai](https://pypi.org/project/totalsegmentator-konfai)

---
