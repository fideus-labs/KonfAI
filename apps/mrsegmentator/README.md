[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![PyPI version](https://img.shields.io/pypi/v/mrsegmentator-konfai.svg?color=blue)](https://pypi.org/project/mrsegmentator-konfai/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/vboussot/KonfAI/actions/workflows/konfai_ci.yml/badge.svg)](https://github.com/vboussot/KonfAI/actions/workflows/konfai_ci.yml)
[![Paper](https://img.shields.io/badge/📌%20Paper-KonfAI-blue)](https://www.arxiv.org/abs/2508.09823)

# MRSegmentator-KonfAI 

**Fast and lightweight CLI for whole-body MRI segmentation using MRSegmentator models within the KonfAI framework.**

---

## 🧩 Overview

**MRSegmentator-KonfAI** is a lightweight **command-line interface (CLI)** for running **[MRSegmentator](https://github.com/hhaentze/MRSegmentator)** models through the [KonfAI](https://github.com/vboussot/KonfAI) deep learning framework.

It provides **fast and efficient inference** for whole-body MRI segmentation, including on low-resource hardware.  

Pretrained models are automatically downloaded from [Hugging Face Hub](https://huggingface.co/VBoussot/MRSegmentator-KonfAI).

## ⭐ Key Advantages

### 📦 Lightweight model distribution

- **~128 MB per model**, with up to **5 folds** available  
- Download **only the folds you need**  
- **Total size with 5 folds:** ~640 MB  
- 🔁 Compared to **~1.07 GB** for the original full MRSegmentator model distribution  

➡️ **Faster setup, smaller disk footprint**

---

## ⚡ Efficient inference

### 🔬 Performance comparison

Same input, same weights (5-fold ensemble), same PyTorch build (cu13.0), single
**NVIDIA RTX PRO 5000 (24 GB)**. Peak RAM = process-tree resident set; peak VRAM = over baseline.

| Case (voxels) | Tool | Time | Peak RAM | Peak VRAM |
|---|---|---|---|---|
| **S** — 249 × 246 × 246 | **KonfAI** | **14 s** | **6.0 GB** | 13.0 GB |
| | Original | 26 s | 8.6 GB | 3.7 GB |
| **M** — 533 × 390 × 177 | **KonfAI** | **25 s** | **7.5 GB** | 15.7 GB |
| | Original | 65 s | 14.6 GB | 5.3 GB |
| **L** — 512 × 512 × 531 | **KonfAI** | **120 s** | **6.2 GB** | 16.7 GB |
| | Original | 192 s | 37.5 GB | 14.6 GB |

### 📈 Key observations

- **1.6–2.6× faster** whole-body inference, **1.4–6.0× less host RAM**.
- The GPU-resident accumulator trades **more VRAM** for the speed and low host RAM,
  while streaming keeps it **bounded** — on the **large** case host RAM stays at
  **6.2 GB** where the original grows to **37.5 GB**.
- **Byte-identical** to KonfAI's own CPU reassembly path.

---

## 🧠 Features

- ⚡ **Fast inference** powered by [KonfAI](https://github.com/vboussot/KonfAI)
- 🤗 **Automatic model download** from Hugging Face
- 🧩 **Multi-model ensembling**
- 🧠 **Supports evaluation workflows with reference data, and uncertainty estimation without reference**
- 🧾 **Multi-format compatibility:** supports all major medical image formats handled by ITK

---

## 🚀 Installation

From PyPI:
```bash
python -m pip install mrsegmentator-konfai
```

From source:
```bash
git clone https://github.com/vboussot/KonfAI.git
python -m pip install -e apps/mrsegmentator
```

---

## ⚙️ Usage

The CLI is organised into sub-commands, mirroring the KonfAI Apps operations:

| Sub-command | Purpose |
|---|---|
| `segment` | Run the segmentation (inference). |
| `eval` | Evaluate a segmentation against a reference. |
| `uncertainty` | Estimate uncertainty (fold-ensemble spread). |
| `pipeline` | Segment, then evaluate and estimate uncertainty in one command. |

Run segmentation on an MRI scan:
```bash
mrsegmentator-konfai segment -i path/to/input.nii.gz -o ./Output/
```

Evaluate against a reference, or run everything at once:
```bash
mrsegmentator-konfai eval -i input.nii.gz --gt reference.nii.gz -o ./Output/
mrsegmentator-konfai pipeline -i input.nii.gz --gt reference.nii.gz --gpu 0 -f 3 -uncertainty
```

### Arguments

| Flag | Description | Default |
|------|--------------|----------|
| `-i`, `--inputs` | Input MRI volume(s) or a dataset directory | *required* |
| `-o`, `--output` | Output directory | `./Output/` |
| `-f`, `--folds` | Number of model folds to ensemble, 1–5 (`segment` / `pipeline`) | `2` |
| `-uncertainty` | Also write the inference stack (`segment` / `pipeline`) | `False` |
| `--gt` | Reference segmentation(s) — required by `eval`, optional in `pipeline` | *unset* |
| `--mask` | Evaluation mask(s) (`eval` / `pipeline`) | *unset* |
| `--gpu` | GPU id(s), e.g. `0` or `0 1` | CPU if unset |
| `--cpu` | Number of CPU worker processes | *unset* |
| `-q`, `--quiet` | Suppress console output | `False` |

---

## 📖 Reference

If you use **MRSegmentator-KonfAI** in your work, please cite the original MRSegmentator work in addition to this CLI tool.

- Häntze, H. *et al.* (2025).  
  **Segmenting Whole-Body MRI and CT for Multiorgan Anatomic Structure Delineation.**  
  *Radiology: Artificial Intelligence*, 7(6). https://doi.org/10.1148/ryai.240777

- Boussot, V., & Dillenseger, J.-L. (2025).  
  **KonfAI: A Modular and Fully Configurable Framework for Deep Learning in Medical Imaging**.  
  arXiv preprint [arXiv:2508.09823](https://arxiv.org/abs/2508.09823)

---

## ⚡ Performance & VRAM

Benchmarked on a single **NVIDIA RTX PRO 5000 (24 GB)** with a real whole-body MR (295 × 259 × 219, 2 mm), patch `[96, 128, 160]`, 5-fold ensemble, half precision (autocast). The app **auto-selects the batch size from your free GPU VRAM** (`vram_plan`); override it in SlicerKonfAI (⚙ **Advanced**) or on the CLI with `--patch-size` / `--batch-size`.

| Free VRAM | Batch (auto) | Peak VRAM | Time / case |
|:--|:--|:--|:--|
| 8 GB  | 4 | ~8 GB  | — |
| 16 GB | 8 | ~15 GB | — |
| 24 GB | 8 | ~22 GB | **~27 s** |

On a 24 GB card the accumulator stays **on the GPU**, keeping host RAM low with a **byte-identical** result. The plan stops short of filling the card — a still-larger batch (12 → ~24 GB) saturates the allocator and *slows* inference ~2× without running faster. Inference scales with the case size.

---

## 🔗 Links

- 🧠 **Original MRSegmentator:** [github.com/hhaentze/MRSegmentator](https://github.com/hhaentze/MRSegmentator)  
- 🤗 **Model Hub:** [huggingface.co/VBoussot/MRSegmentator-KonfAI](https://huggingface.co/VBoussot/MRSegmentator-KonfAI)  
- 📦 **PyPI Package:** [pypi.org/project/mrsegmentator-konfai](https://pypi.org/project/mrsegmentator-konfai)


