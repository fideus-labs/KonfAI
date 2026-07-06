[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![PyPI version](https://img.shields.io/pypi/v/impact_synth_konfai.svg?color=blue)](https://pypi.org/project/impact_synth_konfai/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/vboussot/KonfAI/actions/workflows/konfai_ci.yml/badge.svg)](https://github.com/vboussot/KonfAI/actions/workflows/konfai_ci.yml)
[![Paper](https://img.shields.io/badge/📌%20Paper-KonfAI-blue)](https://arxiv.org/abs/2510.21358)

<p align="center">
  <img src="Logo.png" alt="IMPACT-Synth logo" width="220">
</p>

# IMPACT-Synth-KonfAI

**Fast and lightweight CLI for synthetic CT generation using IMPACT-Synth models within the KonfAI framework.**

---

## 🧩 Overview

**IMPACT-Synth-KonfAI** is the **command-line interface (CLI)** for performing **inference** and **uncertainty estimation** with the *IMPACT-Synth* models.  
It provides a streamlined way to generate **synthetic CT (sCT) images** from MR or CBCT scans, leveraging the [KonfAI](https://github.com/vboussot/KonfAI) framework for efficient inference, test-time augmentation (TTA), model ensembling, and uncertainty quantification.  

The underlying **IMPACT-Synth** models are a family of **supervised convolutional neural networks (CNNs)** dedicated to **sCT generation**.
They build upon the research presented in **“Why Registration Quality Matters: Enhancing sCT Synthesis with IMPACT-Based Registration” (Boussot et al., 2025)**.  
These models are trained on **carefully aligned MR–CT pairs**, where alignment is optimized through the **IMPACT-Reg loss** to minimize spatial bias. Their training further integrates the **IMPACT-Synth loss**, a **perceptual loss derived from semantic representations of segmentation networks**. Together, **precise spatial alignment** and **semantic perceptual supervision** reinforce **anatomical fidelity** and **realistic tissue contrast** in the synthesized CT images.  

The official **IMPACT-Synth models** are available on [Hugging Face](https://huggingface.co/VBoussot/ImpactSynth) and can be executed directly through this CLI.

---

## 🚀 Installation

From PyPI:
```bash
python -m pip install impact-synth-konfai
```

From source:
```bash
git clone https://github.com/vboussot/KonfAI.git
python -m pip install -e apps/impact_synth
```
---

## ⚙️ Usage

The CLI is organised into sub-commands, mirroring the KonfAI Apps operations:

| Sub-command | Purpose |
|---|---|
| `synthesize` | Generate the synthetic CT (inference). |
| `eval` | Evaluate a synthetic CT against a reference CT. |
| `uncertainty` | Estimate uncertainty (TTA / MC-dropout / ensemble spread). |
| `pipeline` | Run synthesis, then evaluation and uncertainty in one command. |

Generate a synthetic CT:

```bash
impact-synth-konfai synthesize MR -i path/to/input.nii.gz -o ./Output/
```

Evaluate against a reference CT, or run everything at once:

```bash
impact-synth-konfai eval MR -i input.nii.gz --gt reference_ct.nii.gz -o ./Output/
impact-synth-konfai pipeline CBCT -i patient01.nii.gz --gt ct.nii.gz -o patient01 --gpu 0 --tta 2 --ensemble 5 -uncertainty
```

### Arguments

| Flag | Description | Default |
|------|--------------|----------|
| `MODEL` | Model name on Hugging Face (`MR` or `CBCT`) — determines what is predicted | *required* |
| `-i`, `--inputs` | Input file(s) or a dataset directory | *required* |
| `-o`, `--output` | Output directory | `./Output/` |
| `--ensemble` | Number of models to ensemble (`synthesize` / `pipeline`) | `0` |
| `--tta` | Number of test-time augmentations (`synthesize` / `pipeline`) | `0` |
| `--mc` | Monte Carlo dropout samples (`synthesize` / `pipeline`) | `0` |
| `-uncertainty` | Also write the inference stack (`synthesize` / `pipeline`) | `False` |
| `--gt` | Reference CT(s) — required by `eval`, optional in `pipeline` | *unset* |
| `--mask` | Evaluation mask(s) (`eval` / `pipeline`) | *unset* |
| `--gpu` | GPU id(s), e.g. `0` or `0 1` | CPU if unset |
| `--cpu` | Number of CPU worker processes | *unset* |
| `-q`, `--quiet` | Suppress console output | `False` |

---

## 🧠 Features

- ⚡ **Fast inference** powered by [KonfAI](https://github.com/vboussot/KonfAI)
- 🤗 **Automatic model download** from Hugging Face
- 🧩 **Multi-model ensembling** and **test-time augmentation (TTA)**
- 🧠 **Supports evaluation workflows with reference data, and uncertainty estimation without reference**
- 🧾 **Multi-format compatibility:** supports all major medical image formats handled by ITK

---

## 📚 References

If you use **IMPACT-Synth-KonfAI** in your work, please cite:

- Boussot, V., Hémon, C., Nunes, J.-C., & Dillenseger, J.-L. (2025).  
  **Why Registration Quality Matters: Enhancing sCT Synthesis with IMPACT-Based Registration.**  
  *arXiv preprint* [arXiv:2510.21358](https://arxiv.org/abs/2510.21358)

- Boussot, V., & Dillenseger, J.-L. (2025).  
  **KonfAI: A Modular and Fully Configurable Framework for Deep Learning in Medical Imaging**.  
  arXiv preprint [arXiv:2508.09823](https://arxiv.org/abs/2508.09823)

---

## ⚡ Performance & VRAM

Benchmarked on a single **NVIDIA RTX PRO 5000 (24 GB)** with a real whole-body MR (295 × 259 × 219, 2 mm), patch `[1, 512, 512]`. The app **auto-selects the batch size from your free GPU VRAM** (`vram_plan`); override it in SlicerKonfAI (⚙ **Advanced**) or on the CLI with `--patch-size` / `--batch-size`.

| Free VRAM | Batch (auto) | Peak VRAM | Time / case |
|:--|:--|:--|:--|
| 8 GB  | 16 | ~7.6 GB | — |
| 16 GB | 28 | ~15 GB  | — |
| 24 GB | 32 | ~16 GB  | **~24 s** |

Single-model sCT keeps **system RAM ~2 GB**. The plan leaves memory headroom — a larger batch saturates the card and slows inference (batch 48 → ~22 GB). A full **5-model ensemble** runs in ~82 s. Inference scales with the case size.

---

## 🔗 Links

- 🤗 **Model Hub:** [huggingface.co/VBoussot/ImpactSynth](https://huggingface.co/VBoussot/ImpactSynth)  
- 📦 **PyPI Package:** [pypi.org/project/impact_synth_konfai](https://pypi.org/project/impact_synth_konfai)  

---
